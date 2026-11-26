"""Multifidelity inversion: cost accounting, active surrogates, multilevel MCMC.

Experimental (added in 0.28).

Three independent tools that make a cheap model pay off *with an accuracy
check attached*. Speed without accuracy control gives a precise but wrong
posterior, so every tool reports what it cost and what it cannot guarantee.

Cost accounting
    A :class:`CostModel` holds the cost of one coarse and one fine evaluation,
    either **declared** by the user or **measured** as the median wall time of
    single calls (:func:`measure_cost_model`; the method is stored in the
    model). All work is converted to *fine-equivalent* units
    ``C = n_fine + n_coarse c_coarse / c_fine`` ([EQ-INV-MFX-001]) and
    efficiency is the effective sample size per fine-equivalent unit
    ([EQ-INV-MFX-002]). :func:`delayed_acceptance_cost` converts a
    :class:`~probstat.inference.delayed_acceptance.DelayedAcceptanceResult`
    and :func:`approximation_error_cost` an
    :class:`~probstat.inverse.approximation_error.ApproximationErrorModel`
    (training pairs plus posterior evaluations, with a break-even count,
    [EQ-INV-MFX-003]). No speed-up is claimed by this module: compare two
    reports measured on the same problem.

Active surrogate refinement
    :func:`active_surrogate_posterior` fits a surrogate of a scalar
    log-posterior ``f(x)`` on a design, then repeatedly adds the candidate
    maximising ``a(x) = exp(mu(x) - mu_ref) sigma^2(x)`` ([EQ-INV-MFX-004])
    -- posterior density (up to a constant) times the surrogate's predictive
    variance -- until a stopping rule fires ([EQ-INV-MFX-005]): budget of
    fine runs, small acquisition, or a held-out validation error below a
    tolerance. Stopping by budget is *not* convergence and is reported as
    such. The surrogate enters as a callable ``fit(points, values)`` returning
    an object with ``predict_with_variance(x) -> (mean, variance)``; the
    library's :func:`~probstat.surrogates.fit_gaussian_process` has exactly this
    interface, but ``inverse`` (layer 4) must not import ``surrogates`` (layer
    5), so the surrogate is injected by the caller.
    :func:`extrapolation_report` flags points outside the design's bounding box
    or farther from every design point than the design's largest
    nearest-neighbour spacing ([EQ-INV-MFX-006]); :func:`grid_posterior_error`
    measures the KL divergence and moment errors of a surrogate posterior
    against an exact one on a tensor grid ([EQ-INV-MFX-007]). The held-out
    points used for stopping are no longer an unbiased accuracy estimate; score
    the final surrogate on an independent set.

Multilevel MCMC
    :func:`multilevel_mcmc` estimates ``E_{pi_L}[Q]`` through the telescoping
    sum ``E_0 + sum_l (E_l - E_{l-1})`` ([EQ-INV-MFX-008]) for a hierarchy of
    posteriors ``pi_0, ..., pi_L`` on one parameter space (Dodwell, Ketelsen,
    Scheichl and Teckentrup 2015). The correction at level ``l >= 1`` runs a
    chain whose proposals are the (sub-sampled) states of a level ``l - 1``
    chain, accepted with
    ``min(1, [pi_l(x') pi_{l-1}(x)] / [pi_l(x) pi_{l-1}(x')])``
    ([EQ-INV-MFX-009]). The pair ``(Q_l(x_n), Q_{l-1}(z_n))`` with ``z_n`` the
    proposal of step ``n`` (accepted or not) is the coupled sample: when the
    levels agree the proposal is accepted, ``x_n = z_n`` and the difference has
    small variance.

    Marginals. The coarse sequence ``z_n`` is a stationary ``pi_{l-1}`` chain,
    so ``E Q_{l-1}(z)`` is the level ``l - 1`` expectation exactly. The fine
    chain is an independence sampler with proposal law ``pi_{l-1}``; it
    targets ``pi_l`` exactly **only if the proposals are independent**, which
    holds in the limit of large sub-sampling. For finite sub-sampling the
    proposals are correlated and the fine marginal is approximate. Each
    correction is therefore estimated by an independent run, so the estimator
    variance is the sum of the level variances ``V_l / ESS_l``, and a
    different-posteriors average is never taken. Verify the coupling against
    a case with known level posteriors before trusting it; the discretisation
    bias is ``E_L Q - E Q`` of the finest level and is not estimated here.

    Support: where ``pi_{l-1}`` is zero but ``pi_l`` positive the region is
    unreachable and the level ``l`` chain is not exact there (not checked).
    Levels use the same parameter ``x``; only the forward model differs.

Coarse-to-fine warm starts
    :func:`coarse_to_fine_warm_start` solves the regularised nonlinear least
    squares problem ``min 1/2 ||W (G_l(x) - y)||^2 + lambda^2/2 ||L (x - x_ref)||^2``
    on a hierarchy of discretisations ``G_0, ..., G_L`` with Gauss-Newton-Krylov
    ([EQ-INV-MFX-011]). The level ``l`` solution is mapped by a user-supplied
    prolongation ``P_l`` to the parameter space of level ``l + 1`` and used as
    its start. Work is reported per level and in fine-equivalent units
    (declared or measured level costs), next to a cold solve of the finest
    level from the same start. A saving is a measured outcome for the problem
    at hand, never assumed; the report can show a loss.

Sensitivity-error validation
    :func:`validate_sensitivities` compares ``J v`` and ``J' w`` (and the
    gradient of the misfit when data are given) of an approximate model with a
    fine model at held-out points ([EQ-INV-MFX-012]), together with the angle
    between the two gradients ([EQ-INV-MFX-013]), and states whether the
    approximation is fit for gradient-based use *under a declared tolerance on
    that sample*. This is an empirical statement about the probed points and
    directions, not a guarantee. :func:`surrogate_mean_forward` wraps the mean
    of a surrogate (for example the one returned by
    :func:`active_surrogate_posterior`) as a differentiable model whose
    derivatives are central differences of the mean.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal, Protocol

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference.delayed_acceptance import DelayedAcceptanceResult
from probstat.inference.mcmc import effective_sample_size
from probstat.inverse.approximation_error import ApproximationErrorModel
from probstat.inverse.gauss_newton import (
    CallbackForward,
    DifferentiableForward,
    GaussNewtonResult,
    gauss_newton_krylov,
)
from probstat.linalg import LinearOperator, Vector, as_vector
from probstat.monte_carlo.multilevel import optimal_level_allocation

LogFunction = Callable[[Vector], float]
Quantity = Callable[[Vector], float]

__all__ = [
    "ActiveSurrogateResult",
    "ApproximationErrorCostReport",
    "CostModel",
    "CostReport",
    "ExtrapolationReport",
    "GridPosteriorError",
    "MultilevelLevelResult",
    "MultilevelMCMCResult",
    "PredictiveSurrogate",
    "RefinementStep",
    "SensitivityErrorReport",
    "WarmStartLevel",
    "WarmStartResult",
    "active_surrogate_posterior",
    "approximation_error_cost",
    "coarse_to_fine_warm_start",
    "cost_normalised_ess",
    "declare_cost_model",
    "delayed_acceptance_cost",
    "extrapolation_report",
    "fine_equivalent_cost",
    "grid_posterior_error",
    "measure_cost_model",
    "multilevel_mcmc",
    "multilevel_sample_allocation",
    "surrogate_mean_forward",
    "validate_sensitivities",
]


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Cost of one coarse and one fine evaluation.

    Args:
        coarse_cost: Cost of one coarse evaluation (any consistent unit,
            for example seconds); positive.
        fine_cost: Cost of one fine evaluation, in the same unit, at least
            ``coarse_cost``.
        source: ``"declared"`` (stated by the user) or ``"measured"``.
        method: How the numbers were obtained, in words (for ``"measured"``
            the timer, the statistic and the number of timings).

    Equation:
        [EQ-INV-MFX-001].

    References:
        [EQ-INV-MFX-001], [REF-KAIPIO-SOMERSALO-2007].

    """

    coarse_cost: float
    fine_cost: float
    source: str
    method: str

    def __post_init__(self) -> None:
        """Validate the cost pair.

        Raises:
            ValueError: If a cost is not positive and finite, the coarse cost
                exceeds the fine cost, or ``source`` is unknown.

        References:
            [EQ-INV-MFX-001].

        """
        for label, value in (
            ("coarse_cost", self.coarse_cost),
            ("fine_cost", self.fine_cost),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        if self.coarse_cost > self.fine_cost:
            raise ValueError("coarse_cost must not exceed fine_cost")
        if self.source not in ("declared", "measured"):
            raise ValueError("source must be 'declared' or 'measured'")

    @property
    def ratio(self) -> float:
        """Return ``c_coarse / c_fine`` in ``(0, 1]``.

        Returns:
            The cost of a coarse evaluation in fine-equivalent units.

        References:
            [EQ-INV-MFX-001].

        """
        return self.coarse_cost / self.fine_cost


def declare_cost_model(
    coarse_cost: float, fine_cost: float, *, note: str = "declared by the caller"
) -> CostModel:
    """Return a cost model from declared costs.

    Args:
        coarse_cost: Cost of one coarse evaluation (positive).
        fine_cost: Cost of one fine evaluation (at least ``coarse_cost``).
        note: Where the numbers come from (stored as ``method``).

    Returns:
        A :class:`CostModel` with ``source="declared"``.

    Raises:
        ValueError: For invalid costs.

    Equation:
        [EQ-INV-MFX-001].

    References:
        [EQ-INV-MFX-001].

    """
    return CostModel(
        require_finite_real(coarse_cost, name="coarse_cost"),
        require_finite_real(fine_cost, name="fine_cost"),
        "declared",
        note,
    )


def measure_cost_model(
    fine: Callable[[Sequence[float]], object],
    coarse: Callable[[Sequence[float]], object],
    points: Sequence[Sequence[float]],
    *,
    repeats: int = 3,
    timer: Callable[[], float] | None = None,
) -> CostModel:
    """Measure the per-call cost of both models as the median wall time.

    Every point is evaluated ``repeats`` times by each model; each call is
    timed individually with ``timer`` (default ``time.perf_counter``) and the
    cost is the **median** of all timings, which is robust to scheduler
    outliers. Nothing is cached between calls. Fast models are below the
    timer resolution: use enough points or a coarser unit; a non-positive
    median raises.

    Args:
        fine: Fine model, called as ``fine(x)``.
        coarse: Coarse model.
        points: Non-empty list of parameter vectors to time.
        repeats: Timings per point and model (positive).
        timer: Zero-argument clock returning seconds.

    Returns:
        A :class:`CostModel` with ``source="measured"``.

    Raises:
        TypeError: For non-callables or a non-integer ``repeats``.
        ValueError: For empty ``points``, ``repeats < 1``, or a non-positive
            median, or a coarse cost above the fine cost.

    Equation:
        [EQ-INV-MFX-001].

    References:
        [EQ-INV-MFX-001].

    """
    if not callable(fine) or not callable(coarse):
        raise TypeError("fine and coarse must be callable")
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError("repeats must be an integer")
    if repeats < 1 or not points:
        raise ValueError("repeats must be positive and points non-empty")
    clock = timer if timer is not None else time.perf_counter
    cloud = [as_vector(p, name="point") for p in points]

    def median_time(model: Callable[[Sequence[float]], object]) -> float:
        """Return the median single-call wall time of one model.

        References:
            [EQ-INV-MFX-001].

        """
        timings: list[float] = []
        for point in cloud:
            for _ in range(repeats):
                start = clock()
                model(point)
                timings.append(clock() - start)
        timings.sort()
        mid = len(timings) // 2
        if len(timings) % 2:
            return timings[mid]
        return 0.5 * (timings[mid - 1] + timings[mid])

    name = getattr(clock, "__name__", "timer")
    return CostModel(
        median_time(coarse),
        median_time(fine),
        "measured",
        f"median of {len(cloud) * repeats} single-call timings per model "
        f"({name}), {len(cloud)} points x {repeats} repeats",
    )


def fine_equivalent_cost(
    fine_evaluations: int, coarse_evaluations: int, model: CostModel
) -> float:
    """Return ``n_fine + n_coarse c_coarse / c_fine``.

    Args:
        fine_evaluations: Non-negative number of fine calls.
        coarse_evaluations: Non-negative number of coarse calls.
        model: The cost model.

    Returns:
        Work in fine-equivalent units.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative.

    Equation:
        [EQ-INV-MFX-001].

    References:
        [EQ-INV-MFX-001].

    """
    for label, count in (
        ("fine_evaluations", fine_evaluations),
        ("coarse_evaluations", coarse_evaluations),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{label} must be an integer")
        if count < 0:
            raise ValueError(f"{label} must be non-negative")
    return fine_evaluations + coarse_evaluations * model.ratio


def cost_normalised_ess(effective_samples: float, cost: float) -> float:
    """Return the effective sample size per cost unit.

    Args:
        effective_samples: ESS (non-negative; ``nan`` is returned as ``nan``).
        cost: Positive cost in any consistent unit.

    Returns:
        ``ESS / cost``.

    Raises:
        ValueError: If ``cost`` is not positive and finite or ``ESS`` negative.

    Equation:
        [EQ-INV-MFX-002].

    References:
        [EQ-INV-MFX-002].

    """
    if not math.isfinite(cost) or cost <= 0.0:
        raise ValueError("cost must be positive and finite")
    if effective_samples < 0.0:
        raise ValueError("effective_samples must be non-negative")
    return effective_samples / cost


@dataclass(frozen=True, slots=True)
class CostReport:
    """Fine-equivalent cost and cost-normalised ESS of a delayed-acceptance run.

    Attributes:
        fine_evaluations: Exact-target calls.
        coarse_evaluations: Surrogate calls.
        fine_equivalent_cost: ``n_fine + n_coarse c_coarse / c_fine``.
        effective_sample_size: Smallest finite per-coordinate ESS (``nan`` if
            none is finite).
        ess_per_fine_equivalent: ESS per fine-equivalent unit.
        relative_efficiency: ``ess_per_fine_equivalent`` divided by the
            caller-supplied baseline (ESS per fine evaluation of a plain
            fine-model chain) or ``nan`` without a baseline; above 1 means
            delayed acceptance won on this problem.
        cost_source: ``"declared"`` or ``"measured"``.

    References:
        [EQ-INV-MFX-001], [EQ-INV-MFX-002].

    """

    fine_evaluations: int
    coarse_evaluations: int
    fine_equivalent_cost: float
    effective_sample_size: float
    ess_per_fine_equivalent: float
    relative_efficiency: float
    cost_source: str


def delayed_acceptance_cost(
    result: DelayedAcceptanceResult,
    model: CostModel,
    *,
    baseline_ess_per_fine_evaluation: float | None = None,
) -> CostReport:
    """Convert a delayed-acceptance result into fine-equivalent cost and ESS.

    ``fine_evaluations = result.exact_evaluations`` and ``coarse_evaluations
    = result.surrogate_evaluations`` (burn-in included, as in the result).

    Args:
        result: Output of ``delayed_acceptance_metropolis``.
        model: Costs of the surrogate (coarse) and exact (fine) target.
        baseline_ess_per_fine_evaluation: ESS per fine evaluation of an
            ordinary Metropolis chain on the fine target, measured by the
            caller, to form ``relative_efficiency``.

    Returns:
        A :class:`CostReport`.

    Raises:
        ValueError: If the baseline is not positive and finite.

    Equation:
        [EQ-INV-MFX-001], [EQ-INV-MFX-002].

    References:
        [EQ-INV-MFX-001], [EQ-INV-MFX-002], [REF-CHRISTEN-FOX-2005].

    """
    cost = fine_equivalent_cost(
        result.exact_evaluations, result.surrogate_evaluations, model
    )
    finite = [e for e in result.effective_sample_size if math.isfinite(e)]
    ess = min(finite) if finite else math.nan
    per_cost = cost_normalised_ess(ess, cost) if finite else math.nan
    relative = math.nan
    if baseline_ess_per_fine_evaluation is not None:
        if (
            not math.isfinite(baseline_ess_per_fine_evaluation)
            or baseline_ess_per_fine_evaluation <= 0.0
        ):
            raise ValueError("baseline must be positive and finite")
        relative = per_cost / baseline_ess_per_fine_evaluation
    return CostReport(
        fine_evaluations=result.exact_evaluations,
        coarse_evaluations=result.surrogate_evaluations,
        fine_equivalent_cost=cost,
        effective_sample_size=ess,
        ess_per_fine_equivalent=per_cost,
        relative_efficiency=relative,
        cost_source=model.source,
    )


@dataclass(frozen=True, slots=True)
class ApproximationErrorCostReport:
    """Cost of an approximation-error model setup in fine-equivalent units.

    Attributes:
        setup_fine_evaluations: Fine calls spent on the paired training set.
        setup_coarse_evaluations: Coarse calls spent on it.
        setup_cost: Setup cost in fine-equivalent units.
        posterior_evaluations: Likelihood evaluations in the posterior run.
        posterior_cost: Their cost with the coarse model.
        total_cost: ``setup_cost + posterior_cost``.
        fine_only_cost: Cost of the same number of evaluations with the fine
            model alone (no setup).
        saving: ``fine_only_cost - total_cost``; negative when the setup does
            not pay off.
        break_even_evaluations: ``setup_cost / (1 - c_coarse / c_fine)``, the
            number of posterior evaluations above which the coarse model with
            the error model is cheaper (``inf`` if the models cost the same).
        cost_source: ``"declared"`` or ``"measured"``.

    References:
        [EQ-INV-MFX-003].

    """

    setup_fine_evaluations: int
    setup_coarse_evaluations: int
    setup_cost: float
    posterior_evaluations: int
    posterior_cost: float
    total_cost: float
    fine_only_cost: float
    saving: float
    break_even_evaluations: float
    cost_source: str


def approximation_error_cost(
    error_model: ApproximationErrorModel,
    model: CostModel,
    *,
    posterior_evaluations: int = 0,
) -> ApproximationErrorCostReport:
    """Cost an approximation-error setup and compare it with fine-only work.

    The setup uses ``error_model.fine_evaluations`` fine and
    ``error_model.coarse_evaluations`` coarse calls (all pairs, including the
    held-out ones). The posterior run then uses only the coarse model.
    Accuracy is not part of this report: the error model only pays off if the
    posterior it gives is adequate, which must be checked separately.

    Args:
        error_model: A fitted ``ApproximationErrorModel``.
        model: Cost model.
        posterior_evaluations: Number of likelihood evaluations in the
            posterior computation (non-negative).

    Returns:
        An :class:`ApproximationErrorCostReport`.

    Raises:
        TypeError: If ``posterior_evaluations`` is not an integer.
        ValueError: If it is negative.

    Equation:
        [EQ-INV-MFX-003].

    References:
        [EQ-INV-MFX-003], [REF-KAIPIO-SOMERSALO-2007].

    """
    setup = fine_equivalent_cost(
        error_model.fine_evaluations, error_model.coarse_evaluations, model
    )
    posterior = fine_equivalent_cost(0, posterior_evaluations, model)
    fine_only = fine_equivalent_cost(posterior_evaluations, 0, model)
    gain = 1.0 - model.ratio
    return ApproximationErrorCostReport(
        setup_fine_evaluations=error_model.fine_evaluations,
        setup_coarse_evaluations=error_model.coarse_evaluations,
        setup_cost=setup,
        posterior_evaluations=posterior_evaluations,
        posterior_cost=posterior,
        total_cost=setup + posterior,
        fine_only_cost=fine_only,
        saving=fine_only - (setup + posterior),
        break_even_evaluations=setup / gain if gain > 0.0 else math.inf,
        cost_source=model.source,
    )


# ---------------------------------------------------------------------------
# Active surrogate refinement
# ---------------------------------------------------------------------------


class PredictiveSurrogate(Protocol):
    """Structural interface of a surrogate with a Gaussian predictive law.

    References:
        [EQ-INV-MFX-004], [REF-RASMUSSEN-WILLIAMS-2006].

    """

    def predict_with_variance(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the predictive mean and variance at ``x``.

        References:
            [EQ-INV-MFX-004]. No equation is evaluated by the protocol.

        """
        ...


SurrogateFit = Callable[[Sequence[Vector], Sequence[float]], PredictiveSurrogate]


@dataclass(frozen=True, slots=True)
class RefinementStep:
    """One iteration of the active refinement loop.

    Attributes:
        training_evaluations: Fine runs in the design before this step.
        max_acquisition: Largest acquisition value over the remaining
            candidates (``nan`` if none remained).
        selected: Candidate added by this step (``None`` for the last one).
        validation_rmse: RMSE of the surrogate mean on the held-out points
            (``None`` without validation points).

    References:
        [EQ-INV-MFX-004], [EQ-INV-MFX-005].

    """

    training_evaluations: int
    max_acquisition: float
    selected: Vector | None
    validation_rmse: float | None


@dataclass(frozen=True, slots=True)
class ExtrapolationReport:
    """Which points lie outside the region covered by a design.

    Attributes:
        outside_box: Per point, outside the axis-aligned bounding box.
        distance_to_design: Per point, Euclidean distance to the nearest
            design point.
        largest_spacing: Largest nearest-neighbour distance among design
            points (``0`` for a single point).
        extrapolating: ``outside_box`` or ``distance_to_design`` above
            ``distance_factor * largest_spacing``.
        weighted_fraction: Weight fraction of extrapolating points.

    References:
        [EQ-INV-MFX-006].

    """

    outside_box: tuple[bool, ...]
    distance_to_design: tuple[float, ...]
    largest_spacing: float
    extrapolating: tuple[bool, ...]
    weighted_fraction: float


@dataclass(frozen=True, slots=True)
class ActiveSurrogateResult:
    """Output of :func:`active_surrogate_posterior`.

    Attributes:
        design: Training inputs (initial design, then the refinements).
        values: Fine log-target values at the design.
        surrogate: The final fitted surrogate.
        training_evaluations: Fine runs on the design.
        validation_evaluations: Fine runs spent on held-out points.
        fine_evaluations: Their sum (the total fine cost).
        history: One :class:`RefinementStep` per fit.
        stop_reason: ``"acquisition_tolerance"``, ``"validation_tolerance"``,
            ``"budget"`` or ``"candidates_exhausted"``.
        converged: ``True`` only for the two accuracy-based reasons.
        validation_points: The held-out inputs (never trained on).

    References:
        [EQ-INV-MFX-004], [EQ-INV-MFX-005].

    """

    design: tuple[Vector, ...]
    values: tuple[float, ...]
    surrogate: PredictiveSurrogate
    training_evaluations: int
    validation_evaluations: int
    fine_evaluations: int
    history: tuple[RefinementStep, ...]
    stop_reason: str
    converged: bool
    validation_points: tuple[Vector, ...]

    def log_density(self, x: Sequence[float]) -> float:
        """Return the surrogate mean of the log-target at ``x``.

        Args:
            x: Parameter vector.

        Returns:
            ``mu(x)``; outside the design region this is an extrapolation,
            see :meth:`extrapolation`.

        References:
            [EQ-INV-MFX-004].

        """
        return self.surrogate.predict_with_variance(x)[0]

    def extrapolation(
        self,
        points: Sequence[Sequence[float]],
        *,
        weights: Sequence[float] | None = None,
        distance_factor: float = 1.0,
    ) -> ExtrapolationReport:
        """Report which ``points`` are extrapolations of this design.

        Args:
            points: Query points.
            weights: Non-negative weights (for example the surrogate
                posterior density at ``points``); uniform by default.
            distance_factor: See :func:`extrapolation_report`.

        Returns:
            An :class:`ExtrapolationReport`.

        References:
            [EQ-INV-MFX-006].

        """
        return extrapolation_report(
            self.design, points, weights=weights, distance_factor=distance_factor
        )


def _distance(a: Vector, b: Vector) -> float:
    """Return the Euclidean distance.

    References:
        [EQ-INV-MFX-006].

    """
    return math.sqrt(math.fsum((p - q) ** 2 for p, q in zip(a, b, strict=True)))


def extrapolation_report(
    design: Sequence[Sequence[float]],
    points: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
    distance_factor: float = 1.0,
) -> ExtrapolationReport:
    """Flag points that a surrogate trained on ``design`` extrapolates to.

    A point ``x`` is flagged when it lies outside the axis-aligned bounding
    box of the design, **or** when its distance to the nearest design point
    exceeds ``distance_factor * h`` with ``h`` the largest nearest-neighbour
    spacing among design points. This is a conservative geometric heuristic,
    not a statement about the surrogate's error: a flagged point may still
    be predicted well, an unflagged one badly.

    Args:
        design: Training inputs (at least one).
        points: Query points of the same dimension.
        weights: Optional non-negative weights, not all zero.
        distance_factor: Positive multiplier of ``h``.

    Returns:
        An :class:`ExtrapolationReport`.

    Raises:
        ValueError: For empty or mismatched inputs, or invalid weights.

    Equation:
        [EQ-INV-MFX-006].

    References:
        [EQ-INV-MFX-006].

    """
    cloud = [as_vector(p, name="design point") for p in design]
    query = [as_vector(p, name="point") for p in points]
    if not cloud or not query:
        raise ValueError("design and points must be non-empty")
    n = len(cloud[0])
    if any(len(p) != n for p in cloud + query):
        raise ValueError("design and points must share one dimension")
    if distance_factor <= 0.0 or not math.isfinite(distance_factor):
        raise ValueError("distance_factor must be positive and finite")
    if weights is None:
        w = [1.0] * len(query)
    else:
        w = [require_finite_real(v, name="weight") for v in weights]
        if len(w) != len(query) or any(v < 0.0 for v in w) or sum(w) <= 0.0:
            raise ValueError("weights must be non-negative, nonzero, one per point")
    lower = [min(p[j] for p in cloud) for j in range(n)]
    upper = [max(p[j] for p in cloud) for j in range(n)]
    spacing = 0.0
    if len(cloud) > 1:
        spacing = max(
            min(_distance(a, b) for k, b in enumerate(cloud) if k != i)
            for i, a in enumerate(cloud)
        )
    outside = tuple(
        any(not lower[j] <= p[j] <= upper[j] for j in range(n)) for p in query
    )
    nearest = tuple(min(_distance(p, c) for c in cloud) for p in query)
    flags = tuple(
        o or d > distance_factor * spacing
        for o, d in zip(outside, nearest, strict=True)
    )
    total = math.fsum(w)
    fraction = math.fsum(v for v, f in zip(w, flags, strict=True) if f) / total
    return ExtrapolationReport(outside, nearest, spacing, flags, fraction)


def _checked_log_target(function: object) -> LogFunction:
    """Wrap the fine log-target so that NaN, ``inf`` values raise.

    References:
        [EQ-INV-MFX-004]. No equation is evaluated.

    """
    if not callable(function):
        raise TypeError("log_target must be callable")

    def call(point: Vector) -> float:
        """Evaluate and validate one fine value.

        References:
            [EQ-INV-MFX-004]. No equation is evaluated.

        """
        value = float(function(point))
        if not math.isfinite(value):
            raise ValueError(
                "log_target must be finite on every evaluated point; restrict "
                f"the candidates to the support (got {value} at {point})"
            )
        return value

    return call


def active_surrogate_posterior(
    log_target: LogFunction,
    fit: SurrogateFit,
    initial_design: Sequence[Sequence[float]],
    candidates: Sequence[Sequence[float]],
    *,
    max_fine_evaluations: int,
    acquisition_tolerance: float = 0.0,
    validation_points: Sequence[Sequence[float]] = (),
    validation_tolerance: float | None = None,
) -> ActiveSurrogateResult:
    """Fit a surrogate of a fine log-posterior and refine it where it matters.

    Loop: fit the surrogate on the design; compute ``a(c) = exp(mu(c) -
    mu_ref) sigma^2(c)`` on every candidate not yet in the design, with
    ``mu_ref`` the largest predicted mean over those candidates
    ([EQ-INV-MFX-004]); stop if the held-out RMSE is at most
    ``validation_tolerance``, if no candidate is left, if ``max a <=
    acquisition_tolerance``, or if the budget is spent ([EQ-INV-MFX-005]);
    otherwise evaluate the fine target at the maximiser (first one on ties) and
    repeat. Only the first
    two of those stops count as ``converged``. The held-out points are
    evaluated once, never enter the design, and are removed from the
    candidates.

    Args:
        log_target: Expensive scalar log-posterior (up to a constant); must
            be finite at every evaluated point.
        fit: ``fit(points, values)`` returning an object with
            ``predict_with_variance``; for the library GP use
            ``lambda x, v: fit_gaussian_process(x, v, seed=0)``.
        initial_design: Starting design (at least two points).
        candidates: Finite candidate set (for example a grid or prior draws)
            from which refinements are chosen.
        max_fine_evaluations: Total budget of fine runs, validation points
            included.
        acquisition_tolerance: Non-negative threshold on ``max a`` in squared
            log-density units: at the posterior mode ``a = sigma^2`` is the
            squared relative error of the predicted density, so ``1e-4``
            means about 1% there. ``0`` disables the rule.
        validation_points: Held-out inputs for the stopping check; disjoint
            from the initial design. The RMSE is unweighted, so choose points
            where the posterior has mass (pilot draws), not in the far tails
            where a log-density surrogate is legitimately poor.
        validation_tolerance: Positive RMSE threshold (log-density units),
            or ``None``.

    Returns:
        An :class:`ActiveSurrogateResult`.

    Raises:
        TypeError: If ``fit`` is not callable or the budget is not an int.
        ValueError: For an empty or inconsistent design, a budget smaller
            than the initial design plus validation points, held-out points
            inside the design, non-finite fine values, invalid tolerances, or
            a failing surrogate fit (with the reason).

    Equation:
        [EQ-INV-MFX-004], [EQ-INV-MFX-005].

    References:
        [EQ-INV-MFX-004], [EQ-INV-MFX-005], [REF-RASMUSSEN-WILLIAMS-2006],
        [REF-LI-MARZOUK-2014].

    """
    exact = _checked_log_target(log_target)
    if not callable(fit):
        raise TypeError("fit must be callable")
    if isinstance(max_fine_evaluations, bool) or not isinstance(
        max_fine_evaluations, int
    ):
        raise TypeError("max_fine_evaluations must be an integer")
    if acquisition_tolerance < 0.0 or not math.isfinite(acquisition_tolerance):
        raise ValueError("acquisition_tolerance must be non-negative and finite")
    if validation_tolerance is not None and not (
        math.isfinite(validation_tolerance) and validation_tolerance > 0.0
    ):
        raise ValueError("validation_tolerance must be positive and finite")
    design = [as_vector(p, name="design point") for p in initial_design]
    held = [as_vector(p, name="validation point") for p in validation_points]
    pool = [as_vector(p, name="candidate") for p in candidates]
    if len(design) < 2:
        raise ValueError("initial_design needs at least two points")
    dimension = len(design[0])
    if any(len(p) != dimension for p in design + held + pool):
        raise ValueError("all points must share one dimension")
    if len(set(design)) != len(design):
        raise ValueError("initial_design must not repeat points")
    if set(design) & set(held):
        raise ValueError("validation points must not be in the initial design")
    if validation_tolerance is not None and not held:
        raise ValueError("validation_tolerance needs validation_points")
    if len(design) + len(held) > max_fine_evaluations:
        raise ValueError(
            "max_fine_evaluations is smaller than the initial design plus "
            "the validation points"
        )
    values = [exact(p) for p in design]
    held_values = [exact(p) for p in held]
    excluded = set(held)
    open_pool = [c for c in dict.fromkeys(pool) if c not in excluded]
    steps: list[RefinementStep] = []
    stop = "budget"
    while True:
        try:
            surrogate = fit(tuple(design), tuple(values))
        except (ValueError, ArithmeticError) as error:
            raise ValueError(f"surrogate fit failed: {error}") from error
        rmse: float | None = None
        if held:
            rmse = math.sqrt(
                math.fsum(
                    (surrogate.predict_with_variance(p)[0] - v) ** 2
                    for p, v in zip(held, held_values, strict=True)
                )
                / len(held)
            )
        taken = set(design)
        remaining = [c for c in open_pool if c not in taken]
        best: Vector | None = None
        top = math.nan
        if remaining:
            predictions = [surrogate.predict_with_variance(c) for c in remaining]
            reference = max(m for m, _ in predictions)
            scores = [math.exp(m - reference) * v for m, v in predictions]
            index = max(range(len(scores)), key=lambda i: (scores[i], -i))
            best, top = remaining[index], scores[index]
        if validation_tolerance is not None and rmse is not None:
            if rmse <= validation_tolerance:
                steps.append(RefinementStep(len(design), top, None, rmse))
                stop = "validation_tolerance"
                break
        if best is None:
            steps.append(RefinementStep(len(design), top, None, rmse))
            stop = "candidates_exhausted"
            break
        if top <= acquisition_tolerance:
            steps.append(RefinementStep(len(design), top, None, rmse))
            stop = "acquisition_tolerance"
            break
        if len(design) + len(held) >= max_fine_evaluations:
            steps.append(RefinementStep(len(design), top, None, rmse))
            stop = "budget"
            break
        steps.append(RefinementStep(len(design), top, best, rmse))
        design.append(best)
        values.append(exact(best))
    return ActiveSurrogateResult(
        design=tuple(design),
        values=tuple(values),
        surrogate=surrogate,
        training_evaluations=len(design),
        validation_evaluations=len(held),
        fine_evaluations=len(design) + len(held),
        history=tuple(steps),
        stop_reason=stop,
        converged=stop in ("acquisition_tolerance", "validation_tolerance"),
        validation_points=tuple(held),
    )


@dataclass(frozen=True, slots=True)
class GridPosteriorError:
    """Accuracy of a surrogate posterior against an exact one on a grid.

    Attributes:
        kl_divergence: ``KL(p_exact || p_surrogate)`` (``inf`` if the
            surrogate has zero mass where the exact posterior has some).
        mean_exact: Exact posterior mean per coordinate.
        mean_surrogate: Surrogate posterior mean.
        mean_error: ``|mean_surrogate - mean_exact|`` per coordinate.
        std_exact: Exact marginal standard deviations.
        std_surrogate: Surrogate marginal standard deviations.
        std_error: Absolute differences of the standard deviations.
        grid_points: Number of grid nodes.

    References:
        [EQ-INV-MFX-007].

    """

    kl_divergence: float
    mean_exact: Vector
    mean_surrogate: Vector
    mean_error: Vector
    std_exact: Vector
    std_surrogate: Vector
    std_error: Vector
    grid_points: int


def _trapezoid_weights(axis: Sequence[float]) -> list[float]:
    """Return trapezoid quadrature weights of a strictly increasing axis.

    References:
        [EQ-INV-MFX-007].

    """
    if len(axis) < 2 or any(b <= a for a, b in zip(axis, axis[1:], strict=False)):
        raise ValueError("each axis needs >= 2 strictly increasing nodes")
    weights = [0.0] * len(axis)
    for i in range(len(axis) - 1):
        h = axis[i + 1] - axis[i]
        weights[i] += 0.5 * h
        weights[i + 1] += 0.5 * h
    return weights


def grid_posterior_error(
    log_exact: LogFunction,
    log_surrogate: LogFunction,
    axes: Sequence[Sequence[float]],
) -> GridPosteriorError:
    """Compare two unnormalised log-densities on a tensor grid.

    Both densities are normalised by trapezoid quadrature on the product of
    ``axes``; then ``KL(p || q) = sum_i w_i p_i log(p_i / q_i)`` and the
    means and marginal standard deviations are computed from the same
    weights. The result is only meaningful if the grid covers the posterior
    mass and is fine enough; quadrature error is not estimated. Where
    ``log_exact = -inf`` the exact density is zero.

    Args:
        log_exact: Exact log-density (up to a constant).
        log_surrogate: Surrogate log-density (up to a constant); finite.
        axes: One strictly increasing node list per coordinate.

    Returns:
        A :class:`GridPosteriorError`.

    Raises:
        ValueError: For invalid axes, a non-finite surrogate value, or an
            exact density that is zero or non-finite everywhere.

    Equation:
        [EQ-INV-MFX-007].

    References:
        [EQ-INV-MFX-007].

    """
    grids = [tuple(float(v) for v in axis) for axis in axes]
    if not grids:
        raise ValueError("axes must be non-empty")
    axis_weights = [_trapezoid_weights(g) for g in grids]
    nodes: list[Vector] = []
    weights: list[float] = []
    for index in product(*(range(len(g)) for g in grids)):
        nodes.append(tuple(g[i] for g, i in zip(grids, index, strict=True)))
        weights.append(
            math.prod(w[i] for w, i in zip(axis_weights, index, strict=True))
        )
    lp = [log_exact(x) for x in nodes]
    lq = [log_surrogate(x) for x in nodes]
    if any(math.isnan(v) or v == math.inf for v in lp + lq):
        raise ValueError("log densities must not be NaN or +inf")
    if any(v == -math.inf for v in lq):
        raise ValueError("the surrogate log-density must be finite on the grid")
    finite_p = [v for v in lp if v > -math.inf]
    if not finite_p:
        raise ValueError("the exact density vanishes everywhere on the grid")
    max_p, max_q = max(finite_p), max(lq)
    zp = math.fsum(w * math.exp(v - max_p) for w, v in zip(weights, lp, strict=True))
    zq = math.fsum(w * math.exp(v - max_q) for w, v in zip(weights, lq, strict=True))
    log_zp, log_zq = max_p + math.log(zp), max_q + math.log(zq)
    kl = math.fsum(
        w * math.exp(a - log_zp) * ((a - log_zp) - (b - log_zq))
        for w, a, b in zip(weights, lp, lq, strict=True)
        if a > -math.inf
    )
    dim = len(grids)

    def moments(logs: list[float], log_z: float) -> tuple[Vector, Vector]:
        """Return the mean and marginal standard deviation of a grid law.

        References:
            [EQ-INV-MFX-007].

        """
        mass = [w * math.exp(v - log_z) for w, v in zip(weights, logs, strict=True)]
        mean = tuple(
            math.fsum(m * x[j] for m, x in zip(mass, nodes, strict=True))
            for j in range(dim)
        )
        std = tuple(
            math.sqrt(
                math.fsum(
                    m * (x[j] - mean[j]) ** 2 for m, x in zip(mass, nodes, strict=True)
                )
            )
            for j in range(dim)
        )
        return mean, std

    mean_p, std_p = moments(lp, log_zp)
    mean_q, std_q = moments(lq, log_zq)
    return GridPosteriorError(
        kl_divergence=max(kl, 0.0),
        mean_exact=mean_p,
        mean_surrogate=mean_q,
        mean_error=tuple(abs(a - b) for a, b in zip(mean_q, mean_p, strict=True)),
        std_exact=std_p,
        std_surrogate=std_q,
        std_error=tuple(abs(a - b) for a, b in zip(std_q, std_p, strict=True)),
        grid_points=len(nodes),
    )


# ---------------------------------------------------------------------------
# Multilevel MCMC
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultilevelLevelResult:
    """Statistics of one level of the telescoping estimator.

    Attributes:
        level: Level ``l`` (0 is the coarsest posterior).
        samples: Number of kept coupled samples ``N_l``.
        mean: Estimate of ``E_l Q_l - E_{l-1} Q_{l-1}`` (``E_0 Q_0`` for
            level 0).
        variance: Sample variance ``V_l`` of the coupled differences.
        effective_sample_size: ESS of the difference series (``nan`` if the
            series is constant).
        mcse: Monte Carlo standard error ``sqrt(V_l / ESS_l)`` (``0`` for a
            constant series).
        acceptance_rate: Fraction of coupled proposals accepted (level 0:
            random-walk acceptance).
        evaluations: Evaluations of ``pi_k`` made by this level's chain stack,
            ``evaluations[k]`` for ``k = 0..l``, burn-in included.
        cost: Total cost in fine-equivalent units.
        cost_per_sample: ``cost / (burn_in + samples)``.
        coarse_mean: Mean of ``Q_{l-1}`` over the coarse proposals
            (``nan`` for level 0).

    References:
        [EQ-INV-MFX-008], [EQ-INV-MFX-009], [EQ-INV-MFX-010].

    """

    level: int
    samples: int
    mean: float
    variance: float
    effective_sample_size: float
    mcse: float
    acceptance_rate: float
    evaluations: tuple[int, ...]
    cost: float
    cost_per_sample: float
    coarse_mean: float


@dataclass(frozen=True, slots=True)
class MultilevelMCMCResult:
    """Combined multilevel MCMC estimate.

    Attributes:
        estimate: ``sum_l mean_l``, the estimate of ``E_{pi_L}[Q]`` up to the
            fine-level bias of the sub-sampling approximation.
        standard_error: ``sqrt(sum_l mcse_l^2)`` (independent level runs).
        levels: One :class:`MultilevelLevelResult` per level.
        total_cost: Sum of level costs in fine-equivalent units.
        warnings: Human-readable diagnostics (low acceptance, small ESS).
        seed: Seed of the random stream.

    References:
        [EQ-INV-MFX-008].

    """

    estimate: float
    standard_error: float
    levels: tuple[MultilevelLevelResult, ...]
    total_cost: float
    warnings: tuple[str, ...]
    seed: int | None


class _Chain:
    """One level of the nested chain stack (Metropolis, or coupled proposals).

    Level 0 is a Gaussian random walk. Level ``k >= 1`` advances its parent
    (level ``k - 1``) ``subsampling`` steps and proposes the parent's state.

    References:
        [EQ-INV-MFX-009].

    """

    def __init__(
        self,
        level: int,
        densities: Sequence[LogFunction],
        initial: Vector,
        scales: Vector,
        subsampling: Sequence[int],
        streams: Sequence[RandomStream],
        counts: list[int],
    ) -> None:
        """Build the chain for ``level`` and, recursively, its parent.

        References:
            [EQ-INV-MFX-009].

        """
        self.level = level
        self._densities = densities
        self._scales = scales
        self._subsampling = subsampling
        self._counts = counts
        self._generator = streams[level].generator()
        self.parent: _Chain | None = None
        if level > 0:
            self.parent = _Chain(
                level - 1, densities, initial, scales, subsampling, streams, counts
            )
        self.state = initial
        self.own = 0.0
        self.below = 0.0
        if self.parent is not None:
            self.below = self.parent.own
        self.own = self._evaluate(level, initial)
        if not (math.isfinite(self.own) and math.isfinite(self.below)):
            raise ValueError(
                f"the initial point must have finite log-density at level {level}"
            )
        self.proposal: Vector = initial
        self.accepted = 0
        self.steps = 0

    def _evaluate(self, level: int, point: Vector) -> float:
        """Evaluate and count ``log pi_level``.

        References:
            [EQ-INV-MFX-009].

        """
        value = float(self._densities[level](point))
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"log-density of level {level} returned {value}")
        self._counts[level] += 1
        return value

    def step(self) -> bool:
        """Advance one step; return whether the proposal was accepted.

        Returns:
            Acceptance flag of this step.

        References:
            [EQ-INV-MFX-009].

        """
        self.steps += 1
        generator = self._generator
        if self.parent is None:
            moved = tuple(
                x + s * generator.gauss(0.0, 1.0)
                for x, s in zip(self.state, self._scales, strict=True)
            )
            self.proposal = moved
            value = self._evaluate(0, moved)
            if value > -math.inf and (
                math.log(1.0 - generator.random()) < value - self.own
            ):
                self.state, self.own = moved, value
                self.accepted += 1
                return True
            return False
        for _ in range(self._subsampling[self.level - 1]):
            self.parent.step()
        moved = self.parent.state
        coarse_value = self.parent.own
        self.proposal = moved
        value = self._evaluate(self.level, moved)
        if value > -math.inf and (
            math.log(1.0 - generator.random())
            < (value - coarse_value) - (self.own - self.below)
        ):
            self.state, self.own, self.below = moved, value, coarse_value
            self.accepted += 1
            return True
        return False


def _positive_int(value: object, name: str, *, minimum: int) -> int:
    """Return an integer at least ``minimum``.

    References:
        [EQ-INV-MFX-008]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _expand(
    value: int | Sequence[int], length: int, name: str, *, minimum: int
) -> tuple[int, ...]:
    """Broadcast an int or validate a sequence of ints to ``length``.

    References:
        [EQ-INV-MFX-008]. No equation is evaluated.

    """
    if isinstance(value, int):
        return (_positive_int(value, name, minimum=minimum),) * length
    items = tuple(_positive_int(v, name, minimum=minimum) for v in value)
    if len(items) != length:
        raise ValueError(f"{name} must be an int or have {length} entries")
    return items


def multilevel_mcmc(
    log_densities: Sequence[LogFunction],
    quantity: Quantity | Sequence[Quantity],
    initial: Sequence[float],
    *,
    samples: int | Sequence[int],
    step: float | Sequence[float] = 1.0,
    burn_in: int | Sequence[int] = 0,
    subsampling: int | Sequence[int] = 1,
    costs: Sequence[float] | None = None,
    seed: int | RandomStream | None = None,
) -> MultilevelMCMCResult:
    """Estimate ``E_{pi_L}[Q]`` by multilevel MCMC with coupled chains.

    ``log_densities[l]`` is ``log pi_l`` up to a constant, ``l = 0..L`` from
    coarse to fine, on one parameter space. Level 0 is a random-walk
    Metropolis chain. For ``l >= 1`` an independent run builds a nested stack
    of chains ``0..l``; the level ``l`` chain advances level ``l - 1`` by
    ``subsampling[l-1]`` steps per iteration and proposes its state ``z``,
    accepted with ``min(1, [pi_l(z) pi_{l-1}(x)] / [pi_l(x) pi_{l-1}(z)])``
    ([EQ-INV-MFX-009]). The coupled sample is ``Y = Q_l(x_n) - Q_{l-1}(z_n)``,
    the proposal ``z_n`` taken whether accepted or not, which keeps the coarse
    marginal ``pi_{l-1}``. The estimate is ``sum_l mean(Y_l)`` with standard
    error ``sqrt(sum_l V_l / ESS_l)`` ([EQ-INV-MFX-008]). See the module
    docstring for what exactness requires (independent proposals,
    sub-sampling large enough) and the support policy.

    Costs: the evaluations of every ``pi_k`` made by level ``l``'s run are
    counted (burn-in included) and converted to fine-equivalent units with
    ``costs`` (cost of one evaluation of each ``pi_l``; the finest cost is the
    unit; default: equal costs, which understates the saving) ([EQ-INV-MFX-010]).

    Args:
        log_densities: At least two log-density callables.
        quantity: ``Q`` (one callable used on every level) or one callable
            per level (``Q_l``).
        initial: Start point, finite density at every level.
        samples: Kept samples per level (int or one per level, each >= 4).
        step: Random-walk scale of the level 0 chain (scalar or per
            coordinate, positive).
        burn_in: Discarded steps (int or one per level) of each level's chain.
        subsampling: Level ``l - 1`` steps between proposals of level ``l``
            (int or one per correction level ``1..L``, each >= 1).
        costs: Positive cost of one evaluation of each ``pi_l``.
        seed: Integer seed or stream; the levels use independent children.

    Returns:
        A :class:`MultilevelMCMCResult`.

    Raises:
        TypeError: For non-callables or non-integer counts.
        ValueError: For fewer than two levels, wrong lengths, invalid
            ``step`` or ``costs``, a non-finite start, or a NaN or ``+inf``
            log-density.

    Equation:
        [EQ-INV-MFX-008], [EQ-INV-MFX-009], [EQ-INV-MFX-010].

    References:
        [EQ-INV-MFX-008], [EQ-INV-MFX-009], [EQ-INV-MFX-010],
        [REF-DODWELL-ETAL-2015], [REF-GILES-2015].

    """
    if len(log_densities) < 2:
        raise ValueError("at least two levels are required")
    levels = len(log_densities)
    if not all(callable(d) for d in log_densities):
        raise TypeError("log_densities must be callable")
    if callable(quantity):
        quantities: list[Quantity] = [quantity] * levels
    else:
        quantities = list(quantity)
        if len(quantities) != levels or not all(callable(q) for q in quantities):
            raise ValueError("quantity must be callable or one callable per level")
    counts_n = _expand(samples, levels, "samples", minimum=4)
    burns = _expand(burn_in, levels, "burn_in", minimum=0)
    sub = _expand(subsampling, levels - 1, "subsampling", minimum=1)
    start = as_vector(initial, name="initial")
    if isinstance(step, Sequence):
        scales = tuple(require_finite_real(s, name="step") for s in step)
        if len(scales) != len(start):
            raise ValueError("step must be a scalar or have one entry per coordinate")
    else:
        scales = (require_finite_real(step, name="step"),) * len(start)
    if any(s <= 0.0 for s in scales):
        raise ValueError("step entries must be positive")
    if costs is None:
        unit_costs = (1.0,) * levels
    else:
        unit_costs = tuple(require_finite_real(c, name="costs") for c in costs)
        if len(unit_costs) != levels or any(c <= 0.0 for c in unit_costs):
            raise ValueError("costs must be positive, one per level")
    stream = resolve_stream(seed)

    results: list[MultilevelLevelResult] = []
    notes: list[str] = []
    for level in range(levels):
        run = stream.child("level", level)
        stack = [run.child("chain", k) for k in range(level + 1)]
        evals = [0] * levels
        chain = _Chain(level, log_densities, start, scales, sub, stack, evals)
        for _ in range(burns[level]):
            chain.step()
        chain.accepted = 0
        chain.steps = 0
        series: list[float] = []
        coarse_values: list[float] = []
        for _ in range(counts_n[level]):
            chain.step()
            fine_value = float(quantities[level](chain.state))
            if level == 0:
                series.append(fine_value)
            else:
                coarse_value = float(quantities[level - 1](chain.proposal))
                coarse_values.append(coarse_value)
                series.append(fine_value - coarse_value)
        n = len(series)
        if not all(math.isfinite(v) for v in series):
            raise ValueError("quantity returned a non-finite value")
        mean = math.fsum(series) / n
        variance = math.fsum((v - mean) ** 2 for v in series) / (n - 1)
        if variance > 0.0:
            ess = effective_sample_size([series])
            mcse = math.sqrt(variance / ess) if ess > 0.0 else math.inf
        else:
            ess, mcse = math.nan, 0.0
        accept = chain.accepted / chain.steps
        total_evals = evals[: level + 1]
        cost = math.fsum(
            e * c / unit_costs[-1]
            for e, c in zip(total_evals, unit_costs, strict=False)
        )
        if accept < 0.1:
            notes.append(f"level {level}: acceptance rate {accept:.3f} below 0.1")
        if math.isfinite(ess) and ess < 30.0:
            notes.append(f"level {level}: effective sample size {ess:.1f} below 30")
        results.append(
            MultilevelLevelResult(
                level=level,
                samples=n,
                mean=mean,
                variance=variance,
                effective_sample_size=ess,
                mcse=mcse,
                acceptance_rate=accept,
                evaluations=tuple(total_evals),
                cost=cost,
                cost_per_sample=cost / (burns[level] + n),
                coarse_mean=math.fsum(coarse_values) / n if coarse_values else math.nan,
            )
        )
    return MultilevelMCMCResult(
        estimate=math.fsum(r.mean for r in results),
        standard_error=math.sqrt(math.fsum(r.mcse**2 for r in results)),
        levels=tuple(results),
        total_cost=math.fsum(r.cost for r in results),
        warnings=tuple(notes),
        seed=stream.seed,
    )


def multilevel_sample_allocation(
    result: MultilevelMCMCResult, target_variance: float
) -> tuple[int, ...]:
    """Return cost-optimal chain lengths for a target estimator variance.

    Each level's variance per iteration is ``V_l N_l / ESS_l`` (its
    autocorrelation-inflated variance) and its cost per iteration is
    ``cost_per_sample``; Giles' allocation
    :func:`~probstat.monte_carlo.multilevel.optimal_level_allocation` then
    gives ``N_l`` for ``sum_l V_l tau_l / N_l = target_variance``
    ([EQ-INV-MFX-010]). Levels with a constant series get zero.

    Args:
        result: A :class:`MultilevelMCMCResult` (pilot run).
        target_variance: Positive target variance of the combined estimator.

    Returns:
        Suggested number of kept iterations per level.

    Raises:
        ValueError: If ``target_variance`` is not positive.

    Equation:
        [EQ-INV-MFX-010].

    References:
        [EQ-INV-MFX-010], [REF-GILES-2015].

    """
    per_iteration = [
        r.variance * r.samples / r.effective_sample_size
        if math.isfinite(r.effective_sample_size)
        else 0.0
        for r in result.levels
    ]
    return optimal_level_allocation(
        per_iteration, [r.cost_per_sample for r in result.levels], target_variance
    )


# ---------------------------------------------------------------------------
# Coarse-to-fine optimisation warm starts
# ---------------------------------------------------------------------------

Prolongation = Callable[[Vector], Sequence[float]]


@dataclass(frozen=True, slots=True)
class WarmStartLevel:
    """Work and outcome of the solve at one level of the hierarchy.

    Attributes:
        level: Level index, ``0`` the coarsest.
        dimension: Parameter dimension of the level.
        initial: Start of the level (the prolongated solution of the level
            below, or the user start at level 0).
        result: The :class:`~probstat.inverse.gauss_newton.GaussNewtonResult`.
        iterations: Accepted Gauss-Newton steps.
        model_evaluations: ``value + derivative_value + c_d (jvp + vjp)`` with
            ``c_d`` the declared ``derivative_cost``, in units of one value
            call *of this level*.
        cost_ratio: ``c_l / c_L``, the cost of one call at this level in
            finest-level units.
        cost: ``model_evaluations * cost_ratio``, fine-equivalent work.

    References:
        [EQ-INV-MFX-011].

    """

    level: int
    dimension: int
    initial: Vector
    result: GaussNewtonResult
    iterations: int
    model_evaluations: float
    cost_ratio: float
    cost: float


@dataclass(frozen=True, slots=True)
class WarmStartResult:
    """Output of :func:`coarse_to_fine_warm_start`.

    Attributes:
        estimate: Final iterate of the last level solved.
        levels: One :class:`WarmStartLevel` per level solved.
        reached_finest: ``False`` if ``stop_on_failure`` stopped the sweep.
        converged: The last solved level converged (a solver stop reason in
            ``gradient``, ``step``, ``discrepancy``).
        stop_reason: Stop reason of the last solved level, or
            ``"level_<l>_failed:<reason>"`` if the sweep was stopped.
        warm_iterations: Gauss-Newton steps summed over the levels.
        warm_cost: Fine-equivalent work summed over the levels.
        cold: Cold solve of the finest level from the prolongated user start
            (``None`` if not requested).
        cold_iterations: Steps of the cold solve (``-1`` if not run).
        cold_cost: Fine-equivalent work of the cold solve (``nan`` if not run).
        cost_ratio: ``warm_cost / cold_cost`` (``nan`` if no cold run); a
            value above 1 means the warm start cost more.
        objective_gap: ``Phi_warm - Phi_cold`` at the finest level (``nan``
            if no cold run, or the finest level was not reached). A positive
            value means the warm start ended at a worse objective; compare
            costs only when the gap is negligible.
        estimate_distance: ``||x_warm - x_cold||_2`` (``nan`` as above).
        cost_source: How the level costs were obtained.

    References:
        [EQ-INV-MFX-011].

    """

    estimate: Vector
    levels: tuple[WarmStartLevel, ...]
    reached_finest: bool
    converged: bool
    stop_reason: str
    warm_iterations: int
    warm_cost: float
    cold: GaussNewtonResult | None
    cold_iterations: int
    cold_cost: float
    cost_ratio: float
    objective_gap: float
    estimate_distance: float
    cost_source: str


def _per_level(value: object, count: int, name: str) -> list[object]:
    """Broadcast a scalar or validate a per-level list.

    References:
        [EQ-INV-MFX-011].

    """
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"{name} must have one entry per level ({count})")
        return list(value)
    return [value] * count


def _warm_evaluations(result: GaussNewtonResult, derivative_cost: float) -> float:
    """Return value-equivalent model calls of a solve.

    References:
        [EQ-INV-MFX-011].

    """
    counts = result.counts
    return (
        counts.value
        + counts.derivative_value
        + derivative_cost * (counts.jvp + counts.vjp)
    )


def coarse_to_fine_warm_start(
    models: Sequence[DifferentiableForward],
    prolongations: Sequence[Prolongation],
    data: Sequence[float],
    *,
    level_costs: Sequence[float] | None = None,
    derivative_cost: float = 1.0,
    noise: object = None,
    regularization: float | Sequence[float] = 0.0,
    penalties: Sequence[LinearOperator | None] | None = None,
    references: Sequence[Sequence[float] | None] | None = None,
    initial: Sequence[float] | None = None,
    compare_cold_start: bool = True,
    stop_on_failure: bool = False,
    max_iterations: int = 50,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    inner_solver: Literal["lsqr", "cgls"] = "lsqr",
    inner_tolerance: float = 1e-8,
) -> WarmStartResult:
    """Solve a regularised least-squares problem coarse to fine.

    Level ``l`` minimises
    ``Phi_l(x) = 1/2 ||W (G_l(x) - y)||^2 + lambda_l^2/2 ||L_l (x - x_ref,l)||^2``
    with :func:`~probstat.inverse.gauss_newton.gauss_newton_krylov`, starting at
    ``x_l^0 = P_(l-1)(x_(l-1)^*)`` (``x_0^0`` the user start, zeros by
    default) [EQ-INV-MFX-011]. The data ``y`` and noise are shared by all
    levels, so every ``G_l`` must have the same output dimension. The
    prolongation maps are caller supplied (for nested meshes: piecewise
    constant injection or interpolation); they are checked for dimension and
    finiteness but not for consistency with the discretisation. The solver
    options are identical for the warm sweep and the cold comparison.

    Work is ``model_evaluations_l = value + derivative_value +
    derivative_cost (jvp + vjp)`` calls, converted to finest-level units with
    the cost ratio ``c_l / c_L`` through :func:`fine_equivalent_cost`.
    ``level_costs`` are per-call costs declared by the caller; the default is
    the parameter dimension of each level, which is an assumption (linear cost
    in the unknowns) and is recorded in ``cost_source``. Nothing here asserts a
    saving: ``cost_ratio`` and ``objective_gap`` report what was measured, and
    the cold solve may be cheaper. A level that does not converge is reported
    and, unless ``stop_on_failure``, its iterate is still used as the next
    start.

    Args:
        models: Differentiable forward models, coarsest first, at least two.
        prolongations: ``len(models) - 1`` callables; ``prolongations[l]``
            maps a level ``l`` parameter vector to the level ``l + 1`` space.
        data: Observations ``y``.
        level_costs: Positive per-call costs, nondecreasing, one per level.
        derivative_cost: Cost of one ``jvp``/``vjp`` relative to one value
            call (positive; 1 for a PDE solve that reuses a factorisation is
            an assumption, not a measurement).
        noise: Covariance factor, as in ``gauss_newton_krylov``.
        regularization: ``lambda`` (one number or one per level).
        penalties: Optional ``L`` per level (``None`` entries: identity).
        references: Optional ``x_ref`` per level (``None`` entries: zeros).
        initial: Start at the coarsest level.
        compare_cold_start: Also solve the finest level from the prolongated
            start (counted in ``cold_cost``).
        stop_on_failure: Stop at the first level that does not converge.
        max_iterations: Gauss-Newton iteration cap per level.
        gradient_tolerance: Relative gradient tolerance.
        step_tolerance: Relative step tolerance.
        inner_solver: ``"lsqr"`` or ``"cgls"``.
        inner_tolerance: Krylov tolerance.

    Returns:
        A :class:`WarmStartResult`.

    Raises:
        TypeError: For wrongly typed arguments.
        ValueError: If fewer than two levels are given, a list length or a
            dimension is inconsistent, a cost is invalid, or a prolongation
            returns a vector of the wrong length or with non-finite entries.

    Equation:
        [EQ-INV-MFX-011].

    References:
        [EQ-INV-MFX-011], [REF-NOCEDAL-WRIGHT-2006].

    """
    count = len(models)
    if count < 2:
        raise ValueError("at least two levels are required")
    if len(prolongations) != count - 1:
        raise ValueError("need exactly one prolongation per pair of levels")
    for model in models:
        if not isinstance(model, DifferentiableForward):
            raise TypeError("every model must implement DifferentiableForward")
    for prolongation in prolongations:
        if not callable(prolongation):
            raise TypeError("prolongations must be callable")
    if len({model.output_dimension for model in models}) != 1:
        raise ValueError("all levels must have the same output dimension")
    dims = [model.input_dimension for model in models]
    derivative_weight = require_finite_real(derivative_cost, name="derivative_cost")
    if derivative_weight <= 0.0:
        raise ValueError("derivative_cost must be positive")
    if level_costs is None:
        costs = [float(d) for d in dims]
        source = "assumed proportional to the parameter dimension"
    else:
        costs = [require_finite_real(c, name="level_costs") for c in level_costs]
        if len(costs) != count:
            raise ValueError("level_costs must have one entry per level")
        if any(a > b for a, b in zip(costs, costs[1:], strict=False)):
            raise ValueError("level_costs must be nondecreasing")
        source = "declared by the caller"
    lambdas = _per_level(regularization, count, "regularization")
    penalty_list = _per_level(
        None if penalties is None else list(penalties), count, "penalties"
    )
    reference_list = _per_level(
        None if references is None else list(references), count, "references"
    )
    level_models = [CostModel(c, costs[-1], "declared", source) for c in costs]

    def solve(level: int, start: Vector) -> GaussNewtonResult:
        """Run one level from ``start``.

        References:
            [EQ-INV-MFX-011].

        """
        return gauss_newton_krylov(
            models[level],
            data,
            noise=noise,
            regularization=lambdas[level],  # type: ignore[arg-type]
            penalty=penalty_list[level],  # type: ignore[arg-type]
            reference=reference_list[level],  # type: ignore[arg-type]
            initial=start,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
            step_tolerance=step_tolerance,
            inner_solver=inner_solver,
            inner_tolerance=inner_tolerance,
        )

    def lift(level: int, vector: Vector) -> Vector:
        """Prolongate a level ``level`` vector to level ``level + 1``.

        References:
            [EQ-INV-MFX-011].

        """
        lifted = as_vector(prolongations[level](vector), name="prolongation output")
        if len(lifted) != dims[level + 1]:
            raise ValueError(
                f"prolongation {level} returned length {len(lifted)}, "
                f"expected {dims[level + 1]}"
            )
        if not all(math.isfinite(v) for v in lifted):
            raise ValueError(f"prolongation {level} returned non-finite entries")
        return lifted

    if initial is None:
        start = tuple(0.0 for _ in range(dims[0]))
    else:
        start = as_vector(initial, name="initial")
        if len(start) != dims[0]:
            raise ValueError("initial must have the coarsest parameter dimension")
    cold_start = start
    for level in range(count - 1):
        cold_start = lift(level, cold_start)

    reports: list[WarmStartLevel] = []
    reached = True
    reason = ""
    current = start
    for level in range(count):
        result = solve(level, current)
        evaluations = _warm_evaluations(result, derivative_weight)
        ratio = level_models[level].ratio
        reports.append(
            WarmStartLevel(
                level=level,
                dimension=dims[level],
                initial=current,
                result=result,
                iterations=result.iterations,
                model_evaluations=evaluations,
                cost_ratio=ratio,
                cost=evaluations * ratio,
            )
        )
        if not result.converged and stop_on_failure and level < count - 1:
            reached = False
            reason = f"level_{level}_failed:{result.stop_reason}"
            break
        if level < count - 1:
            current = lift(level, result.estimate)
    last = reports[-1].result
    warm_cost = math.fsum(r.cost for r in reports)
    cold: GaussNewtonResult | None = None
    cold_cost = math.nan
    cold_iterations = -1
    ratio_total = math.nan
    gap = math.nan
    distance = math.nan
    if compare_cold_start:
        cold = solve(count - 1, cold_start)
        cold_cost = _warm_evaluations(cold, derivative_weight)
        cold_iterations = cold.iterations
        if cold_cost > 0.0:
            ratio_total = warm_cost / cold_cost
        if reached:
            gap = last.objective - cold.objective
            distance = _distance(last.estimate, cold.estimate)
    return WarmStartResult(
        estimate=last.estimate,
        levels=tuple(reports),
        reached_finest=reached,
        converged=last.converged,
        stop_reason=reason or last.stop_reason,
        warm_iterations=sum(r.iterations for r in reports),
        warm_cost=warm_cost,
        cold=cold,
        cold_iterations=cold_iterations,
        cold_cost=cold_cost,
        cost_ratio=ratio_total,
        objective_gap=gap,
        estimate_distance=distance,
        cost_source=source,
    )


# ---------------------------------------------------------------------------
# Sensitivity-error validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensitivityErrorReport:
    """Derivative error of an approximate model against a fine model.

    Relative errors are ``||J_a v - J_f v|| / ||J_f v||`` for Jacobian-vector
    products and ``||g_a - g_f|| / ||g_f||`` for gradients ``g = J' r``; the
    angle is ``angle(g_a, g_f)`` in radians. ``inf`` marks a zero fine
    sensitivity with a nonzero approximate one or a non-finite approximate
    output; an angle with a zero vector is undefined and excluded
    (``undefined_angles``).

    Attributes:
        points: Number of held-out points.
        jvp_relative_errors: One per (point, direction).
        gradient_relative_errors: One per (point, probe).
        gradient_angles: One per (point, probe) with both gradients nonzero.
        jvp_max: Largest JVP relative error.
        jvp_mean: Mean JVP relative error.
        jvp_median: Median JVP relative error.
        gradient_max: Largest gradient relative error.
        gradient_mean: Mean gradient relative error.
        gradient_median: Median gradient relative error.
        angle_max: Largest gradient angle (radians; ``nan`` if none defined).
        angle_mean: Mean gradient angle.
        angle_median: Median gradient angle.
        undefined_angles: Probes with a zero gradient (angle undefined).
        gradient_kind: ``"misfit"`` (``J'(G - y)`` with each model's own
            residual) or ``"adjoint_probe"`` (``J' w`` for shared weights).
        tolerance: Declared tolerance on relative errors.
        angle_tolerance: Declared tolerance on the angle (radians).
        fit_for_gradient_use: All checks passed on this sample.
        failures: Human-readable reasons for ``False``.

    References:
        [EQ-INV-MFX-012], [EQ-INV-MFX-013].

    """

    points: int
    jvp_relative_errors: tuple[float, ...]
    gradient_relative_errors: tuple[float, ...]
    gradient_angles: tuple[float, ...]
    jvp_max: float
    jvp_mean: float
    jvp_median: float
    gradient_max: float
    gradient_mean: float
    gradient_median: float
    angle_max: float
    angle_mean: float
    angle_median: float
    undefined_angles: int
    gradient_kind: str
    tolerance: float
    angle_tolerance: float
    fit_for_gradient_use: bool
    failures: tuple[str, ...]


def _norm2(vector: Sequence[float]) -> float:
    """Return the Euclidean norm with ``hypot`` scaling.

    References:
        [EQ-INV-MFX-012].

    """
    return math.hypot(*vector) if vector else 0.0


def _relative_error(approx: Sequence[float], exact: Sequence[float]) -> float:
    """Return ``||approx - exact|| / ||exact||`` (``inf`` for a zero exact).

    References:
        [EQ-INV-MFX-012].

    """
    if not all(math.isfinite(a) for a in approx):
        return math.inf
    numerator = _norm2([a - e for a, e in zip(approx, exact, strict=True)])
    denominator = _norm2(exact)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _angle(approx: Sequence[float], exact: Sequence[float]) -> float | None:
    """Return the angle between two vectors via ``atan2`` (``None`` if undefined).

    References:
        [EQ-INV-MFX-013].

    """
    if not all(math.isfinite(a) for a in approx):
        return math.pi
    n_exact = _norm2(exact)
    n_approx = _norm2(approx)
    if n_exact == 0.0 or n_approx == 0.0:
        return None
    unit = [e / n_exact for e in exact]
    along = math.fsum(a * u for a, u in zip(approx, unit, strict=True))
    across = _norm2([a - along * u for a, u in zip(approx, unit, strict=True)])
    return math.atan2(across, along)


def _summary(values: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(max, mean, median)`` (``nan`` triple for no values).

    References:
        [EQ-INV-MFX-012].

    """
    if not values:
        return (math.nan, math.nan, math.nan)
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    )
    mean = math.inf if math.inf in ordered else math.fsum(ordered) / len(ordered)
    return (ordered[-1], mean, median)


def validate_sensitivities(
    fine: DifferentiableForward,
    approximate: DifferentiableForward,
    points: Sequence[Sequence[float]],
    *,
    tolerance: float,
    angle_tolerance: float | None = None,
    data: Sequence[float] | None = None,
    directions: Sequence[Sequence[float]] | None = None,
    adjoint_weights: Sequence[Sequence[float]] | None = None,
    probes: int = 3,
    training_points: Sequence[Sequence[float]] | None = None,
    seed: int | RandomStream | None = 0,
) -> SensitivityErrorReport:
    """Estimate the derivative error of an approximate model at held-out points.

    For each held-out point ``x`` the report compares, between the approximate
    model ``G_a`` and the fine model ``G_f``,

    * Jacobian-vector products ``J v`` along unit directions ``v``
      (relative error ``||J_a v - J_f v|| / ||J_f v||``),
    * gradients ``g = J' r``: with ``data`` the misfit gradient
      ``J(x)' (G(x) - y)`` of each model *with its own value and Jacobian*
      (this is what an optimiser sees, so a value error also counts); without
      ``data`` the adjoint product ``J' w`` for weights ``w`` shared by both
      models,
    * the angle between the two gradients ([EQ-INV-MFX-012],
      [EQ-INV-MFX-013]).

    Directions and weights are given or drawn from a seeded Gaussian and
    normalised. The approximation is called *fit for gradient use* when,
    over all probes, the largest JVP and gradient relative errors are at most
    ``tolerance`` and the largest angle is at most ``angle_tolerance``
    (default ``asin(min(tolerance, 1))``, the largest angle compatible with a
    relative gradient error ``tolerance``). The decision uses the worst case
    of the sample only: it is evidence on the probed points and directions, not
    a bound over the parameter space, and fewer probes give a weaker claim.
    Held-out points must not be training points; pass ``training_points`` to
    have coincidences rejected. Unit noise is assumed in the misfit gradient.

    Args:
        fine: Reference model (must return finite values and derivatives).
        approximate: Coarse model or surrogate with the same dimensions.
        points: Non-empty held-out parameter vectors.
        tolerance: Declared tolerance on relative errors (positive).
        angle_tolerance: Declared tolerance on the angle in radians, in
            ``(0, pi]``; default as above.
        data: Observations for the misfit gradient (length ``m``).
        directions: Directions for JVP probes, used at every point; default
            ``probes`` random unit directions per point.
        adjoint_weights: Weights for adjoint probes (ignored with ``data``);
            default ``probes`` random unit weights per point.
        probes: Number of random directions/weights per point (positive).
        training_points: Points used to build the approximation, to enforce
            disjointness from ``points``.
        seed: Seed for the random probes (``None`` is not reproducible).

    Returns:
        A :class:`SensitivityErrorReport`.

    Raises:
        TypeError: For wrongly typed arguments.
        ValueError: If dimensions differ, ``points`` is empty, a tolerance is
            invalid, a held-out point is a training point, or the fine model
            returns non-finite values.

    Equation:
        [EQ-INV-MFX-012], [EQ-INV-MFX-013].

    References:
        [EQ-INV-MFX-012], [EQ-INV-MFX-013], [REF-NOCEDAL-WRIGHT-2006].

    """
    for label, model in (("fine", fine), ("approximate", approximate)):
        if not isinstance(model, DifferentiableForward):
            raise TypeError(f"{label} must implement DifferentiableForward")
    n, m = fine.input_dimension, fine.output_dimension
    if (approximate.input_dimension, approximate.output_dimension) != (n, m):
        raise ValueError("fine and approximate models must have equal dimensions")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    if angle_tolerance is None:
        angle_tol = math.asin(min(tol, 1.0))
    else:
        angle_tol = require_finite_real(angle_tolerance, name="angle_tolerance")
        if not 0.0 < angle_tol <= math.pi:
            raise ValueError("angle_tolerance must lie in (0, pi]")
    count = _positive_int(probes, "probes", minimum=1)
    if not points:
        raise ValueError("points must be non-empty")
    cloud = [as_vector(p, name="point") for p in points]
    if any(len(p) != n for p in cloud):
        raise ValueError("every point must have the parameter dimension")
    if training_points is not None:
        trained = {as_vector(p, name="training point") for p in training_points}
        clash = [p for p in cloud if p in trained]
        if clash:
            raise ValueError(f"{len(clash)} held-out point(s) are training points")
    y: Vector | None = None
    if data is not None:
        y = as_vector(data, name="data")
        if len(y) != m:
            raise ValueError("data must have the output dimension")

    def fixed(
        rows: Sequence[Sequence[float]] | None, size: int, name: str
    ) -> list[Vector] | None:
        """Validate user-supplied probe vectors.

        References:
            [EQ-INV-MFX-012].

        """
        if rows is None:
            return None
        out = [as_vector(r, name=name) for r in rows]
        if not out or any(len(r) != size for r in out):
            raise ValueError(f"{name} must be non-empty with length {size}")
        if any(_norm2(r) == 0.0 for r in out):
            raise ValueError(f"{name} must be nonzero")
        return out

    given_v = fixed(directions, n, "directions")
    given_w = fixed(adjoint_weights, m, "adjoint_weights")
    rng = resolve_stream(seed).generator()

    def draw(size: int) -> Vector:
        """Return a random unit vector.

        References:
            [EQ-INV-MFX-012].

        """
        while True:
            raw = [rng.gauss(0.0, 1.0) for _ in range(size)]
            length = _norm2(raw)
            if length > 0.0:
                return tuple(r / length for r in raw)

    jvp_errors: list[float] = []
    grad_errors: list[float] = []
    angles: list[float] = []
    undefined = 0
    failures: list[str] = []
    for point in cloud:
        value_f = fine.value(point)
        if not all(math.isfinite(v) for v in value_f):
            raise ValueError("the fine model returned non-finite values")
        value_a = approximate.value(point)
        finite_a = all(math.isfinite(v) for v in value_a)
        if not finite_a:
            failures.append("approximate model returned non-finite values")
        for v in given_v if given_v is not None else [draw(n) for _ in range(count)]:
            jf = fine.jvp(point, v)
            if not all(math.isfinite(t) for t in jf):
                raise ValueError("the fine model returned a non-finite jvp")
            jvp_errors.append(_relative_error(approximate.jvp(point, v), jf))
        pairs: list[tuple[Vector, Vector]] = []
        if y is not None:
            residual_f = [a - b for a, b in zip(value_f, y, strict=True)]
            if finite_a:
                residual_a = [a - b for a, b in zip(value_a, y, strict=True)]
                g_a = approximate.vjp(point, residual_a)
            else:
                g_a = tuple(math.nan for _ in range(n))
            pairs.append((fine.vjp(point, residual_f), g_a))
        else:
            weights = (
                given_w if given_w is not None else [draw(m) for _ in range(count)]
            )
            for w in weights:
                pairs.append((fine.vjp(point, w), approximate.vjp(point, w)))
        for g_f, g_a in pairs:
            if not all(math.isfinite(t) for t in g_f):
                raise ValueError("the fine model returned a non-finite gradient")
            grad_errors.append(_relative_error(g_a, g_f))
            theta = _angle(g_a, g_f)
            if theta is None:
                undefined += 1
            else:
                angles.append(theta)
    jmax, jmean, jmed = _summary(jvp_errors)
    gmax, gmean, gmed = _summary(grad_errors)
    amax, amean, amed = _summary(angles)
    if jmax > tol:
        failures.append(f"jvp relative error {jmax:.6g} exceeds tolerance {tol:.6g}")
    if gmax > tol:
        failures.append(
            f"gradient relative error {gmax:.6g} exceeds tolerance {tol:.6g}"
        )
    if angles and amax > angle_tol:
        failures.append(
            f"gradient angle {amax:.6g} rad exceeds tolerance {angle_tol:.6g} rad"
        )
    return SensitivityErrorReport(
        points=len(cloud),
        jvp_relative_errors=tuple(jvp_errors),
        gradient_relative_errors=tuple(grad_errors),
        gradient_angles=tuple(angles),
        jvp_max=jmax,
        jvp_mean=jmean,
        jvp_median=jmed,
        gradient_max=gmax,
        gradient_mean=gmean,
        gradient_median=gmed,
        angle_max=amax,
        angle_mean=amean,
        angle_median=amed,
        undefined_angles=undefined,
        gradient_kind="misfit" if y is not None else "adjoint_probe",
        tolerance=tol,
        angle_tolerance=angle_tol,
        fit_for_gradient_use=not failures,
        failures=tuple(failures),
    )


def surrogate_mean_forward(
    surrogate: PredictiveSurrogate,
    input_dimension: int,
    *,
    step: float | None = None,
) -> CallbackForward:
    """Wrap the predictive mean of a surrogate as a differentiable model.

    The scalar output is the mean ``mu(x)`` (for example the log-target mean of
    :class:`ActiveSurrogateResult`); ``jvp`` and ``vjp`` are central
    differences of ``mu`` with relative step ``step`` (default
    ``eps^(1/3)``), the dense ``2 n`` evaluation adjoint being enabled. The
    derivative of the surrogate is therefore itself approximate, with an
    error of order ``step^2 |mu'''|`` plus roundoff ``eps |mu| / step``; it is
    far below the surrogate's own model error unless ``mu`` is non-smooth.

    Args:
        surrogate: Object with ``predict_with_variance(x) -> (mean, var)``.
        input_dimension: Parameter dimension ``n``.
        step: Relative difference step, or ``None``.

    Returns:
        A :class:`~probstat.inverse.gauss_newton.CallbackForward` with output
        dimension 1.

    Raises:
        TypeError: If ``surrogate`` has no ``predict_with_variance``.
        ValueError: For an invalid dimension or step.

    Equation:
        [EQ-INV-MFX-012].

    References:
        [EQ-INV-MFX-012], [EQ-INV-GN-002].

    """
    if not callable(getattr(surrogate, "predict_with_variance", None)):
        raise TypeError("surrogate must provide predict_with_variance")

    def mean(x: Vector) -> tuple[float]:
        """Return the surrogate mean at ``x``.

        References:
            [EQ-INV-MFX-012].

        """
        return (float(surrogate.predict_with_variance(x)[0]),)

    return CallbackForward(
        input_dimension,
        1,
        mean,
        derivative="central_difference",
        step=step,
        allow_dense_fd_vjp=True,
    )
