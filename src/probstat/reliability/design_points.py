"""Multiple design points: bulged searches, multi-point FORM, and mixture IS.

A limit state can have several local design points (locally nearest points
of the failure surface to the standard-normal origin), for example a series
of failure modes merged with ``min`` or a non-convex surface. FORM at the
global design point then misses the probability mass of the other modes.

* :func:`find_design_points` finds them one after another by the bulging
  idea of Der Kiureghian and Dakessian: once a design point is known, a
  smooth bump that is positive inside a ball around it is added to ``G``,
  which lifts the surface out of that neighbourhood, and HL-RF is restarted;
  each point found on the bulged surface is then refined by HL-RF on the
  original ``G`` and accepted only if it is new.
* :func:`multi_point_form` treats the linearized modes as a series system:
  ``P_f ~= 1 - Phi_m(beta; R)`` with ``R_ij = alpha_i . alpha_j``, closed
  form for one or two points (bivariate normal) and Genz's
  separation-of-variables Monte Carlo for three or more.
* :func:`mixture_importance_sampling` samples a Gaussian mixture centred on
  all design points, which keeps the estimator efficient when several modes
  contribute.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_vector, dot, norm
from probstat.modeling import standard_normal_cdf, standard_normal_quantile
from probstat.optimization import central_gradient

from .model_based import (
    FORMResult,
    ModelLimitState,
    _form_result,
    _hlrf_iterate,
    form_hlrf,
    generalized_reliability_index,
)
from .systems import SeriesSystemBounds, bivariate_normal_cdf, series_system_bounds

MixtureWeighting = Literal["equal", "form"]
StartRule = Literal["axes", "origin"]
_PIVOT_FLOOR = 1e-10
_SUPPORT_FRACTION = 1e-6


def _require_limit_state(limit_state: object) -> ModelLimitState:
    """Return the argument when it is a :class:`ModelLimitState`.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    return limit_state


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer option with a lower limit.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _require_positive(value: object, name: str) -> float:
    """Validate a strictly positive finite real option.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


@dataclass(frozen=True, slots=True)
class MultipleDesignPointsResult:
    """Design points found by successive bulged HL-RF searches.

    Args:
        limit_state: Label of the analyzed limit state.
        forms: FORM results on the original limit state, in discovery order;
            ``forms[0]`` is the ordinary (global-search) FORM result.
        betas: Reliability index of each design point.
        origin_value: ``G(0)``, which scales the bulge height.
        radius_factor: Bulge radius as a multiple of each point's ``beta``.
        height_factor: Bulge height at its centre as a fraction of ``G(0)``.
        searches: Bulged HL-RF searches attempted after the first FORM.
        evaluations: Model evaluations of this call (including the first
            FORM run unless ``first`` was supplied).
        message: Why the search stopped.

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-DERKIUREGHIAN-2005].

    """

    limit_state: str
    forms: tuple[FORMResult, ...]
    betas: Vector
    origin_value: float
    radius_factor: float
    height_factor: float
    searches: int
    evaluations: int
    message: str


def bulge_value(
    u: Sequence[float],
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    heights: Sequence[float],
) -> float:
    """Return the sum of quartic bulges ``B(u)`` around known design points.

    ``B_j(u) = h_j (1 - ||u - u_j||^2 / r_j^2)^2`` inside the ball
    ``||u - u_j|| < r_j`` and zero outside, so ``B_j`` is continuously
    differentiable, equals ``h_j`` at the centre, and vanishes with its
    gradient on the sphere ``||u - u_j|| = r_j``.

    Args:
        u: Standard-normal point.
        centers: Design points ``u_j``.
        radii: Positive radii ``r_j``.
        heights: Non-negative heights ``h_j``.

    Returns:
        ``sum_j B_j(u)``.

    Equation:
        [EQ-REL-MDP-001].

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998].

    """
    total = 0.0
    for center, radius, height in zip(centers, radii, heights, strict=True):
        distance = math.fsum((a - b) ** 2 for a, b in zip(u, center, strict=True))
        ratio = distance / (radius * radius)
        if ratio < 1.0:
            total += height * (1.0 - ratio) ** 2
    return total


def find_design_points(
    limit_state: ModelLimitState,
    max_points: int = 3,
    *,
    first: FORMResult | None = None,
    starts: StartRule | Sequence[Sequence[float]] = "axes",
    radius_factor: float = 1.1,
    height_factor: float = 0.75,
    beta_ratio: float | None = None,
    distinct_tolerance: float = 1e-2,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    relative_step: float = 1e-6,
) -> MultipleDesignPointsResult:
    """Find several design points by successive bulging of the limit state.

    After the ordinary FORM point ``u_1`` (``first`` or a fresh
    :func:`form_hlrf` run), each round runs HL-RF on the bulged function
    ``G_b(u) = G(u) + sum_j B_j(u)`` (see :func:`bulge_value`) with radii
    ``r_j = radius_factor * beta_j`` and heights ``h_j = height_factor *
    G(0)``. At each known point ``G_b(u_j) = h_j > 0``, so the bulged surface
    leaves the ball around it and the searches are deflected from known
    points. Every point found on the bulged surface is refined by HL-RF on
    the original ``G`` (feasibility judged relative to ``|G(0)|``); a refined
    point is a candidate when it converges, has ``beta > 0``, and lies at
    least ``distinct_tolerance * max(1, beta_j)`` from every known point. The
    candidate with the smallest ``beta`` is accepted. The search stops at
    ``max_points`` points, when a round yields no candidate, or when the best
    candidate has ``beta > beta_ratio * beta_1``.

    Starts: HL-RF is local, and the bulge gradient at the origin is parallel
    to ``u_1``, so a single restart from the origin is drawn straight back
    towards ``u_1`` whenever ``grad G(0)`` points there (for example for
    ``G = min(g_1, g_2)``; the search then stalls on the bulge axis or rim and
    refines back to ``u_1``). With ``starts="axes"`` each round therefore
    runs from the origin and from the ``2d`` points ``+-beta_1 e_i``;
    ``starts="origin"`` uses the origin only, and an explicit sequence of
    standard-normal points may be given. The cost per round is about
    ``(2d + 1)`` HL-RF searches for ``"axes"``.

    The bulging idea is from [REF-DERKIUREGHIAN-DAKESSIAN-1998]. The quartic
    profile, the height scaled by ``G(0)``, the multi-start rule, and the
    defaults ``radius_factor = 1.1`` and ``height_factor = 0.75`` are this
    library's choices; they are not taken from the paper. Secondary design
    points inside a ball of radius ``r_j`` are hidden by its bulge.

    Args:
        limit_state: Model limit state with a safe origin (``G(0) > 0``).
        max_points: Maximum number of design points to return (at least 1).
        first: Optional converged FORM result for the first point.
        starts: ``"axes"``, ``"origin"``, or explicit starting points.
        radius_factor: Bulge radius relative to ``beta_j`` (positive).
        height_factor: Bulge height relative to ``G(0)`` (positive).
        beta_ratio: Optional cut-off; a best candidate with
            ``beta > beta_ratio * beta_1`` ends the search.
        distinct_tolerance: Relative distance below which a refined point is
            treated as a known point.
        max_iterations: HL-RF iteration limit per search.
        tolerance: HL-RF relative tolerance.
        relative_step: Central-difference step for gradients.

    Returns:
        Design points, indices, bulge settings, cost, and stop reason.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If options are invalid, ``G(0) <= 0``, the first FORM run
            did not converge or has ``beta <= 0``, or a start point has the
            wrong length.

    Equation:
        [EQ-REL-MDP-001], [EQ-REL-HLRF-001], [EQ-REL-HLRF-002].

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-ZHANG-DERKIUREGHIAN-1995],
        [REF-DERKIUREGHIAN-2005].

    """
    state = _require_limit_state(limit_state)
    limit = _require_count(max_points, "max_points", 1)
    radius_scale = _require_positive(radius_factor, "radius_factor")
    height_scale = _require_positive(height_factor, "height_factor")
    separation = _require_positive(distinct_tolerance, "distinct_tolerance")
    iterations_limit = _require_count(max_iterations, "max_iterations", 1)
    tol = _require_positive(tolerance, "tolerance")
    step = _require_positive(relative_step, "relative_step")
    ratio_limit = None
    if beta_ratio is not None:
        ratio_limit = _require_positive(beta_ratio, "beta_ratio")
    dimension = state.dimension
    explicit: tuple[Vector, ...] = ()
    if isinstance(starts, str):
        if starts not in ("axes", "origin"):
            raise ValueError("starts must be 'axes', 'origin', or a point list")
    else:
        explicit = tuple(as_vector(point, name="start") for point in starts)
        if not explicit or any(len(point) != dimension for point in explicit):
            raise ValueError("start points must be non-empty with the input length")
    before = state.model.evaluations
    origin: Vector = tuple(0.0 for _ in range(dimension))
    origin_value = state.standard(origin)
    if not origin_value > 0.0:
        raise ValueError("multiple design points need a safe origin, G(0) > 0")
    if first is None:
        first = form_hlrf(
            state,
            max_iterations=iterations_limit,
            tolerance=tol,
            relative_step=step,
        )
    elif not isinstance(first, FORMResult):
        raise TypeError("first must be a FORMResult")
    if not first.converged or not first.beta > 0.0:
        raise ValueError("the first FORM search must converge with beta > 0")
    if explicit:
        start_points = explicit
    elif starts == "origin":
        start_points = (origin,)
    else:
        axes = [
            tuple(sign * first.beta if k == i else 0.0 for k in range(dimension))
            for i in range(dimension)
            for sign in (1.0, -1.0)
        ]
        start_points = (origin, *axes)
    forms: list[FORMResult] = [first]
    centers: list[Vector] = [first.design_point_u]
    radii: list[float] = [radius_scale * first.beta]
    heights: list[float] = [height_scale * origin_value]
    searches = 0
    message = f"found max_points = {limit} design points"

    def bulged(u: Vector) -> float:
        """Evaluate ``G(u) + B(u)`` (one model call).

        References:
            [EQ-REL-MDP-001], [REF-DERKIUREGHIAN-DAKESSIAN-1998].

        """
        return state.standard(u) + bulge_value(u, centers, radii, heights)

    def gradient_function(
        function: Callable[[Vector], float],
    ) -> Callable[[Vector], Vector]:
        """Return a central-difference gradient callable for ``function``.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """

        def gradient(point: Vector) -> Vector:
            """Return the central-difference gradient at ``point``.

            References:
                [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

            """
            values, _ = central_gradient(function, point, relative_step=step)
            return values

        return gradient

    def is_new(candidate: FORMResult, known: Sequence[FORMResult]) -> bool:
        """Return whether ``candidate`` is away from every known point.

        References:
            [REF-DERKIUREGHIAN-DAKESSIAN-1998]. No equation is evaluated.

        """
        return all(
            norm(
                tuple(
                    a - b
                    for a, b in zip(
                        candidate.design_point_u, item.design_point_u, strict=True
                    )
                )
            )
            > separation * max(1.0, item.beta)
            for item in known
        )

    while len(forms) < limit:
        candidates: list[FORMResult] = []
        for start in start_points:
            searches += 1
            try:
                found = _hlrf_iterate(
                    bulged,
                    start,
                    gradient_function(bulged),
                    True,
                    iterations_limit,
                    tol,
                    scale=abs(origin_value),
                )
                if not found[4]:
                    continue
                refine_before = state.model.evaluations
                refined = _hlrf_iterate(
                    state.standard,
                    found[0],
                    gradient_function(state.standard),
                    True,
                    iterations_limit,
                    tol,
                    scale=abs(origin_value),
                )
            except ValueError:
                continue
            candidate = _form_result(
                state, refined, state.model.evaluations - refine_before, 0
            )
            if (
                candidate.converged
                and candidate.beta > 0.0
                and is_new(candidate, forms)
                and is_new(candidate, candidates)
            ):
                candidates.append(candidate)
        if not candidates:
            message = "no start reached a new design point"
            break
        best = min(candidates, key=lambda item: item.beta)
        if ratio_limit is not None and best.beta > ratio_limit * first.beta:
            message = "next design point exceeds beta_ratio * beta_1"
            break
        forms.append(best)
        centers.append(best.design_point_u)
        radii.append(radius_scale * best.beta)
        heights.append(height_scale * origin_value)
    return MultipleDesignPointsResult(
        limit_state=state.label,
        forms=tuple(forms),
        betas=tuple(form.beta for form in forms),
        origin_value=origin_value,
        radius_factor=radius_scale,
        height_factor=height_scale,
        searches=searches,
        evaluations=state.model.evaluations - before,
        message=message,
    )


@dataclass(frozen=True, slots=True)
class MultiPointFORMResult:
    """Series-system FORM approximation over several design points.

    Args:
        probability: ``1 - Phi_m(beta; R)``.
        standard_error: Monte Carlo standard error (zero for the closed forms
            with one or two points).
        generalized_beta: ``-Phi^-1(probability)``.
        betas: Component indices ``beta_i``.
        correlations: ``R_ij = alpha_i . alpha_j``.
        method: ``"single"``, ``"bivariate"``, or ``"genz-sov"``.
        samples: Separation-of-variables samples (zero for closed forms).
        bounds: Uni-modal and Ditlevsen bounds from the same linearization.

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-GENZ-BRETZ-2009],
        [REF-DITLEVSEN-1979].

    """

    probability: float
    standard_error: float
    generalized_beta: float
    betas: Vector
    correlations: Matrix
    method: str
    samples: int
    bounds: SeriesSystemBounds


def _as_forms(
    design_points: MultipleDesignPointsResult | Sequence[FORMResult],
) -> tuple[FORMResult, ...]:
    """Return FORM results from a search result or a sequence.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(design_points, MultipleDesignPointsResult):
        forms = design_points.forms
    else:
        forms = tuple(design_points)
    if not forms:
        raise ValueError("at least one design point is required")
    for index, item in enumerate(forms):
        if not isinstance(item, FORMResult):
            raise TypeError(f"design point {index} must be a FORMResult")
    dimension = len(forms[0].design_point_u)
    if any(len(item.design_point_u) != dimension for item in forms):
        raise ValueError("design points must share one dimension")
    return forms


def _semidefinite_cholesky(matrix: Matrix) -> Matrix:
    """Return a lower factor ``C`` with ``C C' = R`` for a PSD correlation.

    Pivots below ``1e-10`` are set to zero together with their column, so
    exactly dependent directions (``rho = +-1``, repeated design points) are
    represented as deterministic linear combinations of earlier ones.

    References:
        [EQ-REL-MVN-001], [REF-GOLUB-VANLOAN-2013, Section 4.2],
        [REF-GENZ-BRETZ-2009, Chapter 4].

    """
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for j in range(size):
        pivot = matrix[j][j] - math.fsum(lower[j][k] ** 2 for k in range(j))
        if pivot <= _PIVOT_FLOOR:
            continue
        root = math.sqrt(pivot)
        lower[j][j] = root
        for i in range(j + 1, size):
            lower[i][j] = (
                matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            ) / root
    return tuple(tuple(row) for row in lower)


def _union_by_separation_of_variables(
    betas: Vector,
    correlations: Matrix,
    samples: int,
    generator_random: Callable[[], float],
) -> tuple[float, float]:
    """Estimate ``1 - Phi_m(beta; R)`` by Genz's separation of variables.

    With ``R = C C'`` (lower ``C``) and ``V = C Z``, the safe-set probability
    is ``E[prod_j e_j]``: at step ``j`` the constraints on ``Z_j`` given the
    earlier ``y_k`` form an interval ``(t_lo, t_hi)``,
    ``e_j = Phi(t_hi) - Phi(t_lo)`` and ``y_j = Phi^-1(Phi(t_lo) + w_j e_j)``
    with ``w_j ~ U(0, 1)``. A row ``i`` with a zero pivot (rank deficiency,
    e.g. ``rho = -1``) has ``V_i = sum_{k <= j} c_ik Z_k`` with last non-zero
    column ``j``, so ``V_i <= beta_i`` is folded into the interval of step
    ``j`` as an upper (``c_ij > 0``) or lower (``c_ij < 0``) bound; this keeps
    the estimator smooth instead of sampling an indicator. Entries below
    ``1e-6`` of the row's largest entry are ignored when choosing ``j``, so
    rounding noise in nearly antipodal FORM directions does not attach the
    constraint to a column with a negligible coefficient. The complement
    ``1 - prod_j e_j = -expm1(sum_j log1p(-q_j))`` with
    ``q_j = Phi(-t_hi) + Phi(t_lo)`` is averaged directly, which keeps full
    relative precision for small failure probabilities.

    References:
        [EQ-REL-MVN-001], [REF-GENZ-BRETZ-2009, Chapter 4].

    """
    size = len(betas)
    lower = _semidefinite_cholesky(correlations)
    attached: dict[int, list[int]] = {j: [] for j in range(size)}
    constant_violation = False
    for i in range(size):
        if lower[i][i] > 0.0:
            continue
        largest = max((abs(lower[i][k]) for k in range(i)), default=0.0)
        support = [
            k for k in range(i) if abs(lower[i][k]) > _SUPPORT_FRACTION * largest
        ]
        if support:
            attached[support[-1]].append(i)
        elif betas[i] < 0.0:
            constant_violation = True
    if constant_violation:
        return 1.0, 0.0
    active = [j for j in range(size) if lower[j][j] > 0.0]
    values: list[float] = []
    for _ in range(samples):
        drawn = [0.0] * size
        log_safe = 0.0
        for j in active:
            shift = math.fsum(lower[j][k] * drawn[k] for k in range(j))
            t_hi = (betas[j] - shift) / lower[j][j]
            t_lo = -math.inf
            for i in attached[j]:
                rest = math.fsum(lower[i][k] * drawn[k] for k in range(j))
                bound = (betas[i] - rest) / lower[i][j]
                if lower[i][j] > 0.0:
                    t_hi = min(t_hi, bound)
                else:
                    t_lo = max(t_lo, bound)
            if t_lo >= t_hi:
                log_safe = -math.inf
                break
            low_mass = standard_normal_cdf(t_lo) if t_lo > -math.inf else 0.0
            outside = standard_normal_cdf(-t_hi) + low_mass
            if outside >= 1.0:
                log_safe = -math.inf
                break
            log_safe += math.log1p(-outside)
            level = low_mass + generator_random() * (1.0 - outside)
            level = min(max(level, 1e-300), 1.0 - 1e-16)
            drawn[j] = standard_normal_quantile(level)
        values.append(-math.expm1(log_safe))
    mean = math.fsum(values) / samples
    variance = math.fsum((v - mean) ** 2 for v in values) / (samples - 1)
    return mean, math.sqrt(variance / samples)


def multi_point_form(
    design_points: MultipleDesignPointsResult | Sequence[FORMResult],
    *,
    samples: int = 20_000,
    seed: int | RandomStream | None = None,
) -> MultiPointFORMResult:
    """Combine linearized modes at several design points as a series system.

    Each design point ``u_i*`` with index ``beta_i`` and direction
    ``alpha_i`` defines the half-space ``alpha_i . u >= beta_i``. The union
    has probability ``P_f = 1 - Phi_m(beta; R)`` with
    ``R_ij = alpha_i . alpha_j``: ``Phi(-beta_1)`` for one point,
    ``P_1 + P_2 - Phi_2(-beta_1, -beta_2; rho)`` for two (exact to about
    ``1e-15``), and a separation-of-variables Monte Carlo estimate with a
    standard error for three or more.

    Args:
        design_points: Search result or converged FORM results that share one
            standard-normal space (the modes of one limit state, or the
            components of a series system).
        samples: Separation-of-variables samples for three or more points.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Probability, standard error, generalized index, correlations, method,
        and Ditlevsen bounds.

    Raises:
        TypeError: If an item is not a :class:`FORMResult`.
        ValueError: If no points are given, dimensions differ, a FORM result
            did not converge, or ``samples < 2``.

    Equation:
        [EQ-REL-MDP-002], [EQ-REL-BVN-001], [EQ-REL-MVN-001].

    References:
        [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-GENZ-BRETZ-2009],
        [REF-DITLEVSEN-1979].

    """
    forms = _as_forms(design_points)
    count = _require_count(samples, "samples", 2)
    bounds = series_system_bounds(forms)
    betas = tuple(form.beta for form in forms)
    correlations = bounds.correlations
    if len(forms) == 1:
        probability, error, method, used = forms[0].probability, 0.0, "single", 0
    elif len(forms) == 2:
        joint = bivariate_normal_cdf(-betas[0], -betas[1], correlations[0][1])
        probability = forms[0].probability + forms[1].probability - joint
        error, method, used = 0.0, "bivariate", 0
    else:
        generator = resolve_stream(seed).child("multi-point-form").generator()
        probability, error = _union_by_separation_of_variables(
            betas, correlations, count, generator.random
        )
        method, used = "genz-sov", count
    probability = min(1.0, max(0.0, probability))
    return MultiPointFORMResult(
        probability=probability,
        standard_error=error,
        generalized_beta=generalized_reliability_index(probability),
        betas=betas,
        correlations=correlations,
        method=method,
        samples=used,
        bounds=bounds,
    )


@dataclass(frozen=True, slots=True)
class MixtureImportanceResult:
    """Importance-sampling estimate with a mixture of design-point Gaussians.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Weighted failure-indicator mean.
        standard_error: Standard error of the weighted mean.
        coefficient_of_variation: ``standard_error / probability``.
        samples: Number of proposal samples.
        failures: Samples in the failure domain.
        centers: Mixture centres (the design points) in standard space.
        mixture_weights: Component probabilities ``pi_k``.
        failure_effective_sample_size: Kish effective size of the failure
            weights.
        evaluations: Model evaluations for sampling.

    References:
        [REF-OWEN-2013, Chapter 9], [REF-MELCHERS-BECK-2018, Chapter 6].

    """

    limit_state: str
    probability: float
    standard_error: float
    coefficient_of_variation: float
    samples: int
    failures: int
    centers: tuple[Vector, ...]
    mixture_weights: Vector
    failure_effective_sample_size: float
    evaluations: int


def mixture_importance_sampling(
    limit_state: ModelLimitState,
    design_points: MultipleDesignPointsResult | Sequence[FORMResult],
    samples: int = 2000,
    *,
    weighting: MixtureWeighting = "equal",
    seed: int | RandomStream | None = None,
) -> MixtureImportanceResult:
    """Estimate ``P_f`` with a Gaussian mixture centred on all design points.

    The proposal is ``h(u) = sum_k pi_k phi_n(u - u_k*)`` and the weight is
    ``w(u) = phi_n(u) / h(u) = 1 / sum_k pi_k exp(u . u_k* - ||u_k*||^2 / 2)``
    (evaluated by log-sum-exp). Every mixture component has unit covariance,
    so ``h > 0`` everywhere and the estimator
    ``P_f = N^-1 sum_i I(G(u_i) <= 0) w(u_i)`` is unbiased. On the
    half-space ``u . u_k* >= ||u_k*||^2`` beyond each centre the weight is at
    most ``exp(-||u_k*||^2 / 2) / pi_k``, so failure mass near any design
    point is sampled with bounded weights.

    Args:
        limit_state: Model limit state.
        design_points: Search result or FORM results whose design points
            centre the mixture.
        samples: Number of samples (at least two).
        weighting: ``"equal"`` (``pi_k = 1/m``) or ``"form"``
            (``pi_k`` proportional to ``Phi(-beta_k)``).
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, standard error, c.o.v., failure count, mixture, and cost.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If options are invalid or the design points do not match
            the input dimension.

    Equation:
        [EQ-REL-MDP-003], [EQ-REL-IS-001].

    References:
        [REF-OWEN-2013, Chapter 9], [REF-MELCHERS-BECK-2018, Chapter 6],
        [REF-DERKIUREGHIAN-DAKESSIAN-1998].

    """
    state = _require_limit_state(limit_state)
    forms = _as_forms(design_points)
    count = _require_count(samples, "samples", 2)
    if weighting not in ("equal", "form"):
        raise ValueError("weighting must be 'equal' or 'form'")
    centers = tuple(form.design_point_u for form in forms)
    if len(centers[0]) != state.dimension:
        raise ValueError("design points must match the input dimension")
    if weighting == "equal":
        raw = [1.0 for _ in forms]
    else:
        raw = [form.probability for form in forms]
        if not math.fsum(raw) > 0.0:
            raise ValueError("FORM probabilities underflow; use weighting='equal'")
    total = math.fsum(raw)
    mixture = tuple(value / total for value in raw)
    log_mixture = tuple(
        math.log(value) if value > 0.0 else -math.inf for value in mixture
    )
    half_norms = tuple(0.5 * dot(center, center) for center in centers)
    cumulative: list[float] = []
    running = 0.0
    for value in mixture:
        running += value
        cumulative.append(running)
    generator = resolve_stream(seed).child("mixture-is").generator()
    before = state.model.evaluations
    contributions: list[float] = []
    failure_weights: list[float] = []
    for _ in range(count):
        pick = generator.random() * running
        component = next(
            (k for k, edge in enumerate(cumulative) if pick < edge),
            len(cumulative) - 1,
        )
        center = centers[component]
        u = tuple(generator.gauss(0.0, 1.0) + c for c in center)
        exponents = [
            log_pi + dot(u, c) - half
            for log_pi, c, half in zip(log_mixture, centers, half_norms, strict=True)
        ]
        peak = max(exponents)
        log_h_ratio = peak + math.log(math.fsum(math.exp(e - peak) for e in exponents))
        weight = math.exp(-log_h_ratio)
        if state.standard(u) <= 0.0:
            contributions.append(weight)
            failure_weights.append(weight)
        else:
            contributions.append(0.0)
    mean = math.fsum(contributions) / count
    variance = math.fsum((c - mean) ** 2 for c in contributions) / (count - 1)
    error = math.sqrt(variance / count)
    squared = math.fsum(w * w for w in failure_weights)
    return MixtureImportanceResult(
        limit_state=state.label,
        probability=mean,
        standard_error=error,
        coefficient_of_variation=error / mean if mean > 0.0 else math.inf,
        samples=count,
        failures=len(failure_weights),
        centers=centers,
        mixture_weights=mixture,
        failure_effective_sample_size=(
            math.fsum(failure_weights) ** 2 / squared if squared > 0.0 else 0.0
        ),
        evaluations=state.model.evaluations - before,
    )
