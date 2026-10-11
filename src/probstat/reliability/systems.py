"""System reliability of several model limit states sharing one joint input.

A series system fails when any component fails (``min_k g_k <= 0``); a
parallel system fails when all components fail (``max_k g_k <= 0``). This
module estimates both by crude Monte Carlo or by multi-point importance
sampling on the shared joint input, and brackets series systems with
first-order uni-modal and Ditlevsen bi-modal bounds built from component FORM
results and an accurate bivariate normal CDF.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, dot, symmetric_eigen
from probstat.modeling import EvaluationCounter, standard_normal_cdf
from probstat.monte_carlo import estimate_mean, estimate_probability

from .model_based import FORMResult, ModelLimitState, form_hlrf

SystemKind = Literal["series", "parallel"]
SystemMethod = Literal["mc", "is"]

_TWO_PI = 2.0 * math.pi
_LEGENDRE_CACHE: dict[int, tuple[Vector, Vector]] = {}


def gauss_legendre(order: int) -> tuple[Vector, Vector]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]`` by Golub-Welsch.

    The nodes are eigenvalues of the symmetric Jacobi matrix with
    off-diagonal entries ``k / sqrt(4k^2 - 1)``; the weights are twice the
    squared first eigenvector components, so they sum to two.

    Args:
        order: Number of nodes, between 1 and 64.

    Returns:
        Ascending nodes and positive weights.

    Raises:
        ValueError: If ``order`` is outside the supported range.

    Equation:
        [EQ-REL-BVN-002], [EQ-LINALG-005].

    References:
        [REF-GOLUB-WELSCH-1969].

    """
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 64:
        raise ValueError("order must be an integer between 1 and 64")
    cached = _LEGENDRE_CACHE.get(order)
    if cached is not None:
        return cached
    jacobi = tuple(
        tuple(
            max(i, j) / math.sqrt(4.0 * max(i, j) ** 2 - 1.0)
            if abs(i - j) == 1
            else 0.0
            for j in range(order)
        )
        for i in range(order)
    )
    values, vectors = symmetric_eigen(jacobi)
    weights = tuple(2.0 * vectors[0][k] ** 2 for k in range(order))
    total = math.fsum(weights)
    rule = (values, tuple(2.0 * w / total for w in weights))
    _LEGENDRE_CACHE[order] = rule
    return rule


def _bivariate_upper(h: float, k: float, rho: float) -> float:
    """Return ``P(X > h, Y > k)`` for a standard bivariate normal pair.

    This is Genz's double-precision refinement of the Drezner-Wesolowsky
    method: Gauss-Legendre quadrature of Plackett's ``arcsin`` integral for
    ``|rho| < 0.925`` and of an asymptotically expanded integrand otherwise.

    References:
        [EQ-REL-BVN-001], [REF-GENZ-2004], [REF-DREZNER-WESOLOWSKY-1990].

    """
    if rho == 0.0:
        return standard_normal_cdf(-h) * standard_normal_cdf(-k)
    magnitude = abs(rho)
    if magnitude < 0.3:
        nodes, weights = gauss_legendre(6)
    elif magnitude < 0.75:
        nodes, weights = gauss_legendre(12)
    else:
        nodes, weights = gauss_legendre(20)
    hk = h * k
    if magnitude < 0.925:
        hs = 0.5 * (h * h + k * k)
        half_angle = 0.5 * math.asin(rho)
        total = 0.0
        for node, weight in zip(nodes, weights, strict=True):
            sine = math.sin(half_angle * (1.0 + node))
            total += weight * math.exp((sine * hk - hs) / (1.0 - sine * sine))
        value = total * half_angle / _TWO_PI
        value += standard_normal_cdf(-h) * standard_normal_cdf(-k)
        return min(1.0, max(0.0, value))
    if rho < 0.0:
        k = -k
        hk = -hk
    value = 0.0
    if magnitude < 1.0:
        one_minus = (1.0 - magnitude) * (1.0 + magnitude)
        a = math.sqrt(one_minus)
        bs = (h - k) ** 2
        c = (4.0 - hk) / 8.0
        d = (12.0 - hk) / 80.0
        exponent = -0.5 * (bs / one_minus + hk)
        if exponent > -100.0:
            value = (
                a
                * math.exp(exponent)
                * (
                    1.0
                    - c * (bs - one_minus) * (1.0 - d * bs) / 3.0
                    + c * d * one_minus * one_minus
                )
            )
        if hk > -100.0:
            b = math.sqrt(bs)
            tail = math.sqrt(_TWO_PI) * standard_normal_cdf(-b / a)
            value -= (
                math.exp(-0.5 * hk) * tail * b * (1.0 - c * bs * (1.0 - d * bs) / 3.0)
            )
        a *= 0.5
        total = 0.0
        for node, weight in zip(nodes, weights, strict=True):
            xs = (a * (1.0 + node)) ** 2
            local = -0.5 * (bs / xs + hk)
            if local > -100.0:
                rs = math.sqrt(1.0 - xs)
                series = 1.0 + c * xs * (1.0 + 5.0 * d * xs)
                correction = math.exp(-0.5 * hk * xs / (1.0 + rs) ** 2) / rs
                total += weight * math.exp(local) * (series - correction)
        value = (a * total - value) / _TWO_PI
    if rho > 0.0:
        value += standard_normal_cdf(-max(h, k))
    elif h >= k:
        value = -value
    else:
        if h < 0.0:
            band = standard_normal_cdf(k) - standard_normal_cdf(h)
        else:
            band = standard_normal_cdf(-h) - standard_normal_cdf(-k)
        value = band - value
    return min(1.0, max(0.0, value))


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """Return ``Phi_2(h, k; rho) = P(X <= h, Y <= k)`` for standard normals.

    Uses Genz's refinement of the Drezner-Wesolowsky Gauss-Legendre scheme,
    accurate to about ``1e-15`` absolute error, including ``|rho| -> 1``.

    Args:
        h: Upper limit for ``X`` (may be infinite).
        k: Upper limit for ``Y`` (may be infinite).
        rho: Correlation in ``[-1, 1]``.

    Returns:
        Bivariate standard normal CDF value.

    Raises:
        ValueError: If ``rho`` lies outside ``[-1, 1]`` or a limit is NaN.

    Equation:
        [EQ-REL-BVN-001], [EQ-REL-BVN-002].

    References:
        [REF-GENZ-2004], [REF-DREZNER-WESOLOWSKY-1990].

    """
    correlation = require_finite_real(rho, name="rho")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    if isinstance(h, bool) or isinstance(k, bool):
        raise TypeError("limits must be real numbers")
    upper_h, upper_k = float(h), float(k)
    if math.isnan(upper_h) or math.isnan(upper_k):
        raise ValueError("limits must not be NaN")
    if upper_h == -math.inf or upper_k == -math.inf:
        return 0.0
    if upper_h == math.inf:
        return 1.0 if upper_k == math.inf else standard_normal_cdf(upper_k)
    if upper_k == math.inf:
        return standard_normal_cdf(upper_h)
    return _bivariate_upper(-upper_h, -upper_k, correlation)


@dataclass(frozen=True, slots=True)
class SeriesSystemBounds:
    """First-order probability bounds for a series system.

    Args:
        component_probabilities: FORM probabilities ``Phi(-beta_i)`` in the
            given component order.
        correlations: Component correlations ``rho_ij = alpha_i . alpha_j``.
        unimodal_lower: ``max_i P_i``.
        unimodal_upper: ``1 - prod_i (1 - P_i)``; a valid upper bound when
            all component correlations are non-negative.
        ditlevsen_lower: Ditlevsen's bi-modal lower bound.
        ditlevsen_upper: Ditlevsen's bi-modal upper bound.
        order: Component indices sorted by decreasing probability, the order
            used for the Ditlevsen bounds.
        positively_correlated: Whether every ``rho_ij >= 0``.

    References:
        [REF-CORNELL-1967], [REF-DITLEVSEN-1979],
        [REF-MELCHERS-BECK-2018, Chapter 5].

    """

    component_probabilities: Vector
    correlations: Matrix
    unimodal_lower: float
    unimodal_upper: float
    ditlevsen_lower: float
    ditlevsen_upper: float
    order: tuple[int, ...]
    positively_correlated: bool


def series_system_bounds(forms: Sequence[FORMResult]) -> SeriesSystemBounds:
    """Bound a series-system failure probability from component FORM results.

    Component events are linearized at their design points:
    ``P_i = Phi(-beta_i)``, ``rho_ij = alpha_i . alpha_j``, and
    ``P_ij = Phi_2(-beta_i, -beta_j; rho_ij)``. With components ordered by
    decreasing ``P_i``, Ditlevsen's bounds are
    ``P_1 + sum_{i>=2} max(0, P_i - sum_{j<i} P_ij) <= P_f`` and
    ``P_f <= sum_i P_i - sum_{i>=2} max_{j<i} P_ij``.

    Args:
        forms: Converged FORM results of components on one joint input.

    Returns:
        Uni-modal and Ditlevsen bounds with the correlation matrix.

    Raises:
        TypeError: If an item is not a :class:`FORMResult`.
        ValueError: If no components are supplied, dimensions differ, or a
            FORM result did not converge.

    Equation:
        [EQ-REL-SYS-002], [EQ-REL-SYS-003], [EQ-REL-BVN-001].

    References:
        [REF-CORNELL-1967], [REF-DITLEVSEN-1979],
        [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    components = tuple(forms)
    if not components:
        raise ValueError("at least one component FORM result is required")
    for index, item in enumerate(components):
        if not isinstance(item, FORMResult):
            raise TypeError(f"forms[{index}] must be a FORMResult")
        if not item.converged:
            raise ValueError(f"forms[{index}] did not converge")
    dimension = len(components[0].alpha)
    if any(len(item.alpha) != dimension for item in components):
        raise ValueError("component FORM results must share one dimension")
    count = len(components)
    probabilities = tuple(item.probability for item in components)
    correlations = tuple(
        tuple(
            1.0
            if i == j
            else max(-1.0, min(1.0, dot(components[i].alpha, components[j].alpha)))
            for j in range(count)
        )
        for i in range(count)
    )
    order = tuple(sorted(range(count), key=lambda i: -probabilities[i]))
    joint = [
        [
            bivariate_normal_cdf(
                -components[i].beta, -components[j].beta, correlations[i][j]
            )
            if i != j
            else probabilities[i]
            for j in range(count)
        ]
        for i in range(count)
    ]
    first = order[0]
    lower = probabilities[first]
    upper = probabilities[first]
    for position in range(1, count):
        i = order[position]
        earlier = order[:position]
        lower += max(0.0, probabilities[i] - math.fsum(joint[i][j] for j in earlier))
        upper += probabilities[i] - max(joint[i][j] for j in earlier)
    # 1 - prod(1 - P_i) evaluated as -expm1(sum log1p(-P_i)): the naive form
    # cancels to zero for component probabilities below about 1e-16.
    union = (
        1.0
        if any(probability >= 1.0 for probability in probabilities)
        else -math.expm1(
            math.fsum(math.log1p(-probability) for probability in probabilities)
        )
    )
    return SeriesSystemBounds(
        component_probabilities=probabilities,
        correlations=correlations,
        unimodal_lower=max(probabilities),
        unimodal_upper=union,
        ditlevsen_lower=min(1.0, lower),
        ditlevsen_upper=min(1.0, upper),
        order=order,
        positively_correlated=all(
            correlations[i][j] >= 0.0 for i in range(count) for j in range(count)
        ),
    )


@dataclass(frozen=True, slots=True)
class SystemReliabilityResult:
    """Sampling estimate of a series or parallel system failure probability.

    Args:
        kind: ``"series"`` or ``"parallel"``.
        method: ``"mc"`` (crude Monte Carlo) or ``"is"`` (multi-point
            importance sampling around component design points).
        probability: Estimated system failure probability.
        standard_error: Estimated standard error.
        coefficient_of_variation: ``standard_error / probability`` (``inf``
            when no failure was observed).
        samples: Number of sampled input vectors.
        failures: Number of samples in the system failure domain.
        evaluations: Model evaluations for sampling (each distinct model is
            evaluated once per sample).
        component_forms: FORM results used to center the proposal (empty for
            crude Monte Carlo).
        form_evaluations: Model evaluations spent on component FORM runs.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5 and 6], [REF-OWEN-2013, Chapter 9].

    """

    kind: SystemKind
    method: SystemMethod
    probability: float
    standard_error: float
    coefficient_of_variation: float
    samples: int
    failures: int
    evaluations: int
    component_forms: tuple[FORMResult, ...]
    form_evaluations: int


def _distinct_counters(
    components: Sequence[ModelLimitState],
) -> list[EvaluationCounter]:
    """Return the distinct evaluation counters of the component models.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """
    counters: dict[int, EvaluationCounter] = {}
    for component in components:
        counters.setdefault(id(component.model.counter), component.model.counter)
    return list(counters.values())


def system_failure_probability(
    components: Sequence[ModelLimitState],
    *,
    kind: SystemKind = "series",
    method: SystemMethod = "mc",
    samples: int = 10_000,
    seed: int | RandomStream | None = None,
    forms: Sequence[FORMResult] | None = None,
) -> SystemReliabilityResult:
    """Estimate a series or parallel system failure probability by sampling.

    All components must share one joint input law. Every sample is mapped
    once to physical space and each distinct model is evaluated once, so
    components that read different quantities of one model cost one call.

    With ``method="is"`` the proposal is the equal-weight mixture of standard
    normals centered at the component design points ``u_k*``; the weight is
    ``phi(u) / (m^-1 sum_k phi(u - u_k*))``. This suits series systems; for a
    parallel system whose joint design point is far from every component
    design point, prefer crude Monte Carlo or a dedicated proposal.

    Args:
        components: Component model limit states.
        kind: ``"series"`` (any component fails) or ``"parallel"`` (all fail).
        method: ``"mc"`` or ``"is"``.
        samples: Number of samples, at least two.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.
        forms: Optional component FORM results for ``method="is"``; computed
            (and charged) when omitted.

    Returns:
        Probability, standard error, coefficient of variation, and costs.

    Raises:
        TypeError: If a component is not a :class:`ModelLimitState`.
        ValueError: If no components are given, joint inputs differ, the kind
            or method is unknown, or ``samples < 2``.

    Equation:
        [EQ-REL-SYS-001], [EQ-REL-ISDP-001], [EQ-MOD-NATAF-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5 and 6], [REF-OWEN-2013, Chapter 9].

    """
    items = tuple(components)
    if not items:
        raise ValueError("at least one component is required")
    for index, item in enumerate(items):
        if not isinstance(item, ModelLimitState):
            raise TypeError(f"components[{index}] must be a ModelLimitState")
    joint = items[0].model.inputs
    if any(item.model.inputs != joint for item in items[1:]):
        raise ValueError("system components must share one joint input law")
    if kind not in ("series", "parallel"):
        raise ValueError("kind must be 'series' or 'parallel'")
    if method not in ("mc", "is"):
        raise ValueError("method must be 'mc' or 'is'")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    stream = resolve_stream(seed)
    generator = stream.child("system", kind, method).generator()
    dimension = joint.dimension
    counters = _distinct_counters(items)
    centers: tuple[Vector, ...] = ()
    component_forms: tuple[FORMResult, ...] = ()
    form_evaluations = 0
    if method == "is":
        if forms is None:
            before_form = sum(counter.count for counter in counters)
            component_forms = tuple(form_hlrf(item) for item in items)
            form_evaluations = sum(counter.count for counter in counters) - before_form
        else:
            component_forms = tuple(forms)
            if len(component_forms) != len(items):
                raise ValueError("forms must match components one to one")
        centers = tuple(result.design_point_u for result in component_forms)
    before = sum(counter.count for counter in counters)
    failures = 0
    contributions: list[float] = []
    events: list[bool] = []
    for _ in range(samples):
        noise = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
        weight = 1.0
        if method == "is":
            center = centers[generator.randrange(len(centers))]
            u = tuple(z + c for z, c in zip(noise, center, strict=True))
            exponents = [dot(u, c) - 0.5 * dot(c, c) for c in centers]
            peak = max(exponents)
            mixture = math.fsum(math.exp(e - peak) for e in exponents) / len(centers)
            weight = math.exp(-peak) / mixture
        else:
            u = noise
        x = joint.from_standard_normal(u)
        outputs: dict[int, Mapping[str, float]] = {}
        margins: list[float] = []
        for item in items:
            key = id(item.model)
            if key not in outputs:
                outputs[key] = item.model.evaluate(x)
            margins.append(item.margin_from_outputs(outputs[key]))
        governing = min(margins) if kind == "series" else max(margins)
        failed = governing <= 0.0
        failures += int(failed)
        events.append(failed)
        contributions.append(weight if failed else 0.0)
    evaluations = sum(counter.count for counter in counters) - before
    if method == "mc":
        estimate = estimate_probability(events)
        probability, error = estimate.probability, estimate.standard_error
    else:
        weighted = estimate_mean(contributions)
        probability, error = weighted.estimate, weighted.standard_error
    return SystemReliabilityResult(
        kind=kind,
        method=method,
        probability=probability,
        standard_error=error,
        coefficient_of_variation=error / probability if probability > 0 else math.inf,
        samples=samples,
        failures=failures,
        evaluations=evaluations,
        component_forms=component_forms,
        form_evaluations=form_evaluations,
    )
