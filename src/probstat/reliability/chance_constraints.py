"""Chance constraints: scenario guarantees, certification, and deterministic forms.

A chance constraint asks that a design ``x`` satisfy an uncertain inequality
``g(x, xi) <= 0`` with probability at least ``1 - epsilon`` over the random
vector ``xi``. This module collects the tools that work without a design point
or a model of the whole input law:

- **Scenario approach** (Calafiore and Campi; Campi and Garatti). For a convex
  program with ``d`` decision variables, the solution built from ``N``
  independent samples of the constraint violates a fresh constraint with
  probability ``V <= epsilon`` except with probability
  ``sum_{i<d} C(N,i) epsilon^i (1-epsilon)^(N-i)`` (the binomial tail,
  :func:`scenario_confidence`). Inverting that tail gives the sample size
  (:func:`scenario_sample_size`, with the explicit sufficient bound
  ``N >= (2/epsilon)(d - 1 + ln(1/beta))``) and the violation bound as a
  function of the sample size (:func:`scenario_violation_bound`). Once the
  solution is known, the number ``k`` of support constraints replaces ``d``
  and the wait-and-judge bound of Campi and Garatti (2018) applies
  (:func:`scenario_posterior_violation_bound`).
- **A-posteriori certification of any design** from an independent
  validation sample (:func:`certify_violation_frequency`,
  :func:`certify_design`): the number of violations ``k`` in ``n`` fresh
  draws is binomial, and the Clopper-Pearson interval gives an exact upper
  confidence bound for the violation probability. The certificate does not
  depend on convexity or on how the design was found.
- **Sample-average approximation** (:func:`solve_chance_constrained`): the
  empirical chance constraint "at most ``floor(gamma N)`` of ``N`` samples
  violate" is the single order-statistic constraint
  ``h_(floor(gamma N)+1)(x) <= 0`` (the ``(floor(gamma N)+1)``-th largest of
  ``h_s(x) = max_i g_i(x, xi_s)``), which is solved with
  :func:`~probstat.reliability.minimize_constrained`. This is exact, needs no
  smoothing, and handles joint constraints through the maximum over the
  rows. It is generally **non-convex** (an order statistic of convex
  functions is not convex) and piecewise smooth, so only a local solution is
  found; the out-of-sample certificate, not the optimiser, supports any claim
  about the violation probability. With ``gamma = 0`` the problem is the
  scenario program and the scenario bounds apply when it is convex.
- **Deterministic equivalents** for the two cases that have a closed form
  (:func:`normal_linear_chance_margin`, a second-order cone constraint, and
  :func:`distribution_free_chance_margin`, the Cantelli and Chebyshev
  versions). They also serve as oracles for the sampling tools.

Convention: ``g(x, xi) <= 0`` is satisfied and ``g > 0`` is a violation (the
optimisation convention). This is the opposite of the limit-state convention
of :class:`~probstat.reliability.ModelLimitState`, where ``g <= 0`` is
failure; use ``-g`` to convert.

Assumptions: independent draws of ``xi``; a scenario guarantee needs a convex
program with a unique solution for every sample (non-degeneracy for the
a-posteriori bound). The Clopper-Pearson bound is conservative because the
binomial law is discrete.

Guide: docs/foundations/reliability-analysis.md

References:
    [REF-CALAFIORE-CAMPI-2006], [REF-CAMPI-GARATTI-2008],
    [REF-CAMPI-GARATTI-2018], [REF-CLOPPER-PEARSON-1934],
    [REF-NEMIROVSKI-SHAPIRO-2006], [REF-LUEDTKE-AHMED-2008].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import clopper_pearson_interval
from probstat.linalg import Matrix, Vector, as_vector, norm, symmetric_eigen
from probstat.modeling import standard_normal_cdf, standard_normal_quantile
from probstat.optimization.bounds import Bounds

from .design_optimization import minimize_constrained

Sampler = Callable[[random.Random], object]
ConstraintFunction = Callable[[Vector, object], float | Sequence[float]]
DISTRIBUTION_FREE_INEQUALITIES = ("cantelli", "chebyshev")


def _unit_open(value: object, name: str) -> float:
    """Return a float strictly inside ``(0, 1)``.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if not 0.0 < checked < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return checked


def _count(value: object, name: str, minimum: int) -> int:
    """Return an integer of at least ``minimum``.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _log_binomial(n: int, k: int) -> float:
    """Return ``ln C(n, k)``.

    References:
        [EQ-REL-SCENARIO-001]. See :func:`scenario_confidence`.

    """
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binomial_lower_tail(n: int, p: float, k: int) -> float:
    """Return ``P(Binomial(n, p) <= k)`` as a log-space sum of ``k + 1`` terms.

    References:
        [EQ-REL-SCENARIO-001]. See :func:`scenario_confidence`.

    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(_log_binomial(n, i) + i * log_p + (n - i) * log_q)
    return min(1.0, total)


def scenario_confidence(
    sample_size: int, decision_dimension: int, violation_level: float
) -> float:
    """Return ``1 - beta`` for a scenario program with ``N`` samples.

    For a convex scenario program in ``d`` variables the solution built from
    ``N`` independent constraint samples satisfies
    ``P{V > epsilon} <= beta`` with
    ``beta = sum_{i=0}^{d-1} C(N,i) epsilon^i (1-epsilon)^(N-i)``, the lower
    tail ``P(Binomial(N, epsilon) <= d - 1)``; the bound is attained by
    fully-supported problems.

    Args:
        sample_size: Number of samples ``N``, at least 1.
        decision_dimension: Number of decision variables ``d``, at least 1.
        violation_level: Violation probability ``epsilon`` in ``(0, 1)``.

    Returns:
        The guaranteed confidence ``1 - beta``.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count or the level is out of range.

    Equation:
        [EQ-REL-SCENARIO-001].

    References:
        [REF-CAMPI-GARATTI-2008, Theorem 2.4].

    """
    n = _count(sample_size, "sample_size", 1)
    d = _count(decision_dimension, "decision_dimension", 1)
    eps = _unit_open(violation_level, "violation_level")
    return 1.0 - _binomial_lower_tail(n, eps, d - 1)


def scenario_violation_bound(
    sample_size: int, decision_dimension: int, confidence: float
) -> float:
    """Return the smallest ``epsilon`` guaranteed with ``confidence`` by ``N`` samples.

    Solves ``sum_{i<d} C(N,i) epsilon^i (1-epsilon)^(N-i) = 1 - confidence``
    for ``epsilon`` by bisection; with ``N <= d - 1`` samples nothing is
    guaranteed and ``1.0`` is returned. The a-priori guarantee is
    ``P{V(x*_N) > epsilon} <= 1 - confidence``.

    Args:
        sample_size: Number of samples ``N``, at least 1.
        decision_dimension: Number of decision variables ``d``, at least 1.
        confidence: Confidence ``1 - beta`` in ``(0, 1)``.

    Returns:
        The violation level ``epsilon`` in ``(0, 1]``.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count or ``confidence`` is out of range.

    Equation:
        [EQ-REL-SCENARIO-001].

    References:
        [REF-CAMPI-GARATTI-2008, Theorem 2.4].

    """
    n = _count(sample_size, "sample_size", 1)
    d = _count(decision_dimension, "decision_dimension", 1)
    beta = 1.0 - _unit_open(confidence, "confidence")
    if n <= d - 1:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _binomial_lower_tail(n, mid, d - 1) > beta:
            low = mid
        else:
            high = mid
        if high - low <= 1e-16:
            break
    return high


@dataclass(frozen=True, slots=True)
class ScenarioSampleSize:
    """Sample sizes that guarantee a scenario violation level.

    Args:
        violation_level: Target ``epsilon``.
        confidence: Target ``1 - beta``.
        decision_dimension: Number of decision variables ``d``.
        size: Smallest ``N`` whose binomial tail is at most ``beta`` (exact).
        explicit_size: ``ceil((2 / epsilon)(d - 1 + ln(1 / beta)))``, a
            sufficient size that is never smaller than ``size``.
        achieved_confidence: ``1 - tail`` at ``size`` (at least ``confidence``).

    References:
        [REF-CAMPI-GARATTI-2008, Theorem 2.4].

    """

    violation_level: float
    confidence: float
    decision_dimension: int
    size: int
    explicit_size: int
    achieved_confidence: float


def scenario_sample_size(
    violation_level: float, confidence: float, decision_dimension: int
) -> ScenarioSampleSize:
    """Find the number of scenarios that guarantees ``V <= epsilon`` with ``1 - beta``.

    The exact size is the smallest ``N`` with
    ``P(Binomial(N, epsilon) <= d - 1) <= beta``. The explicit sufficient
    size follows from the multiplicative Chernoff bound
    ``P(Binomial(N, epsilon) <= a) <= exp(-(N epsilon - a)^2 / (2 N epsilon))``
    for ``a < N epsilon``: with ``a = d - 1`` and ``N epsilon >= 2 (a + L)``,
    ``L = ln(1 / beta)``, the exponent is at least ``L``.

    Args:
        violation_level: Target violation probability ``epsilon`` in ``(0, 1)``.
        confidence: Target confidence ``1 - beta`` in ``(0, 1)``.
        decision_dimension: Number of decision variables ``d``, at least 1.

    Returns:
        Exact and explicit sizes with the achieved confidence.

    Raises:
        TypeError: If ``decision_dimension`` is not an integer.
        ValueError: If a level is outside ``(0, 1)`` or ``d < 1``.

    Equation:
        [EQ-REL-SCENARIO-001], [EQ-REL-SCENARIO-002].

    References:
        [REF-CAMPI-GARATTI-2008, Theorem 2.4].

    """
    eps = _unit_open(violation_level, "violation_level")
    conf = _unit_open(confidence, "confidence")
    d = _count(decision_dimension, "decision_dimension", 1)
    beta = 1.0 - conf
    explicit = math.ceil(2.0 / eps * (d - 1 + math.log(1.0 / beta)))
    low, high = d, max(d, explicit)
    while low < high:
        mid = (low + high) // 2
        if _binomial_lower_tail(mid, eps, d - 1) <= beta:
            high = mid
        else:
            low = mid + 1
    return ScenarioSampleSize(
        violation_level=eps,
        confidence=conf,
        decision_dimension=d,
        size=low,
        explicit_size=max(explicit, low),
        achieved_confidence=1.0 - _binomial_lower_tail(low, eps, d - 1),
    )


def scenario_posterior_violation_bound(
    sample_size: int,
    support_constraints: int,
    decision_dimension: int,
    confidence: float,
) -> float:
    """Return the wait-and-judge violation level ``epsilon(k)`` for ``k`` supports.

    With ``beta = 1 - confidence`` and ``k`` the number of support
    constraints of the computed solution (``k <= d``), Campi and Garatti
    (2018, Theorem 2) show that ``P{V(x*_N) > epsilon(s*_N)} <= beta`` for
    ``epsilon(k) = 1 - t(k)``, where ``t(k)`` is the unique root in ``(0, 1)``
    of

        ``beta/(N+1) sum_{m=k}^{N} C(m,k) t^(m-k) - C(N,k) t^(N-k) = 0``.

    The root is found by bisection on ``ln t`` after dividing by
    ``C(N,k) t^(N-k)`` and summing in log space. ``epsilon(k)`` increases with
    ``k``; a solution with few support constraints earns a much smaller
    bound than the worst case ``k = d``. The result needs a convex program
    with a unique solution and the non-degeneracy assumption of the paper.

    Args:
        sample_size: Number of samples ``N``.
        support_constraints: Observed number ``k`` of support constraints,
            ``0 <= k <= min(d, N - 1)``.
        decision_dimension: Number of decision variables ``d``.
        confidence: Confidence ``1 - beta`` in ``(0, 1)``.

    Returns:
        The violation level ``epsilon(k)`` in ``(0, 1)``.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count or ``confidence`` is out of range.

    Equation:
        [EQ-REL-SCENARIO-003].

    References:
        [REF-CAMPI-GARATTI-2018, Theorem 2].

    """
    n = _count(sample_size, "sample_size", 2)
    k = _count(support_constraints, "support_constraints", 0)
    d = _count(decision_dimension, "decision_dimension", 1)
    beta = 1.0 - _unit_open(confidence, "confidence")
    if k > d:
        raise ValueError("support_constraints cannot exceed decision_dimension")
    if k > n - 1:
        raise ValueError("support_constraints must be below sample_size")
    target = math.log((n + 1) / beta)
    offsets = [_log_binomial(m, k) - _log_binomial(n, k) for m in range(k, n + 1)]

    def log_sum(log_t: float) -> float:
        """Return ``ln sum_m [C(m,k)/C(n,k)] t^(m-n)`` at ``t = exp(log_t)``.

        References:
            [EQ-REL-SCENARIO-003].

        """
        terms = [
            off + (m - n) * log_t
            for off, m in zip(offsets, range(k, n + 1), strict=True)
        ]
        top = max(terms)
        return top + math.log(math.fsum(math.exp(t - top) for t in terms))

    high = -1e-12
    low = -1.0
    while log_sum(low) < target:
        low *= 2.0
        if low < -1e8:
            raise ValueError("the wait-and-judge equation has no root")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if log_sum(mid) > target:
            low = mid
        else:
            high = mid
        if high - low <= 1e-15 * max(1.0, abs(low)):
            break
    return -math.expm1(0.5 * (low + high))


@dataclass(frozen=True, slots=True)
class ViolationCertificate:
    """Exact upper confidence bound for a violation probability.

    Args:
        samples: Number ``n`` of independent validation draws.
        violations: Number ``k`` of draws that violated the constraint.
        frequency: ``k / n``.
        upper_bound: One-sided Clopper-Pearson upper bound ``p_U`` with
            ``P(p <= p_U) >= confidence`` for the true violation probability.
        lower_bound: One-sided lower bound at the same confidence.
        confidence: One-sided confidence level.

    References:
        [REF-CLOPPER-PEARSON-1934].

    """

    samples: int
    violations: int
    frequency: float
    upper_bound: float
    lower_bound: float
    confidence: float


def certify_violation_frequency(
    violations: int, samples: int, *, confidence: float = 0.95
) -> ViolationCertificate:
    """Bound a violation probability from ``k`` violations in ``n`` fresh draws.

    The one-sided limits are the two-sided Clopper-Pearson limits at level
    ``2 confidence - 1``, so each tail carries ``1 - confidence`` exactly.
    The coverage is at least ``confidence`` for every true probability.

    Args:
        violations: Observed violations ``k``, ``0 <= k <= n``.
        samples: Number of independent validation draws ``n``, at least 1.
        confidence: One-sided confidence in ``(0.5, 1)``.

    Returns:
        Frequency with exact one-sided bounds.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count or the confidence is out of range.

    Equation:
        [EQ-INF-CI-CP-001].

    References:
        [REF-CLOPPER-PEARSON-1934].

    """
    n = _count(samples, "samples", 1)
    k = _count(violations, "violations", 0)
    conf = _unit_open(confidence, "confidence")
    if conf <= 0.5:
        raise ValueError("confidence must exceed 0.5 for a one-sided bound")
    if k > n:
        raise ValueError("violations must not exceed samples")
    interval = clopper_pearson_interval(k, n, confidence=2.0 * conf - 1.0)
    return ViolationCertificate(
        samples=n,
        violations=k,
        frequency=k / n,
        upper_bound=interval.upper,
        lower_bound=interval.lower,
        confidence=conf,
    )


def _worst(value: float | Sequence[float]) -> float:
    """Return the largest component of a constraint value (joint constraint).

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if isinstance(value, (int, float)):
        worst = float(value)
    else:
        items = [float(v) for v in value]
        if not items:
            raise ValueError("constraint returned an empty vector")
        worst = max(items)
    if math.isnan(worst):
        raise ValueError("constraint returned NaN")
    return worst


def certify_design(
    design: Sequence[float],
    constraint: ConstraintFunction,
    sampler: Sampler,
    samples: int,
    *,
    confidence: float = 0.95,
    seed: int | RandomStream | None = None,
) -> ViolationCertificate:
    """Certify a design against fresh draws of the uncertain constraint.

    A draw violates when the largest component of ``constraint(design, xi)``
    is strictly positive (a joint chance constraint when it returns a
    vector). The draws are independent of how the design was obtained, so the
    certificate is valid for any design chosen before they were seen.

    Args:
        design: Candidate decision vector.
        constraint: ``g(x, xi)`` returning one value or a vector;
            ``g <= 0`` is satisfied.
        sampler: ``sampler(rng) -> xi`` drawing one sample from a
            :class:`random.Random`.
        samples: Number of validation draws.
        confidence: One-sided confidence in ``(0.5, 1)``.
        seed: Integer seed, stream, or ``None`` for fresh entropy.

    Returns:
        The violation count, frequency, and exact Clopper-Pearson bounds.

    Raises:
        TypeError: If a callable or count has the wrong type.
        ValueError: If a numeric argument is out of range.

    Equation:
        [EQ-INF-CI-CP-001].

    References:
        [REF-CLOPPER-PEARSON-1934].

    """
    if not callable(constraint) or not callable(sampler):
        raise TypeError("constraint and sampler must be callable")
    point = as_vector(design, name="design")
    n = _count(samples, "samples", 1)
    rng = resolve_stream(seed).generator()
    violations = sum(1 for _ in range(n) if _worst(constraint(point, sampler(rng))) > 0)
    return certify_violation_frequency(violations, n, confidence=confidence)


@dataclass(frozen=True, slots=True)
class ChanceConstrainedResult:
    """Solution of a sample-average chance-constrained program.

    Args:
        design: Final decision vector.
        objective: Objective value at ``design``.
        optimisation_samples: Number ``N`` of samples used in the program.
        allowed_violations: ``floor(gamma N)`` violations allowed in them.
        empirical_violation: Fraction of the ``N`` samples violated at
            ``design``.
        certificate: Out-of-sample certificate from a fresh seeded sample.
        converged: Whether the constrained solver converged.
        message: Solver termination reason.
        evaluations: Objective evaluations.
        scenario_bound: A-priori ``epsilon`` guaranteed with ``confidence``
            when no violation is allowed (``None`` otherwise). Valid only when
            the program is convex with a unique solution.
        support_constraints: Number of support constraints found by
            leave-one-out re-solves (``None`` unless requested and applicable).
        posterior_bound: Wait-and-judge ``epsilon(k)`` for that number
            (``None`` when ``support_constraints`` is ``None``).

    References:
        [REF-CAMPI-GARATTI-2008], [REF-CLOPPER-PEARSON-1934].

    """

    design: Vector
    objective: float
    optimisation_samples: int
    allowed_violations: int
    empirical_violation: float
    certificate: ViolationCertificate
    converged: bool
    message: str
    evaluations: int
    scenario_bound: float | None
    support_constraints: int | None
    posterior_bound: float | None


def solve_chance_constrained(
    objective: Callable[[Vector], float],
    constraint: ConstraintFunction,
    sampler: Sampler,
    x0: Sequence[float],
    *,
    violation_level: float = 0.0,
    samples: int = 500,
    validation_samples: int = 10000,
    confidence: float = 0.95,
    seed: int | RandomStream | None = 0,
    bounds: Bounds | None = None,
    tolerance: float = 1e-7,
    count_support_constraints: bool = False,
    support_tolerance: float = 1e-4,
) -> ChanceConstrainedResult:
    """Minimise ``f(x)`` subject to ``P(g(x, xi) <= 0) >= 1 - gamma`` by sampling.

    ``N = samples`` draws are taken from the stream child ``"optimise"`` and
    a separate set of ``validation_samples`` draws from the child
    ``"validate"`` of the same seed, so the certificate never sees the
    optimisation samples. With ``h_s(x) = max_i g_i(x, xi_s)`` the sampled
    constraint "at most ``q = floor(gamma N)`` of the ``N`` values ``h_s``
    are positive" is ``c(x) = h_(q+1)(x) <= 0``, the ``(q + 1)``-th largest
    value; it is solved by :func:`~probstat.reliability.minimize_constrained`
    starting at ``x0``. The program is non-convex in general, so the
    answer is a local solution that depends on ``x0``.

    ``violation_level = 0`` is the scenario program. If it is convex with a
    unique solution, the result reports the a-priori ``epsilon`` of
    :func:`scenario_violation_bound` and, when ``count_support_constraints``
    is set, the number of support constraints (active samples whose removal
    moves the solution by more than ``support_tolerance (1 + ||x||)``, found
    by leave-one-out re-solves) with the a-posteriori
    :func:`scenario_posterior_violation_bound`. Those bounds are not valid
    for a non-convex problem.

    Args:
        objective: Objective ``f(x)``.
        constraint: ``g(x, xi)`` returning a value or a vector; ``g <= 0`` is
            satisfied (a vector gives a joint chance constraint).
        sampler: ``sampler(rng) -> xi`` drawing one sample.
        x0: Starting design.
        violation_level: Allowed violation fraction ``gamma`` in ``[0, 1)``.
        samples: Number ``N`` of optimisation samples.
        validation_samples: Number of fresh samples for the certificate.
        confidence: One-sided confidence of the bounds, in ``(0.5, 1)``.
        seed: Integer seed, stream, or ``None`` for fresh entropy.
        bounds: Optional design bounds.
        tolerance: Feasibility tolerance of the constrained solver, applied to
            the sampled constraint so that the returned design satisfies it.
        count_support_constraints: Count support constraints (scenario case).
        support_tolerance: Relative move that marks a support constraint.

    Returns:
        Design, objective, empirical violation, certificate, and solver
        status.

    Raises:
        TypeError: If a callable or count has the wrong type.
        ValueError: If a numeric argument is out of range or the sampled
            program has fewer samples than ``floor(gamma N) + 1``.

    Equation:
        [EQ-REL-SAA-001], [EQ-REL-SCENARIO-001].

    References:
        [REF-CAMPI-GARATTI-2008], [REF-NEMIROVSKI-SHAPIRO-2006],
        [REF-LUEDTKE-AHMED-2008], [REF-CLOPPER-PEARSON-1934].

    """
    if not callable(objective) or not callable(constraint) or not callable(sampler):
        raise TypeError("objective, constraint, and sampler must be callable")
    start = as_vector(x0, name="x0")
    level = require_finite_real(violation_level, name="violation_level")
    if not 0.0 <= level < 1.0:
        raise ValueError("violation_level must lie in [0, 1)")
    n = _count(samples, "samples", 2)
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    allowed = math.floor(level * n)
    stream = resolve_stream(seed)
    rng = stream.child("optimise").generator()
    draws = [sampler(rng) for _ in range(n)]

    def worst_values(x: Vector, active: Sequence[int]) -> list[float]:
        """Return ``h_s(x)`` for the samples listed in ``active``.

        References:
            [EQ-REL-SAA-001].

        """
        return [_worst(constraint(x, draws[i])) for i in active]

    def quantile_constraint(indices: Sequence[int]) -> Callable[[Vector], Vector]:
        """Return ``x -> (h_(q+1)(x) + tol,)`` over the chosen samples.

        References:
            [EQ-REL-SAA-001].

        """
        rank = min(allowed, len(indices) - 1)

        def evaluate(x: Vector) -> Vector:
            """Return the shifted order-statistic constraint at ``x``.

            References:
                [EQ-REL-SAA-001].

            """
            values = worst_values(x, indices)
            if allowed == 0:
                return tuple(v + tol for v in values)
            return (sorted(values, reverse=True)[rank] + tol,)

        return evaluate

    everyone = list(range(n))
    solved = minimize_constrained(
        objective,
        quantile_constraint(everyone),
        start,
        bounds=bounds,
        tolerance=tol,
    )
    design = solved.x
    final_h = worst_values(design, everyone)
    empirical = sum(1 for h in final_h if h > 0.0) / n
    certificate = certify_design(
        design,
        constraint,
        sampler,
        _count(validation_samples, "validation_samples", 1),
        confidence=confidence,
        seed=stream.child("validate"),
    )
    scenario_bound: float | None = None
    support: int | None = None
    posterior: float | None = None
    evaluations = solved.evaluations
    if allowed == 0:
        scenario_bound = scenario_violation_bound(n, len(start), confidence)
        if count_support_constraints:
            active = [i for i, h in enumerate(final_h) if h >= -1e-5]
            moved = 0
            for index in active:
                rest = [i for i in everyone if i != index]
                resolved = minimize_constrained(
                    objective,
                    quantile_constraint(rest),
                    design,
                    bounds=bounds,
                    tolerance=tol,
                )
                evaluations += resolved.evaluations
                shift = norm(
                    tuple(a - b for a, b in zip(resolved.x, design, strict=True))
                )
                if shift > support_tolerance * (1.0 + norm(design)):
                    moved += 1
            support = moved
            if moved <= len(start):
                posterior = scenario_posterior_violation_bound(
                    n, moved, len(start), confidence
                )
    return ChanceConstrainedResult(
        design=design,
        objective=float(objective(design)),
        optimisation_samples=n,
        allowed_violations=allowed,
        empirical_violation=empirical,
        certificate=certificate,
        converged=solved.converged,
        message=solved.message,
        evaluations=evaluations,
        scenario_bound=scenario_bound,
        support_constraints=support,
        posterior_bound=posterior,
    )


def _quadratic_form(x: Vector, covariance: Matrix) -> float:
    """Return ``sqrt(x' Sigma x)`` after checking symmetry and non-negativity.

    Raises:
        ValueError: If ``covariance`` is not square, symmetric, and finite with
            ``x' Sigma x >= 0``.

    References:
        [EQ-REL-CHANCE-NORMAL-001].

    """
    rows = [as_vector(row, name="covariance row") for row in covariance]
    n = len(x)
    if len(rows) != n or any(len(row) != n for row in rows):
        raise ValueError("covariance must be square with the dimension of x")
    scale = max((abs(v) for row in rows for v in row), default=0.0)
    for i in range(n):
        for j in range(i):
            if abs(rows[i][j] - rows[j][i]) > 1e-9 * max(1.0, scale):
                raise ValueError("covariance must be symmetric")
    eigenvalues, _ = symmetric_eigen(tuple(rows))
    if n and min(eigenvalues) < -1e-9 * max(1.0, scale):
        raise ValueError("covariance must be positive semidefinite")
    q = math.fsum(x[i] * rows[i][j] * x[j] for i in range(n) for j in range(n))
    return math.sqrt(max(q, 0.0))


def normal_linear_chance_margin(
    x: Sequence[float],
    mean: Sequence[float],
    covariance: Matrix,
    bound: float,
    violation_level: float,
) -> float:
    """Return the second-order-cone margin of ``P(a'x <= b) >= 1 - epsilon``.

    For ``a ~ N(mu, Sigma)`` the variable ``a'x`` is normal with mean
    ``mu'x`` and variance ``x' Sigma x``, hence
    ``P(a'x <= b) >= 1 - epsilon`` if and only if
    ``mu'x + Phi^-1(1 - epsilon) sqrt(x' Sigma x) <= b``. The function returns
    ``b - mu'x - Phi^-1(1 - epsilon) sqrt(x' Sigma x)``, which is non-negative
    exactly when the chance constraint holds. For ``epsilon <= 1/2`` the
    set of such ``x`` is convex (a second-order cone constraint).

    Args:
        x: Decision vector.
        mean: Mean ``mu`` of the random coefficient vector.
        covariance: Symmetric positive-semidefinite covariance ``Sigma``.
        bound: Right-hand side ``b``.
        violation_level: ``epsilon`` in ``(0, 1)``.

    Returns:
        The margin (non-negative means the constraint holds).

    Raises:
        ValueError: If shapes disagree, ``Sigma`` is not symmetric
            positive-semidefinite, or ``epsilon`` is outside ``(0, 1)``.

    Equation:
        [EQ-REL-CHANCE-NORMAL-001].

    References:
        [REF-NEMIROVSKI-SHAPIRO-2006].

    """
    point = as_vector(x, name="x")
    mu = as_vector(mean, name="mean")
    if len(mu) != len(point):
        raise ValueError("mean and x must have the same length")
    eps = _unit_open(violation_level, "violation_level")
    spread = _quadratic_form(point, covariance)
    limit = require_finite_real(bound, name="bound")
    return (
        limit
        - math.fsum(m * v for m, v in zip(mu, point, strict=True))
        - standard_normal_quantile(1.0 - eps) * spread
    )


def normal_linear_violation_probability(
    x: Sequence[float], mean: Sequence[float], covariance: Matrix, bound: float
) -> float:
    """Return the exact ``P(a'x > b)`` for ``a ~ N(mu, Sigma)``.

    Equal to ``Phi(-(b - mu'x) / sqrt(x' Sigma x))``; when ``x' Sigma x = 0``
    it is ``1`` if ``mu'x > b`` and ``0`` otherwise.

    Args:
        x: Decision vector.
        mean: Mean ``mu``.
        covariance: Symmetric positive-semidefinite covariance.
        bound: Right-hand side ``b``.

    Returns:
        The violation probability.

    Raises:
        ValueError: If shapes disagree or ``Sigma`` is not symmetric
            positive-semidefinite.

    Equation:
        [EQ-REL-CHANCE-NORMAL-001].

    References:
        [REF-NEMIROVSKI-SHAPIRO-2006].

    """
    point = as_vector(x, name="x")
    mu = as_vector(mean, name="mean")
    if len(mu) != len(point):
        raise ValueError("mean and x must have the same length")
    spread = _quadratic_form(point, covariance)
    slack = require_finite_real(bound, name="bound") - math.fsum(
        m * v for m, v in zip(mu, point, strict=True)
    )
    if spread == 0.0:
        return 1.0 if slack < 0.0 else 0.0
    return standard_normal_cdf(-slack / spread)


def distribution_free_chance_margin(
    x: Sequence[float],
    mean: Sequence[float],
    covariance: Matrix,
    bound: float,
    violation_level: float,
    *,
    inequality: str = "cantelli",
) -> float:
    """Return a margin that guarantees ``P(a'x > b) <= epsilon`` for every law.

    Only the mean ``mu`` and covariance ``Sigma`` of ``a`` are assumed. With
    ``m = mu'x`` and ``s = sqrt(x' Sigma x)``:

    - ``"cantelli"`` (one-sided Chebyshev): ``P(a'x > b) <= s^2 / (s^2 + (b-m)^2)``
      for ``b > m``, so ``m + sqrt((1 - epsilon)/epsilon) s <= b`` suffices; it
      is the tightest bound over all laws with these two moments.
    - ``"chebyshev"``: ``P(a'x > b) <= s^2/(b-m)^2``, so
      ``m + s / sqrt(epsilon) <= b`` suffices (more conservative).

    The function returns ``b - m - kappa s`` with the factor ``kappa`` of the
    chosen inequality; a non-negative value certifies the constraint. Compare
    with :func:`normal_linear_chance_margin`, which needs the normal law and
    uses the smaller factor ``Phi^-1(1 - epsilon)``.

    Args:
        x: Decision vector.
        mean: Mean ``mu`` of the random coefficient vector.
        covariance: Symmetric positive-semidefinite covariance.
        bound: Right-hand side ``b``.
        violation_level: ``epsilon`` in ``(0, 1)``.
        inequality: ``"cantelli"`` or ``"chebyshev"``.

    Returns:
        The margin (non-negative certifies the constraint).

    Raises:
        ValueError: If ``inequality`` is unknown, shapes disagree, the
            covariance is invalid, or ``epsilon`` is outside ``(0, 1)``.

    Equation:
        [EQ-REL-CHANCE-FREE-001].

    References:
        [REF-NEMIROVSKI-SHAPIRO-2006].

    """
    if inequality not in DISTRIBUTION_FREE_INEQUALITIES:
        raise ValueError(f"inequality must be one of {DISTRIBUTION_FREE_INEQUALITIES}")
    point = as_vector(x, name="x")
    mu = as_vector(mean, name="mean")
    if len(mu) != len(point):
        raise ValueError("mean and x must have the same length")
    eps = _unit_open(violation_level, "violation_level")
    spread = _quadratic_form(point, covariance)
    kappa = (
        math.sqrt((1.0 - eps) / eps)
        if inequality == "cantelli"
        else 1.0 / math.sqrt(eps)
    )
    return (
        require_finite_real(bound, name="bound")
        - math.fsum(m * v for m, v in zip(mu, point, strict=True))
        - kappa * spread
    )
