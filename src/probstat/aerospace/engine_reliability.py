"""Launch-vehicle and rocket-engine reliability for programme reviews.

Four connected analyses, each built on the library's reliability, systems and
risk layers instead of re-implementing them:

1. **Reliability demonstration by hot-fire testing.** The success-run and
   ``r``-failure binomial demonstration plan (:func:`demonstration_plan`, which
   wraps :func:`probstat.reliability.success_run_sample_size`), the exact
   Clopper-Pearson lower bound on reliability
   (:func:`reliability_lower_bound`), the operating-characteristic curve of a
   plan (:func:`operating_characteristic`) with consumer and producer risk,
   a Bayesian demonstration with a Beta prior built from discounted heritage
   data (:func:`heritage_prior`, :func:`bayesian_demonstration_size`), the
   beta-binomial predictive probability of passing a plan
   (:func:`predictive_pass_probability`), and duration-based equivalents for
   exponential and Weibull lives (:func:`exponential_test_time`,
   :func:`weibull_demonstration_units`).
2. **Test-campaign reliability growth.** :func:`assess_test_campaign` wraps
   :func:`probstat.reliability.crow_amsaa_fit` and projects the reliability of
   one mission from the fitted power-law intensity; :class:`BetaMixture`
   updates a *mixture* prior (heritage design versus new design) with
   per-start outcomes in closed form.
3. **Engine-out capability of a clustered stage.**
   :class:`EngineClusterModel` is a ``k``-out-of-``n`` engine cluster with
   beta-factor common-cause failure (:class:`probstat.systems.BetaFactorGroup`)
   and a split of engine failures into benign (engine out, containment holds)
   and catastrophic (vehicle loss) modes. The closed form is cross-checked
   against the exact BDD of :class:`probstat.systems.SystemModel`
   (:meth:`EngineClusterModel.system_model`). Epistemic uncertainty in the
   failure probabilities (Beta or Beta-mixture laws) is propagated by
   :func:`cluster_reliability_distribution` (credible interval and Sobol
   ranking of the uncertain parameters) and, exactly, by
   :func:`cluster_reliability_moments`.
4. **Launch-vehicle event tree.** :func:`launch_event_tree` chains the
   first-stage cluster, stage separation, upper-stage ignition and the
   upper-stage engine in a :class:`probstat.risk.EventTree`;
   :func:`vehicle_reliability_distribution` propagates the epistemic
   uncertainty through the whole tree.

All probabilities are per mission (per start for demand-type events). Every
number in the documentation, the tests and the guide is *illustrative*; no
value is data from a real engine or launch vehicle.

Assumptions and validity.
    Engine outcomes are Bernoulli per mission (the demand model); engines in
    a cluster are exchangeable and fail independently except through one
    beta-factor common-cause event. The common-cause split uses the
    *probability* basis of NUREG/CR-5485 (independent part
    ``(1 - beta) q``, common-cause part ``beta q``), so an engine's marginal
    outage probability is ``q - beta (1 - beta) q^2``, which equals ``q`` to
    first order in ``q``; use it for demand probabilities of the order of
    ``1e-3`` to ``1e-1`` rather than for long missions with large ``q``.
    Because ``q`` is held fixed while ``beta`` moves independent mass into the
    common event, raising ``beta`` lowers the success probability of a
    ``k = n - 1`` cluster only when ``q < 1 / (n (n - 1))`` (leading order);
    above that the independent double-outage term dominates and the sign
    reverses, so compare ``beta`` values at a fixed *independent* probability
    when that is the engineering question. A
    common-cause outage takes out *every* engine, so the mission is lost
    whatever the required number ``k >= 1``. The catastrophic fraction is the
    conditional probability that an outage destroys the vehicle, independent
    per engine. Epistemic parameters are drawn once per sample and shared by
    all engines of a cluster (state-of-knowledge dependence), which keeps the
    uncertainty of the cluster larger than treating engines as independently
    uncertain. Duration-based plans assume a known Weibull shape, no
    unit-to-unit extra variation and failures that are caused by run time
    only (not by the start transient).

Experimental (added in 0.28).

Guide: docs/foundations/aerospace-propulsion-uq.md

References:
    [REF-NASA-PRA-2011], [REF-ATWOOD-NUREG-6823-2003],
    [REF-CLOPPER-PEARSON-1934], [REF-HAMADA-ETAL-2008], [REF-CROW-1974],
    [REF-MIL-HDBK-189C-2011], [REF-MOSLEH-ETAL-1998], [REF-KECECIOGLU-1993].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from probstat.distributions import Beta
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.reliability import (
    CrowAMSAAFit,
    crow_amsaa_fit,
    demonstration_test_time,
    success_run_sample_size,
)
from probstat.risk import Branch, EventTree, Heading, Outcome, binary_heading
from probstat.sensitivity import estimate_sobol_indices
from probstat.systems import (
    BetaFactorGroup,
    DemandComponent,
    SystemModel,
    k_out_of_n,
    parallel,
    series,
)

__all__ = [
    "BetaMixture",
    "ClusterOutcomes",
    "ClusterUncertainty",
    "DemonstrationPlan",
    "EngineClusterModel",
    "GrowthAssessment",
    "LaunchVehicleDesign",
    "MixtureUpdate",
    "ReliabilityDistribution",
    "TimeOnTestPlan",
    "VehicleUncertainty",
    "assess_test_campaign",
    "bayesian_demonstration_confidence",
    "bayesian_demonstration_size",
    "cluster_reliability_distribution",
    "cluster_reliability_moments",
    "demonstration_plan",
    "equivalent_prior_tests",
    "exponential_test_time",
    "heritage_prior",
    "launch_event_tree",
    "nhpp_mission_reliability",
    "operating_characteristic",
    "predictive_pass_probability",
    "propagate_reliability",
    "reliability_lower_bound",
    "smallest_plan_for_producer_risk",
    "vehicle_outcome_probabilities",
    "vehicle_reliability",
    "vehicle_reliability_distribution",
    "weibull_demonstration_units",
]


# ---------------------------------------------------------------- validation
def _open_probability(value: object, *, name: str) -> float:
    """Validate a probability strictly inside ``(0, 1)``.

    Args:
        value: Candidate value.
        name: Field name for error messages.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is 0 or 1.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = require_probability(value, name=name)
    if result in (0.0, 1.0):
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _positive(value: object, *, name: str) -> float:
    """Validate a finite positive real.

    Args:
        value: Candidate value.
        name: Field name for error messages.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _count(value: object, *, name: str, minimum: int = 0) -> int:
    """Validate an integer count of at least ``minimum``.

    Args:
        value: Candidate value.
        name: Field name for error messages.
        minimum: Smallest admissible value.

    Returns:
        The integer.

    Raises:
        ValueError: If the count is below ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = require_non_negative_integer(value, name=name)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _log_beta(a: float, b: float) -> float:
    """Return ``ln B(a, b)``.

    Args:
        a: First shape.
        b: Second shape.

    Returns:
        The log of the beta function.

    References:
        [REF-JKB-1995, Chapter 25].

    """
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _binomial_cdf(x: int, n: int, p: float) -> float:
    """Return ``P(X <= x)`` for ``X ~ Binomial(n, p)`` by an exact log-sum.

    Args:
        x: Count (may be negative or exceed ``n``).
        n: Number of trials.
        p: Success probability in ``[0, 1]``.

    Returns:
        The exact distribution function, summed from the log pmf.

    References:
        [EQ-AERO-REL-001].

    """
    if x < 0:
        return 0.0
    if x >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = (
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        for i in range(x + 1)
    )
    return min(1.0, math.fsum(terms))


# ------------------------------------------------- 1. demonstration by testing
@dataclass(frozen=True, slots=True)
class DemonstrationPlan:
    """A binomial hot-fire demonstration plan and its risks.

    Args:
        tests: Number of tests ``n``.
        allowed_failures: Failures ``r`` tolerated for a pass.
        reliability: Demonstrated reliability ``R`` (per test).
        confidence: Required confidence ``C``.
        achieved_confidence: ``1 - P(X <= r | n, 1 - R)``, at least ``C``.
        continuous_tests: ``ln(1 - C) / ln R`` for ``r = 0`` (the real-valued
            success-run size before rounding up), otherwise ``None``.
        design_reliability: True reliability assumed for the producer risk, or
            ``None``.
        producer_risk: ``P(fail the plan | design reliability)``, or ``None``.

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6], [EQ-AERO-REL-001].

    """

    tests: int
    allowed_failures: int
    reliability: float
    confidence: float
    achieved_confidence: float
    continuous_tests: float | None
    design_reliability: float | None
    producer_risk: float | None


def demonstration_plan(
    reliability: float,
    confidence: float,
    *,
    allowed_failures: int = 0,
    design_reliability: float | None = None,
) -> DemonstrationPlan:
    """Return the smallest binomial plan that demonstrates ``R`` at ``C``.

    The plan is the smallest ``n`` with ``P(Binomial(n, 1 - R) <= r) <= 1 - C``;
    for ``r = 0`` it is ``ceil(ln(1 - C) / ln R)``. The size comes from
    :func:`probstat.reliability.success_run_sample_size`; this function adds
    the achieved confidence and, when a design reliability is given, the
    producer's risk of failing a plan the hardware should pass.

    Args:
        reliability: Reliability ``R`` to demonstrate, in ``(0, 1)``.
        confidence: Confidence ``C``, in ``(0, 1)``.
        allowed_failures: Permitted failures ``r``.
        design_reliability: Optional true reliability for the producer risk.

    Returns:
        The plan.

    Raises:
        ValueError: If a probability is outside ``(0, 1)``.

    Equation:
        [EQ-AERO-REL-001], [EQ-AERO-REL-003].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6], [REF-NASA-PRA-2011].

    """
    target = _open_probability(reliability, name="reliability")
    level = _open_probability(confidence, name="confidence")
    r = _count(allowed_failures, name="allowed_failures")
    n = success_run_sample_size(target, level, allowed_failures=r)
    achieved = 1.0 - _binomial_cdf(r, n, 1.0 - target)
    continuous = math.log1p(-level) / math.log(target) if r == 0 else None
    design: float | None = None
    producer: float | None = None
    if design_reliability is not None:
        design = require_probability(design_reliability, name="design_reliability")
        producer = 1.0 - _binomial_cdf(r, n, 1.0 - design)
    return DemonstrationPlan(
        n, r, target, level, achieved, continuous, design, producer
    )


def smallest_plan_for_producer_risk(
    reliability: float,
    confidence: float,
    design_reliability: float,
    producer_risk: float,
    *,
    max_failures: int = 50,
) -> DemonstrationPlan:
    """Return the smallest plan whose producer risk is at most the target.

    For ``r = 0, 1, 2, ...`` the consumer plan of size ``n_r`` (smallest ``n``
    demonstrating ``R`` at ``C`` with ``r`` allowed failures) is formed and its
    producer risk ``1 - OC(R_design)`` is evaluated. The first ``r`` that
    meets the target is returned (test size increases with ``r``).

    Args:
        reliability: Reliability ``R`` to demonstrate.
        confidence: Consumer confidence ``C``.
        design_reliability: True design reliability, above ``reliability``.
        producer_risk: Largest acceptable probability of failing the plan.
        max_failures: Largest ``r`` searched.

    Returns:
        The plan, with ``design_reliability`` and ``producer_risk`` set.

    Raises:
        ValueError: If the design reliability does not exceed ``R`` or no plan
            with at most ``max_failures`` failures meets the target.

    Equation:
        [EQ-AERO-REL-003].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6], [REF-NASA-PRA-2011].

    """
    target = _open_probability(reliability, name="reliability")
    design = _open_probability(design_reliability, name="design_reliability")
    if design <= target:
        raise ValueError("design_reliability must exceed the demonstrated reliability")
    risk = _open_probability(producer_risk, name="producer_risk")
    limit = _count(max_failures, name="max_failures")
    for r in range(limit + 1):
        plan = demonstration_plan(
            target, confidence, allowed_failures=r, design_reliability=design
        )
        if plan.producer_risk is not None and plan.producer_risk <= risk:
            return plan
    raise ValueError(f"no plan with at most {limit} failures meets the producer risk")


def reliability_lower_bound(tests: int, failures: int, confidence: float) -> float:
    """Return the exact one-sided Clopper-Pearson lower bound on reliability.

    With ``s = n - f`` successes the bound solves
    ``P(Binomial(n, R_L) >= s) = 1 - C`` and equals the ``1 - C`` quantile of
    ``Beta(s, f + 1)``; it is ``0`` when ``s = 0``. For ``f = 0`` it is
    ``(1 - C)^(1/n)``, the success-run bound.

    Args:
        tests: Number of tests ``n >= 1``.
        failures: Observed failures ``0 <= f <= n``.
        confidence: One-sided confidence ``C`` in ``(0, 1)``.

    Returns:
        The lower confidence bound on the per-test reliability.

    Raises:
        ValueError: If the counts are inconsistent.

    Equation:
        [EQ-AERO-REL-002].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """
    n = _count(tests, name="tests", minimum=1)
    f = _count(failures, name="failures")
    if f > n:
        raise ValueError("failures must not exceed tests")
    level = _open_probability(confidence, name="confidence")
    s = n - f
    if s == 0:
        return 0.0
    return float(Beta(float(s), float(f + 1)).quantile(1.0 - level))


def operating_characteristic(
    tests: int, allowed_failures: int, reliabilities: Iterable[float]
) -> tuple[float, ...]:
    """Return the probability of passing the plan at each true reliability.

    ``P_a(R) = P(Binomial(n, 1 - R) <= r)``; the curve is the plan's
    operating characteristic. ``1 - P_a(R)`` below the demonstrated ``R`` is
    the consumer's side, and at the design reliability it is the producer
    risk.

    Args:
        tests: Number of tests ``n >= 1``.
        allowed_failures: Permitted failures ``r``.
        reliabilities: True per-test reliabilities in ``[0, 1]``.

    Returns:
        Acceptance probabilities in the order given.

    Raises:
        ValueError: If a reliability is outside ``[0, 1]``.

    Equation:
        [EQ-AERO-REL-003].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6], [REF-NASA-PRA-2011].

    """
    n = _count(tests, name="tests", minimum=1)
    r = _count(allowed_failures, name="allowed_failures")
    return tuple(
        _binomial_cdf(r, n, 1.0 - require_probability(value, name="reliability"))
        for value in reliabilities
    )


def heritage_prior(
    successes: int,
    failures: int,
    *,
    weight: float = 1.0,
    base: Beta | None = None,
) -> Beta:
    """Return a Beta prior on reliability from discounted heritage tests.

    The power-prior form ``Beta(a_0 + w s, b_0 + w f)`` starts from a base
    prior ``Beta(a_0, b_0)`` (default uniform) and counts each heritage
    success and failure with weight ``w`` in ``[0, 1]``: ``w = 0`` ignores the
    heritage data, ``w = 1`` treats them as exchangeable with the new design.

    Args:
        successes: Heritage successes ``s``.
        failures: Heritage failures ``f``.
        weight: Relevance weight ``w``.
        base: Base prior on the reliability; default ``Beta(1, 1)``.

    Returns:
        The prior on the reliability of the new test series.

    Raises:
        TypeError: If ``base`` is not a Beta law.
        ValueError: If a count or the weight is invalid.

    Equation:
        [EQ-AERO-REL-004].

    References:
        [REF-HAMADA-ETAL-2008], [REF-RAIFFA-SCHLAIFER-1961].

    """
    s = _count(successes, name="successes")
    f = _count(failures, name="failures")
    w = require_probability(weight, name="weight")
    root = Beta(1.0, 1.0) if base is None else base
    if not isinstance(root, Beta):
        raise TypeError("base must be a Beta law")
    return Beta(root.alpha + w * s, root.beta + w * f)


def equivalent_prior_tests(prior: Beta) -> float:
    """Return the equivalent number of tests ``alpha + beta`` of a Beta prior.

    A ``Beta(alpha, beta)`` prior carries the information of ``alpha + beta``
    pseudo-tests (the uniform ``Beta(1, 1)`` counts as two), so this is the
    number of hot-fire tests the prior is "worth".

    Args:
        prior: Beta law on the reliability.

    Returns:
        ``alpha + beta``.

    Raises:
        TypeError: If ``prior`` is not a Beta law.

    Equation:
        [EQ-AERO-REL-004].

    References:
        [REF-HAMADA-ETAL-2008], [REF-GELMAN-ETAL-2013].

    """
    if not isinstance(prior, Beta):
        raise TypeError("prior must be a Beta law")
    return prior.alpha + prior.beta


def bayesian_demonstration_confidence(
    prior: Beta, tests: int, failures: int, required_reliability: float
) -> float:
    """Return ``P(R >= R_req)`` after ``n`` tests with ``f`` failures.

    The posterior is ``Beta(a + n - f, b + f)``; the returned probability is
    its survival function at the required reliability.

    Args:
        prior: Beta prior on the reliability ``R``.
        tests: Tests ``n >= 0``.
        failures: Failures ``0 <= f <= n``.
        required_reliability: Reliability ``R_req`` in ``(0, 1)``.

    Returns:
        The posterior probability that the requirement is met.

    Raises:
        TypeError: If ``prior`` is not a Beta law.
        ValueError: If the counts are inconsistent.

    Equation:
        [EQ-AERO-REL-004].

    References:
        [REF-HAMADA-ETAL-2008], [REF-GELMAN-ETAL-2013].

    """
    if not isinstance(prior, Beta):
        raise TypeError("prior must be a Beta law")
    n = _count(tests, name="tests")
    f = _count(failures, name="failures")
    if f > n:
        raise ValueError("failures must not exceed tests")
    target = _open_probability(required_reliability, name="required_reliability")
    posterior = Beta(prior.alpha + n - f, prior.beta + f)
    return float(posterior.sf(target))


def bayesian_demonstration_size(
    prior: Beta,
    required_reliability: float,
    confidence: float,
    *,
    allowed_failures: int = 0,
    max_tests: int = 10_000_000,
) -> int:
    """Return the smallest test count whose posterior demonstrates ``R_req``.

    The smallest ``n >= r`` with ``P(R >= R_req | n, r) >= C`` for ``r``
    observed failures. The posterior probability increases with ``n`` at fixed
    ``r`` (the posterior is stochastically increasing), so the search is a
    doubling bracket followed by bisection. A prior that already carries the
    requirement gives ``n = r``; informative heritage priors therefore reduce
    the number of tests the programme must fly.

    Args:
        prior: Beta prior on the reliability.
        required_reliability: ``R_req`` in ``(0, 1)``.
        confidence: Posterior probability ``C`` required.
        allowed_failures: Failures ``r`` observed in the plan.
        max_tests: Search limit.

    Returns:
        The smallest sufficient number of tests.

    Raises:
        TypeError: If ``prior`` is not a Beta law.
        ValueError: If more than ``max_tests`` tests would be needed.

    Equation:
        [EQ-AERO-REL-004].

    References:
        [REF-HAMADA-ETAL-2008], [REF-GELMAN-ETAL-2013].

    """
    level = _open_probability(confidence, name="confidence")
    r = _count(allowed_failures, name="allowed_failures")
    limit = _count(max_tests, name="max_tests", minimum=1)

    def enough(n: int) -> bool:
        """Return whether ``n`` tests demonstrate the requirement.

        References:
            [EQ-AERO-REL-004].

        """
        return (
            bayesian_demonstration_confidence(prior, n, r, required_reliability)
            >= level
        )

    if enough(r):
        return r
    low, high = r, max(r + 1, 2 * r + 2)
    while not enough(high):
        low, high = high, 2 * high
        if high > limit:
            raise ValueError("more than max_tests tests are needed")
    while high - low > 1:
        middle = (low + high) // 2
        if enough(middle):
            high = middle
        else:
            low = middle
    return high


def predictive_pass_probability(
    prior: Beta, tests: int, allowed_failures: int
) -> float:
    """Return the prior-predictive probability of passing the plan.

    The number of failures in ``n`` tests is beta-binomial with
    ``P(X = x) = C(n, x) B(x + b, n - x + a) / B(a, b)`` for a
    ``Beta(a, b)`` prior on the reliability (``b`` is the failure shape);
    the plan passes when ``X <= r``. This is the programme's assurance that a
    test series passes before it is flown, averaging the operating
    characteristic over the prior.

    Args:
        prior: Beta prior on the reliability.
        tests: Number of tests ``n >= 1``.
        allowed_failures: Permitted failures ``r``.

    Returns:
        ``P(X <= r)``.

    Raises:
        TypeError: If ``prior`` is not a Beta law.

    Equation:
        [EQ-AERO-REL-005].

    References:
        [REF-GELMAN-ETAL-2013], [REF-HAMADA-ETAL-2008].

    """
    if not isinstance(prior, Beta):
        raise TypeError("prior must be a Beta law")
    n = _count(tests, name="tests", minimum=1)
    r = _count(allowed_failures, name="allowed_failures")
    a, b = prior.alpha, prior.beta
    log_norm = _log_beta(a, b)
    terms = (
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(x + 1)
            - math.lgamma(n - x + 1)
            + _log_beta(x + b, n - x + a)
            - log_norm
        )
        for x in range(min(r, n) + 1)
    )
    return min(1.0, math.fsum(terms))


@dataclass(frozen=True, slots=True)
class TimeOnTestPlan:
    """Total hot-fire time that demonstrates a mission reliability.

    Args:
        total_test_time: Accumulated test time ``T`` (same unit as the mission
            time).
        mission_equivalents: ``T / t_m``, the number of mission durations.
        failure_rate: Demonstrated exponential rate ``lambda_0 = -ln R / t_m``.

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6], [EQ-AERO-REL-006].

    """

    total_test_time: float
    mission_equivalents: float
    failure_rate: float


def exponential_test_time(
    reliability: float,
    confidence: float,
    mission_time: float,
    *,
    allowed_failures: int = 0,
) -> TimeOnTestPlan:
    """Return the total time on test that demonstrates ``R(t_m) >= R``.

    For a constant failure rate, ``R(t_m) = exp(-lambda t_m)`` is demonstrated
    when ``lambda <= lambda_0 = -ln R / t_m`` at confidence ``C``, which needs
    ``T = chi2_C(2r + 2) / (2 lambda_0)``
    (:func:`probstat.reliability.demonstration_test_time`). For ``r = 0`` the
    number of mission equivalents ``T / t_m = ln(1 - C) / ln R`` equals the
    success-run size, so the two views agree.

    Args:
        reliability: Mission reliability ``R`` in ``(0, 1)``.
        confidence: Confidence ``C``.
        mission_time: Mission (burn) duration ``t_m > 0``.
        allowed_failures: Failures ``r`` permitted.

    Returns:
        Total test time and its mission-duration equivalent.

    Raises:
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-REL-006].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6], [REF-KECECIOGLU-1993].

    """
    target = _open_probability(reliability, name="reliability")
    level = _open_probability(confidence, name="confidence")
    t_m = _positive(mission_time, name="mission_time")
    r = _count(allowed_failures, name="allowed_failures")
    rate = -math.log(target) / t_m
    total = demonstration_test_time(rate, level, allowed_failures=r)
    return TimeOnTestPlan(total, total / t_m, rate)


def weibull_demonstration_units(
    reliability: float,
    confidence: float,
    mission_time: float,
    test_time: float,
    shape: float,
    *,
    allowed_failures: int = 0,
) -> int:
    """Return the number of units to demonstrate ``R(t_m)`` with Weibull tests.

    For a Weibull life with known shape ``beta_w``,
    ``R(L t_m) = R(t_m)^(L^beta_w)`` with ``L = t_test / t_m``. Testing each
    unit for ``t_test`` therefore demonstrates ``R`` when the per-unit test
    reliability ``R^(L^beta_w)`` is demonstrated by the binomial plan: the
    smallest ``n`` with ``P(Binomial(n, 1 - R^(L^beta_w)) <= r) <= 1 - C``.
    Tests longer than the mission (``L > 1``) need fewer units, shorter tests
    more; ``beta_w = 1`` is the exponential case.

    Args:
        reliability: Mission reliability ``R`` in ``(0, 1)``.
        confidence: Confidence ``C``.
        mission_time: Mission duration ``t_m > 0``.
        test_time: Duration ``t_test > 0`` each unit is run.
        shape: Known Weibull shape ``beta_w > 0``.
        allowed_failures: Failures ``r`` permitted.

    Returns:
        Number of units.

    Raises:
        ValueError: If an argument is out of range or the test reliability
            rounds to one (the test is too short to be informative).

    Equation:
        [EQ-AERO-REL-006].

    References:
        [REF-KECECIOGLU-1993], [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """
    target = _open_probability(reliability, name="reliability")
    level = _open_probability(confidence, name="confidence")
    t_m = _positive(mission_time, name="mission_time")
    t_test = _positive(test_time, name="test_time")
    beta_w = _positive(shape, name="shape")
    r = _count(allowed_failures, name="allowed_failures")
    unit_reliability = math.exp(math.log(target) * (t_test / t_m) ** beta_w)
    if not 0.0 < unit_reliability < 1.0:
        raise ValueError("the test reliability is not strictly between zero and one")
    return success_run_sample_size(unit_reliability, level, allowed_failures=r)


# ------------------------------------------------------ 2. growth and mixture
@dataclass(frozen=True, slots=True)
class GrowthAssessment:
    """Crow-AMSAA assessment of a test campaign.

    Args:
        fit: The underlying :class:`~probstat.reliability.CrowAMSAAFit`.
        mission_duration: Burn duration ``tau`` to be flown after the campaign.
        current_intensity: Failure intensity ``w(T)`` at the end of testing.
        constant_rate_reliability: ``exp(-w(T) tau)``, the projection that
            freezes the configuration at the end of the campaign.
        nhpp_reliability: ``exp(-lambda ((T + tau)^beta - T^beta))``, the
            probability of no failure in ``(T, T + tau]`` under the fitted
            power-law process.
        growth_detected: Whether the upper confidence limit of the shape is
            below one (a statistically visible growth trend).

    References:
        [REF-CROW-1974], [REF-MIL-HDBK-189C-2011], [EQ-AERO-REL-007].

    """

    fit: CrowAMSAAFit
    mission_duration: float
    current_intensity: float
    constant_rate_reliability: float
    nhpp_reliability: float
    growth_detected: bool


def nhpp_mission_reliability(
    fit: CrowAMSAAFit, mission_duration: float, *, start_time: float | None = None
) -> float:
    """Return the probability of no failure over one mission under the fit.

    For the power-law mean ``Lambda(t) = lambda t^beta`` the probability of no
    failure in ``(s, s + tau]`` is ``exp(-(Lambda(s + tau) - Lambda(s)))``.

    Args:
        fit: A Crow-AMSAA fit.
        mission_duration: Mission duration ``tau > 0`` in test-time units.
        start_time: Accumulated test time ``s`` at the start of the mission;
            default the end of the campaign.

    Returns:
        The mission reliability.

    Raises:
        TypeError: If ``fit`` is not a Crow-AMSAA fit.
        ValueError: If a time is invalid.

    Equation:
        [EQ-AERO-REL-007].

    References:
        [REF-CROW-1974], [REF-MIL-HDBK-189C-2011].

    """
    if not isinstance(fit, CrowAMSAAFit):
        raise TypeError("fit must be a CrowAMSAAFit")
    tau = _positive(mission_duration, name="mission_duration")
    start = (
        fit.total_time
        if start_time is None
        else require_finite_real(start_time, name="start_time")
    )
    if start < 0.0:
        raise ValueError("start_time must be non-negative")
    expected = fit.expected_failures(start + tau) - fit.expected_failures(start)
    return math.exp(-expected)


def assess_test_campaign(
    failure_times: Iterable[float],
    end_time: float | None,
    mission_duration: float,
    *,
    confidence: float = 0.90,
) -> GrowthAssessment:
    """Fit Crow-AMSAA to a campaign's failures and project one mission.

    Failure times are cumulative hot-fire (or test-article) times at which a
    failure occurred, followed by a corrective action; growth shows as a shape
    ``beta < 1``. The fit is
    :func:`probstat.reliability.crow_amsaa_fit` and the mission projection is
    :func:`nhpp_mission_reliability` (and the constant-rate variant).

    Args:
        failure_times: Strictly increasing cumulative failure times.
        end_time: End of the time-terminated campaign, or ``None`` for a
            failure-terminated one.
        mission_duration: Burn duration ``tau`` of the mission to project.
        confidence: Level of the interval on the shape.

    Returns:
        The assessment.

    Raises:
        ValueError: If the data are insufficient or invalid (see
            :func:`probstat.reliability.crow_amsaa_fit`).

    Equation:
        [EQ-AERO-REL-007].

    References:
        [REF-CROW-1974], [REF-MIL-HDBK-189C-2011].

    """
    tau = _positive(mission_duration, name="mission_duration")
    fit = crow_amsaa_fit(failure_times, end_time=end_time, confidence=confidence)
    rate = fit.intensity(fit.total_time)
    return GrowthAssessment(
        fit,
        tau,
        rate,
        math.exp(-rate * tau),
        nhpp_mission_reliability(fit, tau),
        fit.beta_interval[1] < 1.0,
    )


@dataclass(frozen=True, slots=True)
class MixtureUpdate:
    """A mixture posterior and the evidence of the data.

    Args:
        posterior: The updated :class:`BetaMixture`.
        log_evidence: ``ln`` of the marginal likelihood of the observed
            *sequence* of outcomes under the prior mixture.

    References:
        [REF-GELMAN-ETAL-2013], [EQ-AERO-REL-008].

    """

    posterior: BetaMixture
    log_evidence: float


@dataclass(frozen=True, slots=True)
class BetaMixture:
    """A finite mixture of Beta laws on a probability.

    Used for a prior such as "with weight ``w`` the new engine behaves like its
    heritage, otherwise it is an unproven design". The family is conjugate:
    :meth:`update` returns another mixture in closed form.

    Args:
        weights: Positive mixture weights summing to one.
        components: One :class:`~probstat.distributions.Beta` per weight.

    Raises:
        TypeError: If a component is not a Beta law.
        ValueError: If the weights are not a probability vector.

    References:
        [REF-GELMAN-ETAL-2013], [REF-HAMADA-ETAL-2008], [EQ-AERO-REL-008].

    """

    weights: tuple[float, ...]
    components: tuple[Beta, ...]

    def __post_init__(self) -> None:
        """Validate the weights and components.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        weights = tuple(require_finite_real(w, name="weight") for w in self.weights)
        components = tuple(self.components)
        if not weights or len(weights) != len(components):
            raise ValueError("weights and components need equal, positive length")
        if any(w <= 0.0 for w in weights):
            raise ValueError("weights must be positive")
        if abs(math.fsum(weights) - 1.0) > 1e-9:
            raise ValueError("weights must sum to one")
        for component in components:
            if not isinstance(component, Beta):
                raise TypeError("components must be Beta laws")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "components", components)

    def moment(self, order: int) -> float:
        """Return the raw moment ``E[p^m] = sum_j w_j prod_i (a_j+i)/(a_j+b_j+i)``.

        Args:
            order: Non-negative moment order ``m``.

        Returns:
            The raw moment.

        Equation:
            [EQ-AERO-REL-010].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        m = _count(order, name="order")
        total = 0.0
        for weight, law in zip(self.weights, self.components, strict=True):
            value = 1.0
            for i in range(m):
                value *= (law.alpha + i) / (law.alpha + law.beta + i)
            total += weight * value
        return total

    @property
    def mean(self) -> float:
        """Return the mixture mean ``sum_j w_j a_j / (a_j + b_j)``.

        References:
            [EQ-AERO-REL-008].

        """
        return self.moment(1)

    @property
    def variance(self) -> float:
        """Return the mixture variance ``E[p^2] - E[p]^2``.

        References:
            [EQ-AERO-REL-008].

        """
        return self.moment(2) - self.moment(1) ** 2

    def cdf(self, x: float) -> float:
        """Return ``sum_j w_j F_j(x)``.

        Args:
            x: Point in ``[0, 1]``.

        Returns:
            The mixture distribution function.

        References:
            [EQ-AERO-REL-008].

        """
        point = require_probability(x, name="x")
        return math.fsum(
            w * float(c.cdf(point))
            for w, c in zip(self.weights, self.components, strict=True)
        )

    def sf(self, x: float) -> float:
        """Return ``P(p > x)``.

        Args:
            x: Point in ``[0, 1]``.

        Returns:
            The mixture survival function, summed directly from the components.

        References:
            [EQ-AERO-REL-008].

        """
        point = require_probability(x, name="x")
        return math.fsum(
            w * float(c.sf(point))
            for w, c in zip(self.weights, self.components, strict=True)
        )

    def quantile(self, probability: float) -> float:
        """Return the ``probability`` quantile by bisection on :meth:`cdf`.

        Args:
            probability: Level in ``(0, 1)``.

        Returns:
            The quantile, to about ``1e-15``.

        References:
            [EQ-AERO-REL-008].

        """
        level = _open_probability(probability, name="probability")
        low, high = 0.0, 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if self.cdf(middle) < level:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def flipped(self) -> BetaMixture:
        """Return the mixture of ``1 - p`` (reliability to failure probability).

        Returns:
            The mixture with every component's shapes exchanged.

        References:
            [EQ-AERO-REL-008].

        """
        return BetaMixture(
            self.weights, tuple(Beta(c.beta, c.alpha) for c in self.components)
        )

    def update(self, events: int, non_events: int) -> MixtureUpdate:
        """Condition on ``e`` events and ``n`` non-events (a Bernoulli sequence).

        Component ``j`` becomes ``Beta(a_j + e, b_j + n)`` with weight
        proportional to ``w_j B(a_j + e, b_j + n) / B(a_j, b_j)``, the
        component's marginal likelihood of the observed sequence.

        Args:
            events: Count of events (``p`` is the probability of an event).
            non_events: Count of non-events.

        Returns:
            The posterior mixture and ``ln`` marginal likelihood.

        Equation:
            [EQ-AERO-REL-008].

        References:
            [REF-GELMAN-ETAL-2013], [REF-HAMADA-ETAL-2008].

        """
        e = _count(events, name="events")
        n = _count(non_events, name="non_events")
        logs = [
            math.log(w)
            + _log_beta(c.alpha + e, c.beta + n)
            - _log_beta(c.alpha, c.beta)
            for w, c in zip(self.weights, self.components, strict=True)
        ]
        top = max(logs)
        shifted = [math.exp(value - top) for value in logs]
        total = math.fsum(shifted)
        posterior = BetaMixture(
            tuple(value / total for value in shifted),
            tuple(Beta(c.alpha + e, c.beta + n) for c in self.components),
        )
        return MixtureUpdate(posterior, top + math.log(total))

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible observations from the mixture.

        Args:
            size: Number of draws, ``>= 0``.
            seed: Integer seed, stream, or ``None`` for fresh entropy.

        Returns:
            The draws, in a fixed order given the seed.

        References:
            [EQ-AERO-REL-008].

        """
        count = _count(size, name="size")
        stream = resolve_stream(seed)
        generator = stream.child("labels").generator()
        cumulative = tuple(
            math.fsum(self.weights[: j + 1]) for j in range(len(self.weights))
        )
        labels = []
        for _ in range(count):
            u = generator.random()
            labels.append(
                next(
                    (j for j, edge in enumerate(cumulative) if u < edge),
                    len(cumulative) - 1,
                )
            )
        pools = [
            iter(c.sample(labels.count(j), seed=stream.child("component", j)))
            for j, c in enumerate(self.components)
        ]
        return tuple(next(pools[j]) for j in labels)


# ---------------------------------------------------------- epistemic inputs
Law: TypeAlias = Beta | BetaMixture | float


def _check_law(value: object, *, name: str) -> Law:
    """Validate an epistemic input law (Beta, Beta mixture, or a constant).

    Args:
        value: Candidate law.
        name: Field name for error messages.

    Returns:
        The law (a constant becomes a float).

    Raises:
        TypeError: If the value is neither a law nor a real number.
        ValueError: If a constant is outside ``[0, 1]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, Beta | BetaMixture):
        return value
    return require_probability(value, name=name)


def _law_moment(law: Law, order: int) -> float:
    """Return the raw moment ``E[x^m]`` of an input law.

    Args:
        law: Beta, mixture, or constant.
        order: Moment order.

    Returns:
        The moment.

    References:
        [EQ-AERO-REL-010].

    """
    if isinstance(law, BetaMixture):
        return law.moment(order)
    if isinstance(law, Beta):
        value = 1.0
        for i in range(order):
            value *= (law.alpha + i) / (law.alpha + law.beta + i)
        return value
    return law**order


def _draw(law: Law, size: int, stream: RandomStream) -> tuple[float, ...]:
    """Draw ``size`` values from an input law.

    Args:
        law: Beta, mixture, or constant.
        size: Number of draws.
        stream: Random stream for this input.

    Returns:
        The draws.

    References:
        [EQ-AERO-REL-011].

    """
    if isinstance(law, Beta | BetaMixture):
        return law.sample(size, seed=stream)
    return (law,) * size


# ------------------------------------------- 3. engine-out capability cluster
def _cluster_probabilities(
    engines: int, required: int, q: float, beta: float, c: float
) -> tuple[float, float, float]:
    """Return ``(success, benign shortfall, catastrophic)`` of a cluster.

    With ``q_i = (1 - beta) q`` and ``q_c = beta q``,
    ``P(success) = (1 - q_c) sum_{b=0}^{n-k} C(n,b) (q_i (1-c))^b (1-q_i)^(n-b)``
    and ``P(no catastrophe) = (1 - q_c)(1 - q_i c)^n + q_c (1 - c)^n``.

    Args:
        engines: Engines ``n``.
        required: Engines ``k`` that must work.
        q: Engine outage probability per mission.
        beta: Beta factor (zero for a single engine).
        c: Catastrophic fraction.

    Returns:
        Three probabilities summing to one.

    References:
        [EQ-AERO-REL-009].

    """
    q_ind = (1.0 - beta) * q
    q_ccf = beta * q
    benign_each = q_ind * (1.0 - c)
    survive_each = 1.0 - q_ind
    working = math.fsum(
        math.comb(engines, b) * benign_each**b * survive_each ** (engines - b)
        for b in range(engines - required + 1)
    )
    success = (1.0 - q_ccf) * working
    no_catastrophe = (1.0 - q_ccf) * (1.0 - q_ind * c) ** engines + q_ccf * (
        1.0 - c
    ) ** engines
    return success, max(0.0, no_catastrophe - success), max(0.0, 1.0 - no_catastrophe)


@dataclass(frozen=True, slots=True)
class ClusterOutcomes:
    """Mutually exclusive outcomes of an engine cluster for one mission.

    Args:
        success: Enough engines work and none fails catastrophically.
        benign_shortfall: No catastrophic failure, but fewer than ``k`` engines
            work (mission lost, vehicle intact).
        catastrophic: At least one engine failure destroys the vehicle.

    References:
        [EQ-AERO-REL-009].

    """

    success: float
    benign_shortfall: float
    catastrophic: float


@dataclass(frozen=True, slots=True)
class EngineClusterModel:
    """A ``k``-out-of-``n`` engine cluster with common-cause and catastrophic modes.

    Args:
        engines: Number of engines ``n >= 1``.
        required: Engines ``k`` that must work, ``1 <= k <= n``.
        failure_probability: Engine outage probability ``q`` per mission
            (start plus burn), in ``[0, 1]``.
        ccf_beta: Beta factor, the fraction of ``q`` that is a common-cause
            outage of all engines; must be zero for ``n = 1``.
        catastrophic_fraction: Probability ``c`` that an outage is
            catastrophic (vehicle loss) rather than benign (contained
            shutdown), independent per engine.

    Raises:
        ValueError: If the counts or probabilities are invalid.

    Equation:
        [EQ-AERO-REL-009].

    References:
        [REF-MOSLEH-ETAL-1998], [REF-NASA-PRA-2011].

    """

    engines: int
    required: int
    failure_probability: float
    ccf_beta: float = 0.0
    catastrophic_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate the cluster definition.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        n = _count(self.engines, name="engines", minimum=1)
        k = _count(self.required, name="required", minimum=1)
        if k > n:
            raise ValueError("required must not exceed engines")
        beta = require_probability(self.ccf_beta, name="ccf_beta")
        if n == 1 and beta != 0.0:
            raise ValueError("ccf_beta must be zero for a single engine")
        object.__setattr__(self, "engines", n)
        object.__setattr__(self, "required", k)
        object.__setattr__(
            self,
            "failure_probability",
            require_probability(self.failure_probability, name="failure_probability"),
        )
        object.__setattr__(self, "ccf_beta", beta)
        object.__setattr__(
            self,
            "catastrophic_fraction",
            require_probability(
                self.catastrophic_fraction, name="catastrophic_fraction"
            ),
        )

    def outcomes(self) -> ClusterOutcomes:
        """Return the exact three-way outcome probabilities.

        Returns:
            Success, benign shortfall and catastrophic probabilities.

        Equation:
            [EQ-AERO-REL-009].

        References:
            [REF-MOSLEH-ETAL-1998], [REF-NASA-PRA-2011].

        """
        return ClusterOutcomes(
            *_cluster_probabilities(
                self.engines,
                self.required,
                self.failure_probability,
                self.ccf_beta,
                self.catastrophic_fraction,
            )
        )

    def success_probability(self) -> float:
        """Return the mission success probability of the cluster.

        Returns:
            ``P(success)`` from the closed form.

        Equation:
            [EQ-AERO-REL-009].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        return self.outcomes().success

    def system_model(self, *, catastrophe_only: bool = False) -> SystemModel:
        """Return the exact BDD model that the closed form must reproduce.

        Components ``E{i}`` (engine ``i`` does not suffer an outage, failure
        probability ``q``) and ``X{i}`` (an outage is benign; failure
        probability ``c`` is its catastrophic share). Mission success is
        ``k``-out-of-``n`` of the ``E{i}`` in series with
        ``parallel(E{i}, X{i})`` for every engine. A
        :class:`~probstat.systems.BetaFactorGroup` over the ``E{i}`` adds the
        common-cause event, with the probability-basis split.

        Args:
            catastrophe_only: When ``True`` the structure is only "no
                catastrophic failure" (series of the parallel pairs).

        Returns:
            The :class:`~probstat.systems.SystemModel`.

        Equation:
            [EQ-AERO-REL-009].

        References:
            [REF-MOSLEH-ETAL-1998], [REF-RAUSAND-HOYLAND-2004].

        """
        names = [f"E{i + 1}" for i in range(self.engines)]
        pairs = [parallel(e, f"X{i + 1}") for i, e in enumerate(names)]
        parts = (
            pairs if catastrophe_only else [k_out_of_n(self.required, *names), *pairs]
        )
        models = {name: DemandComponent(self.failure_probability) for name in names}
        for i in range(self.engines):
            models[f"X{i + 1}"] = DemandComponent(self.catastrophic_fraction)
        groups = (
            [BetaFactorGroup(tuple(names), self.ccf_beta)] if self.engines > 1 else []
        )
        return SystemModel(series(*parts), models, groups, ccf_basis="probability")

    def system_outcomes(self) -> ClusterOutcomes:
        """Return the outcome probabilities computed on the exact BDD.

        Returns:
            The three outcomes from :meth:`system_model`.

        Equation:
            [EQ-AERO-REL-009].

        References:
            [REF-MOSLEH-ETAL-1998], [REF-RAUZY-1993].

        """
        success = self.system_model().reliability(0.0)
        clean = self.system_model(catastrophe_only=True).reliability(0.0)
        return ClusterOutcomes(
            success, max(0.0, clean - success), max(0.0, 1.0 - clean)
        )


@dataclass(frozen=True, slots=True)
class ClusterUncertainty:
    """Epistemic laws of an engine cluster's three failure parameters.

    Each law is a :class:`~probstat.distributions.Beta` (for instance the
    posterior of :func:`probstat.reliability.demand_failure_probability`), a
    :class:`BetaMixture`, or a constant in ``[0, 1]``. One draw is shared by
    all engines of the cluster.

    Args:
        engines: Number of engines ``n``.
        required: Engines ``k`` that must work.
        failure_probability: Law of the engine outage probability ``q``.
        ccf_beta: Law of the beta factor (constant ``0`` for ``n = 1``).
        catastrophic_fraction: Law of the catastrophic fraction ``c``.

    References:
        [REF-APOSTOLAKIS-KAPLAN-1981], [EQ-AERO-REL-011].

    """

    engines: int
    required: int
    failure_probability: Law
    ccf_beta: Law = 0.0
    catastrophic_fraction: Law = 0.0

    def __post_init__(self) -> None:
        """Validate the structure and the laws.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        n = _count(self.engines, name="engines", minimum=1)
        k = _count(self.required, name="required", minimum=1)
        if k > n:
            raise ValueError("required must not exceed engines")
        beta = _check_law(self.ccf_beta, name="ccf_beta")
        if n == 1 and beta != 0.0:
            raise ValueError("ccf_beta must be the constant zero for a single engine")
        object.__setattr__(self, "engines", n)
        object.__setattr__(self, "required", k)
        object.__setattr__(
            self,
            "failure_probability",
            _check_law(self.failure_probability, name="failure_probability"),
        )
        object.__setattr__(self, "ccf_beta", beta)
        object.__setattr__(
            self,
            "catastrophic_fraction",
            _check_law(self.catastrophic_fraction, name="catastrophic_fraction"),
        )

    def laws(self, prefix: str = "") -> dict[str, Law]:
        """Return the named laws, keyed ``prefix + parameter``.

        Args:
            prefix: Name prefix, for example ``"stage1."``.

        Returns:
            The three laws.

        References:
            [EQ-AERO-REL-011].

        """
        return {
            f"{prefix}failure_probability": self.failure_probability,
            f"{prefix}ccf_beta": self.ccf_beta,
            f"{prefix}catastrophic_fraction": self.catastrophic_fraction,
        }

    def success(self, values: Mapping[str, float], prefix: str = "") -> float:
        """Return the cluster success probability at one parameter draw.

        Args:
            values: Parameter values keyed as in :meth:`laws`.
            prefix: Name prefix used in ``values``.

        Returns:
            ``P(success)``.

        Equation:
            [EQ-AERO-REL-009].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        return _cluster_probabilities(
            self.engines,
            self.required,
            values[f"{prefix}failure_probability"],
            values[f"{prefix}ccf_beta"],
            values[f"{prefix}catastrophic_fraction"],
        )[0]


# --- exact moments by polynomial expansion in (q, beta, c) -------------------
_Polynomial: TypeAlias = dict[tuple[int, int, int], float]


def _poly_mul(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    """Return the product of two polynomials in ``(q, beta, c)``.

    Args:
        left: Coefficients keyed by exponent triples.
        right: Coefficients keyed by exponent triples.

    Returns:
        The product polynomial.

    References:
        [EQ-AERO-REL-010].

    """
    result: _Polynomial = {}
    for (a1, a2, a3), x in left.items():
        for (b1, b2, b3), y in right.items():
            key = (a1 + b1, a2 + b2, a3 + b3)
            result[key] = result.get(key, 0.0) + x * y
    return result


def _poly_add(left: _Polynomial, right: _Polynomial, scale: float = 1.0) -> _Polynomial:
    """Return ``left + scale * right``.

    Args:
        left: Coefficients keyed by exponent triples.
        right: Coefficients keyed by exponent triples.
        scale: Multiplier of ``right``.

    Returns:
        The sum polynomial.

    References:
        [EQ-AERO-REL-010].

    """
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0.0) + scale * value
    return result


def _success_polynomial(engines: int, required: int, *, with_ccf: bool) -> _Polynomial:
    """Return ``P(success)`` as a polynomial in ``(q, beta, c)``.

    Args:
        engines: Engines ``n``.
        required: Engines ``k``.
        with_ccf: Whether ``beta`` is a variable (``False``: ``beta = 0``).

    Returns:
        The polynomial of :func:`_cluster_probabilities` (success component).

    References:
        [EQ-AERO-REL-009], [EQ-AERO-REL-010].

    """
    one: _Polynomial = {(0, 0, 0): 1.0}
    q_ind: _Polynomial = {(1, 0, 0): 1.0}
    q_ccf: _Polynomial = {}
    if with_ccf:
        q_ind = {(1, 0, 0): 1.0, (1, 1, 0): -1.0}
        q_ccf = {(1, 1, 0): 1.0}
    one_minus_c = _poly_add(one, {(0, 0, 1): 1.0}, -1.0)
    benign = _poly_mul(q_ind, one_minus_c)
    survive = _poly_add(one, q_ind, -1.0)
    total: _Polynomial = {}
    for b in range(engines - required + 1):
        term = one
        for _ in range(b):
            term = _poly_mul(term, benign)
        for _ in range(engines - b):
            term = _poly_mul(term, survive)
        total = _poly_add(total, term, float(math.comb(engines, b)))
    return _poly_mul(_poly_add(one, q_ccf, -1.0), total)


def cluster_reliability_moments(uncertainty: ClusterUncertainty) -> tuple[float, float]:
    """Return the exact mean and variance of the cluster success probability.

    With independent epistemic laws, the success probability is a polynomial in
    ``(q, beta, c)``; its mean and second moment follow from the raw moments of
    the Beta (or mixture) laws, with no sampling. It is the oracle against
    which :func:`cluster_reliability_distribution` is checked. Expanding the
    polynomial loses a few digits to cancellation for large clusters.

    Args:
        uncertainty: Cluster structure and parameter laws.

    Returns:
        ``(mean, variance)`` of the success probability.

    Raises:
        TypeError: If ``uncertainty`` is not a :class:`ClusterUncertainty`.

    Equation:
        [EQ-AERO-REL-010].

    References:
        [REF-MOSLEH-ETAL-1998], [REF-GELMAN-ETAL-2013].

    """
    if not isinstance(uncertainty, ClusterUncertainty):
        raise TypeError("uncertainty must be a ClusterUncertainty")
    polynomial = _success_polynomial(
        uncertainty.engines, uncertainty.required, with_ccf=uncertainty.engines > 1
    )
    laws = (
        uncertainty.failure_probability,
        uncertainty.ccf_beta,
        uncertainty.catastrophic_fraction,
    )

    def expectation(poly: _Polynomial) -> float:
        """Return ``E[poly]`` under independent input laws.

        References:
            [EQ-AERO-REL-010].

        """
        return math.fsum(
            coefficient
            * _law_moment(laws[0], i)
            * _law_moment(laws[1], j)
            * _law_moment(laws[2], k)
            for (i, j, k), coefficient in poly.items()
        )

    mean = expectation(polynomial)
    second = expectation(_poly_mul(polynomial, polynomial))
    return mean, max(0.0, second - mean * mean)


@dataclass(frozen=True, slots=True)
class ReliabilityDistribution:
    """Distribution of a reliability under epistemic uncertainty.

    Args:
        mean: Sample mean.
        standard_deviation: Sample standard deviation.
        interval: Equal-tailed credible interval.
        level: Credible level of the interval.
        values: Sorted sample of the reliability.
        input_names: Names of the *uncertain* inputs (constants omitted).
        sobol_first: First-order Sobol indices in that order, or ``None``
            when no input is uncertain.
        sobol_total: Total-effect indices, or ``None``.
        sobol_samples: Base sample size of the Sobol design.

    References:
        [REF-SALTELLI-2010], [EQ-AERO-REL-011].

    """

    mean: float
    standard_deviation: float
    interval: tuple[float, float]
    level: float
    values: tuple[float, ...]
    input_names: tuple[str, ...]
    sobol_first: tuple[float, ...] | None
    sobol_total: tuple[float, ...] | None
    sobol_samples: int

    @property
    def ranking(self) -> tuple[str, ...]:
        """Return the uncertain inputs sorted by total-effect index, largest first.

        References:
            [REF-SALTELLI-2010].

        """
        if self.sobol_total is None:
            return ()
        order = sorted(
            range(len(self.input_names)), key=lambda i: -(self.sobol_total or ())[i]
        )
        return tuple(self.input_names[i] for i in order)


def _empirical_quantile(sorted_values: Sequence[float], level: float) -> float:
    """Return the type-7 (linear interpolation) quantile of sorted data.

    Args:
        sorted_values: Non-empty ascending data.
        level: Level in ``[0, 1]``.

    Returns:
        The quantile.

    References:
        [EQ-AERO-REL-011].

    """
    position = level * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def propagate_reliability(
    model: Callable[[Mapping[str, float]], float],
    inputs: Mapping[str, Law],
    *,
    samples: int = 4000,
    sobol_samples: int = 1000,
    level: float = 0.90,
    seed: int | RandomStream | None = 0,
) -> ReliabilityDistribution:
    """Propagate epistemic Beta-type inputs through a reliability function.

    Draws ``samples`` independent parameter sets (each input from its own
    child stream), evaluates ``model`` on each, and reports the mean, the
    standard deviation, the equal-tailed credible interval and a Sobol
    first-order and total-effect ranking of the uncertain inputs, estimated by
    :func:`probstat.sensitivity.estimate_sobol_indices` on two further
    independent samples of ``sobol_samples`` rows each. Constant inputs are
    held fixed and left out of the ranking.

    Args:
        model: Maps the named parameter values to a reliability.
        inputs: Law (Beta, mixture or constant) of every parameter.
        samples: Monte Carlo sample size, at least 10.
        sobol_samples: Sobol base sample size, at least 10.
        level: Credible level in ``(0, 1)``.
        seed: Integer seed or stream.

    Returns:
        The distribution summary.

    Raises:
        TypeError: If ``model`` is not callable.
        ValueError: If an argument is invalid or the Sobol design has zero
            output variance.

    Equation:
        [EQ-AERO-REL-011].

    References:
        [REF-SALTELLI-2010], [REF-APOSTOLAKIS-KAPLAN-1981].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    laws = {name: _check_law(law, name=name) for name, law in inputs.items()}
    if not laws:
        raise ValueError("at least one input is required")
    count = _count(samples, name="samples", minimum=10)
    base = _count(sobol_samples, name="sobol_samples", minimum=10)
    credible = _open_probability(level, name="level")
    stream = resolve_stream(seed)
    names = tuple(laws)
    columns = {
        name: _draw(laws[name], count, stream.child("mc", name)) for name in names
    }
    values = sorted(
        model({name: columns[name][i] for name in names}) for i in range(count)
    )
    mean = math.fsum(values) / count
    variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
    tail = 0.5 * (1.0 - credible)
    interval = (
        _empirical_quantile(values, tail),
        _empirical_quantile(values, 1.0 - tail),
    )
    uncertain = tuple(
        name for name in names if isinstance(laws[name], Beta | BetaMixture)
    )
    first: tuple[float, ...] | None = None
    total: tuple[float, ...] | None = None
    if uncertain:
        fixed = {name: float(laws[name]) for name in names if name not in uncertain}  # type: ignore[arg-type]
        design = {
            tag: [
                _draw(laws[name], base, stream.child(tag, name)) for name in uncertain
            ]
            for tag in ("A", "B")
        }
        matrix_a = tuple(zip(*design["A"], strict=True))
        matrix_b = tuple(zip(*design["B"], strict=True))

        def row_model(row: tuple[float, ...]) -> float:
            """Evaluate the reliability at one row of the Sobol design.

            References:
                [EQ-AERO-REL-011].

            """
            point = dict(fixed)
            point.update(zip(uncertain, row, strict=True))
            return model(point)

        result = estimate_sobol_indices(row_model, matrix_a, matrix_b)
        first, total = result.first_order, result.total_effect
    return ReliabilityDistribution(
        mean,
        math.sqrt(variance),
        interval,
        credible,
        tuple(values),
        uncertain,
        first,
        total,
        base if uncertain else 0,
    )


def cluster_reliability_distribution(
    uncertainty: ClusterUncertainty,
    *,
    samples: int = 4000,
    sobol_samples: int = 1000,
    level: float = 0.90,
    seed: int | RandomStream | None = 0,
) -> ReliabilityDistribution:
    """Return the distribution of the cluster success probability.

    Args:
        uncertainty: Cluster structure and parameter laws.
        samples: Monte Carlo sample size.
        sobol_samples: Sobol base sample size.
        level: Credible level of the interval.
        seed: Integer seed or stream.

    Returns:
        Mean, spread, credible interval and the Sobol ranking of ``q``,
        ``beta`` and ``c``.

    Raises:
        TypeError: If ``uncertainty`` is not a :class:`ClusterUncertainty`.

    Equation:
        [EQ-AERO-REL-011].

    References:
        [REF-SALTELLI-2010], [REF-APOSTOLAKIS-KAPLAN-1981].

    """
    if not isinstance(uncertainty, ClusterUncertainty):
        raise TypeError("uncertainty must be a ClusterUncertainty")
    return propagate_reliability(
        uncertainty.success,
        uncertainty.laws(),
        samples=samples,
        sobol_samples=sobol_samples,
        level=level,
        seed=seed,
    )


# ------------------------------------------------ 4. launch-vehicle event tree
@dataclass(frozen=True, slots=True)
class LaunchVehicleDesign:
    """A two-stage launch vehicle described by its failure probabilities.

    Args:
        first_stage: Clustered first-stage engines.
        separation_failure: Probability ``q_sep`` that stage separation fails
            (classified as loss of mission).
        upper_ignition_failure: Probability that the upper stage does not
            ignite (loss of mission).
        upper_stage: Upper-stage engine cluster (usually one engine).

    References:
        [REF-NASA-PRA-2011], [EQ-AERO-REL-012].

    """

    first_stage: EngineClusterModel
    separation_failure: float
    upper_ignition_failure: float
    upper_stage: EngineClusterModel

    def __post_init__(self) -> None:
        """Validate the probabilities and the stage models.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        for stage in (self.first_stage, self.upper_stage):
            if not isinstance(stage, EngineClusterModel):
                raise TypeError("stages must be EngineClusterModel instances")
        object.__setattr__(
            self,
            "separation_failure",
            require_probability(self.separation_failure, name="separation_failure"),
        )
        object.__setattr__(
            self,
            "upper_ignition_failure",
            require_probability(
                self.upper_ignition_failure, name="upper_ignition_failure"
            ),
        )


def _stage_heading(
    name: str, outcomes: ClusterOutcomes, nxt: Heading | Outcome
) -> Heading:
    """Return a three-way heading for a cluster stage.

    Args:
        name: Heading label.
        outcomes: Cluster outcome probabilities.
        nxt: Successor after a successful stage.

    Returns:
        The heading with ``success``, ``benign_shortfall`` and
        ``catastrophic`` branches.

    References:
        [EQ-AERO-REL-012].

    """
    return Heading(
        name,
        (
            Branch("success", outcomes.success, nxt),
            Branch(
                "benign_shortfall",
                outcomes.benign_shortfall,
                Outcome("loss_of_mission", 1.0),
            ),
            Branch(
                "catastrophic", outcomes.catastrophic, Outcome("loss_of_vehicle", 1.0)
            ),
        ),
    )


def launch_event_tree(design: LaunchVehicleDesign) -> EventTree:
    """Build the launch event tree of the vehicle.

    Headings in order: first-stage cluster (success, benign shortfall,
    catastrophic), stage separation, upper-stage ignition, upper-stage engine
    (same three branches). The initiating event is the launch attempt with
    frequency one, so sequence frequencies are mission probabilities. The end
    states are ``success`` (consequence 0), ``loss_of_mission`` and
    ``loss_of_vehicle`` (consequence 1, a payload-loss indicator); the
    assignment of separation and ignition failures to loss of mission is an
    analyst's assumption.

    Args:
        design: The vehicle.

    Returns:
        The :class:`~probstat.risk.EventTree`.

    Raises:
        TypeError: If ``design`` is not a :class:`LaunchVehicleDesign`.

    Equation:
        [EQ-AERO-REL-012].

    References:
        [REF-NASA-PRA-2011], [REF-KAPLAN-GARRICK-1981].

    """
    if not isinstance(design, LaunchVehicleDesign):
        raise TypeError("design must be a LaunchVehicleDesign")
    upper = _stage_heading(
        "upper_stage_engine", design.upper_stage.outcomes(), Outcome("success", 0.0)
    )
    ignition = binary_heading(
        "upper_stage_ignition",
        design.upper_ignition_failure,
        upper,
        Outcome("loss_of_mission", 1.0),
    )
    separation = binary_heading(
        "stage_separation",
        design.separation_failure,
        ignition,
        Outcome("loss_of_mission", 1.0),
    )
    root = _stage_heading(
        "first_stage_cluster", design.first_stage.outcomes(), separation
    )
    return EventTree("launch_attempt", 1.0, root)


def vehicle_outcome_probabilities(design: LaunchVehicleDesign) -> Mapping[str, float]:
    """Return the probability of each end state from the event tree.

    Args:
        design: The vehicle.

    Returns:
        Probabilities of ``success``, ``loss_of_mission`` and
        ``loss_of_vehicle`` (they sum to one).

    Equation:
        [EQ-AERO-REL-012].

    References:
        [REF-NASA-PRA-2011], [REF-KAPLAN-GARRICK-1981].

    """
    return MappingProxyType(dict(launch_event_tree(design).outcome_frequencies()))


def vehicle_reliability(design: LaunchVehicleDesign) -> float:
    """Return the probability of mission success from the event tree.

    Args:
        design: The vehicle.

    Returns:
        ``P(success)``, equal to the product of the stage success factors.

    Equation:
        [EQ-AERO-REL-012].

    References:
        [REF-NASA-PRA-2011], [REF-KAPLAN-GARRICK-1981].

    """
    return vehicle_outcome_probabilities(design)["success"]


@dataclass(frozen=True, slots=True)
class VehicleUncertainty:
    """Epistemic laws of a two-stage vehicle.

    Args:
        first_stage: First-stage cluster uncertainty.
        separation_failure: Law of the separation failure probability.
        upper_ignition_failure: Law of the upper-stage ignition failure
            probability.
        upper_stage: Upper-stage cluster uncertainty.

    References:
        [REF-APOSTOLAKIS-KAPLAN-1981], [EQ-AERO-REL-012].

    """

    first_stage: ClusterUncertainty
    separation_failure: Law
    upper_ignition_failure: Law
    upper_stage: ClusterUncertainty

    def __post_init__(self) -> None:
        """Validate the stage uncertainties and laws.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        for stage in (self.first_stage, self.upper_stage):
            if not isinstance(stage, ClusterUncertainty):
                raise TypeError("stages must be ClusterUncertainty instances")
        object.__setattr__(
            self,
            "separation_failure",
            _check_law(self.separation_failure, name="separation_failure"),
        )
        object.__setattr__(
            self,
            "upper_ignition_failure",
            _check_law(self.upper_ignition_failure, name="upper_ignition_failure"),
        )


def vehicle_reliability_distribution(
    uncertainty: VehicleUncertainty,
    *,
    samples: int = 4000,
    sobol_samples: int = 500,
    level: float = 0.90,
    seed: int | RandomStream | None = 0,
) -> ReliabilityDistribution:
    """Return the distribution of the vehicle success probability.

    The reliability at each draw is the event-tree success probability, which
    is the product ``P_1 (1 - q_sep)(1 - q_ign) P_2`` of the stage success
    factors; the product form is evaluated directly in the loop (it is tested
    to equal :func:`vehicle_reliability`).

    Args:
        uncertainty: Vehicle uncertainty.
        samples: Monte Carlo sample size.
        sobol_samples: Sobol base sample size (the design costs
            ``N (2 + d + d (d - 1) / 2)`` evaluations for ``d`` uncertain
            inputs).
        level: Credible level of the interval.
        seed: Integer seed or stream.

    Returns:
        Mean, spread, credible interval and the Sobol ranking.

    Raises:
        TypeError: If ``uncertainty`` is not a :class:`VehicleUncertainty`.

    Equation:
        [EQ-AERO-REL-012], [EQ-AERO-REL-011].

    References:
        [REF-NASA-PRA-2011], [REF-SALTELLI-2010].

    """
    if not isinstance(uncertainty, VehicleUncertainty):
        raise TypeError("uncertainty must be a VehicleUncertainty")
    inputs: dict[str, Law] = {}
    inputs.update(uncertainty.first_stage.laws("stage1."))
    inputs["separation_failure"] = uncertainty.separation_failure
    inputs["upper_ignition_failure"] = uncertainty.upper_ignition_failure
    inputs.update(uncertainty.upper_stage.laws("stage2."))

    def success(values: Mapping[str, float]) -> float:
        """Return the product of stage success factors at one draw.

        References:
            [EQ-AERO-REL-012].

        """
        return (
            uncertainty.first_stage.success(values, "stage1.")
            * (1.0 - values["separation_failure"])
            * (1.0 - values["upper_ignition_failure"])
            * uncertainty.upper_stage.success(values, "stage2.")
        )

    return propagate_reliability(
        success,
        inputs,
        samples=samples,
        sobol_samples=sobol_samples,
        level=level,
        seed=seed,
    )
