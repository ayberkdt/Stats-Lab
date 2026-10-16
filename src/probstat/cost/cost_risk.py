"""Cost-risk analysis: total-cost S-curves, confidence levels, and reserves.

The total cost is a weighted sum of uncertain element costs,
``C = c_0 + sum_i a_i X_i``. Two methods give its distribution (the
"S-curve", i.e. its CDF), following the NASA Cost Estimating Handbook v4.0
practice:

1. **Method of moments.** The mean ``sum a_i mu_i`` and the variance
   ``sum_i sum_j a_i a_j rho_ij sigma_i sigma_j`` are *exact* for any element
   laws given their Pearson correlations ``rho_ij``. A lognormal (default) or
   normal law with those two moments is then *assumed* for the total; its
   percentiles are an approximation whose quality depends on the elements.
2. **Monte Carlo.** Element costs are sampled jointly with their marginal
   laws and a Gaussian copula calibrated (Nataf) to the same Pearson
   correlations, through :class:`~probstat.modeling.JointInput`. Percentiles
   are sample quantiles with sampling error that shrinks like ``1/sqrt(n)``.

**Correlation matters.** With positive correlations the variance of the sum
exceeds the independent value ``sum a_i^2 sigma_i^2``; analyses that omit
correlation therefore understate the spread of the total and the reserve
needed at a given confidence level (Garvey 2000; NASA CEH v4.0). Every result
reports the independent spread next to the correlated one so the effect is
visible.

The *contingency* (reserve) at confidence level ``q`` is the ``q``-quantile of
the total minus the point estimate.

A joint cost-schedule confidence level (JCL) is the probability that cost and
schedule are *both* within their targets, estimated from joint samples.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.distributions import LogNormal, Normal
from probstat.foundations import RandomStream, require_finite_real, require_probability
from probstat.modeling import JointInput, RandomInput

from .life_cycle import LifeCycleCostModel
from .moments import MomentTotal, _correlation, method_of_moments_total

FAMILIES = ("lognormal", "normal")
DEFAULT_LEVELS = (0.5, 0.7, 0.8, 0.9)


def fit_total_distribution(
    mean: float, standard_deviation: float, family: str = "lognormal"
) -> LogNormal | Normal:
    """Return the lognormal or normal law with a given mean and deviation.

    Args:
        mean: Mean of the total (positive for the lognormal).
        standard_deviation: Positive standard deviation.
        family: ``"lognormal"`` or ``"normal"``.

    Returns:
        For the lognormal, ``sigma^2 = log(1 + s^2/m^2)`` and
        ``mu = log m - sigma^2 / 2``.

    Raises:
        ValueError: If the family is unknown or the moments are invalid.

    Equation:
        [EQ-COST-MOM-002].

    References:
        [REF-GARVEY-2000], [REF-JKB-1994, Chapter 14].

    """
    m = require_finite_real(mean, name="mean")
    s = require_finite_real(standard_deviation, name="standard_deviation")
    if s <= 0.0:
        raise ValueError("standard_deviation must be positive")
    if family == "normal":
        return Normal(m, s)
    if family != "lognormal":
        raise ValueError("family must be 'lognormal' or 'normal'")
    if m <= 0.0:
        raise ValueError("a lognormal total requires a positive mean")
    sigma2 = math.log1p((s / m) ** 2)
    return LogNormal(math.log(m) - 0.5 * sigma2, math.sqrt(sigma2))


def _sample_quantile(ordered: Sequence[float], level: float) -> float:
    """Return the lower empirical quantile ``x_(ceil(n q))`` of sorted data.

    Args:
        ordered: Ascending sample.
        level: Level in ``(0, 1)``.

    Returns:
        Order statistic (Hyndman-Fan type 1).

    Equation:
        [EQ-RISK-VAR-001].

    References:
        [REF-HYNDMAN-FAN-1996].

    """
    index = max(1, math.ceil(len(ordered) * level - 1e-10))
    return ordered[min(index, len(ordered)) - 1]


@dataclass(frozen=True, slots=True)
class ConfidenceLevelRow:
    """Total cost at one confidence level by every method.

    Args:
        level: Confidence level ``q``.
        analytic: Quantile of the fitted correlated total.
        analytic_independent: Same family fitted with correlation ignored.
        monte_carlo: Empirical quantile of the simulated totals.
        contingency_analytic: ``analytic - point estimate``.
        contingency_monte_carlo: ``monte_carlo - point estimate``.

    References:
        [REF-NASA-CEH-2015].

    """

    level: float
    analytic: float
    analytic_independent: float
    monte_carlo: float
    contingency_analytic: float
    contingency_monte_carlo: float


@dataclass(frozen=True)
class CostRiskResult:
    """Total-cost S-curve by the method of moments and by Monte Carlo.

    Args:
        point_estimate: Deterministic estimate the reserve is measured from.
        moments: Exact moments of the total (correlated and independent).
        family: Assumed analytic family of the total.
        analytic_law: Fitted law of the correlated total.
        independent_law: Fitted law with correlation ignored.
        samples: Ascending Monte Carlo totals.
        monte_carlo_mean: Sample mean.
        monte_carlo_standard_deviation: Sample standard deviation.
        monte_carlo_standard_error: Standard error of the sample mean.
        rows: One :class:`ConfidenceLevelRow` per requested level.
        notes: Assumptions and warnings in plain language.

    References:
        [REF-NASA-CEH-2015], [REF-GARVEY-2000].

    """

    point_estimate: float
    moments: MomentTotal
    family: str
    analytic_law: LogNormal | Normal
    independent_law: LogNormal | Normal | None
    samples: tuple[float, ...]
    monte_carlo_mean: float
    monte_carlo_standard_deviation: float
    monte_carlo_standard_error: float
    rows: tuple[ConfidenceLevelRow, ...]
    notes: tuple[str, ...]

    def analytic_cdf(self, cost: float) -> float:
        """Return the fitted S-curve ``P(C <= cost)``.

        Equation:
            [EQ-COST-MOM-002].

        References:
            [REF-NASA-CEH-2015].

        """
        return float(self.analytic_law.cdf(require_finite_real(cost, name="cost")))

    def monte_carlo_cdf(self, cost: float) -> float:
        """Return the empirical S-curve: the fraction of totals ``<= cost``.

        Equation:
            [EQ-COST-SCURVE-001].

        References:
            [REF-NASA-CEH-2015].

        """
        value = require_finite_real(cost, name="cost")
        return bisect.bisect_right(self.samples, value) / len(self.samples)

    def confidence_of_point_estimate(self) -> tuple[float, float]:
        """Return ``(analytic, Monte Carlo)`` confidence of the point estimate.

        A point estimate usually sits well below the 50th percentile because
        cost distributions are right-skewed.

        Equation:
            [EQ-COST-SCURVE-001].

        References:
            [REF-NASA-CEH-2015].

        """
        return (
            self.analytic_cdf(self.point_estimate),
            self.monte_carlo_cdf(self.point_estimate),
        )

    def s_curve(self, points: int = 101) -> tuple[tuple[float, float, float], ...]:
        """Return ``(cost, analytic CDF, Monte Carlo CDF)`` on a cost grid.

        Args:
            points: Number of equally spaced costs from the smallest to the
                largest simulated total (at least two).

        Returns:
            S-curve table for plotting.

        Equation:
            [EQ-COST-SCURVE-001], [EQ-COST-MOM-002].

        References:
            [REF-NASA-CEH-2015].

        """
        if isinstance(points, bool) or not isinstance(points, int) or points < 2:
            raise ValueError("points must be an integer of at least two")
        low, high = self.samples[0], self.samples[-1]
        width = (high - low) or 1.0
        grid = [low + width * k / (points - 1) for k in range(points)]
        return tuple((c, self.analytic_cdf(c), self.monte_carlo_cdf(c)) for c in grid)


def cost_risk_analysis(
    elements: Sequence[RandomInput],
    correlation: Sequence[Sequence[float]] | None = None,
    *,
    weights: Sequence[float] | None = None,
    constant: float = 0.0,
    point_estimate: float | None = None,
    levels: Iterable[float] = DEFAULT_LEVELS,
    family: str = "lognormal",
    samples: int = 20_000,
    seed: int | RandomStream | None = None,
) -> CostRiskResult:
    """Return the total-cost S-curve, confidence levels, and contingency.

    Args:
        elements: Uncertain element costs (distinct names).
        correlation: Pearson correlation matrix of the element costs;
            ``None`` means independence (a note warns that this usually
            understates the spread).
        weights: Coefficients ``a_i`` (e.g. present-value factors; default 1).
        constant: Deterministic cost added to the total.
        point_estimate: Deterministic estimate for the contingency; defaults
            to the mean total (then contingency is quantile minus mean).
        levels: Confidence levels strictly inside ``(0, 1)``.
        family: Analytic family for the total, ``"lognormal"`` or ``"normal"``.
        samples: Monte Carlo sample size (at least 100).
        seed: Integer seed or stream for reproducibility.

    Returns:
        Both S-curves, the confidence-level table, and notes.

    Raises:
        ValueError: If inputs are inconsistent or a Pearson correlation is
            not attainable for the marginals.

    Equation:
        [EQ-COST-MOM-001], [EQ-COST-MOM-002], [EQ-COST-SCURVE-001],
        [EQ-MOD-NATAF-002].

    References:
        [REF-NASA-CEH-2015], [REF-GARVEY-2000],
        [REF-LIU-DERKIUREGHIAN-1986].

    """
    inputs = tuple(elements)
    if not inputs:
        raise ValueError("at least one element is required")
    for item in inputs:
        if not isinstance(item, RandomInput):
            raise TypeError("elements must be RandomInput instances")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("samples must be an integer of at least 100")
    checked_levels = tuple(require_probability(q, name="level") for q in levels)
    if not checked_levels or any(q in (0.0, 1.0) for q in checked_levels):
        raise ValueError("levels must lie strictly inside (0, 1)")
    a = (
        tuple(1.0 for _ in inputs)
        if weights is None
        else tuple(require_finite_real(v, name="weight") for v in weights)
    )
    rho = None if correlation is None else _correlation(correlation, len(inputs))
    moments = method_of_moments_total(
        [item.mean for item in inputs],
        [item.standard_deviation for item in inputs],
        rho,
        weights=a,
        constant=constant,
    )
    law = fit_total_distribution(moments.mean, moments.standard_deviation, family)
    independent_law = (
        fit_total_distribution(
            moments.mean, moments.independent_standard_deviation, family
        )
        if moments.independent_standard_deviation > 0.0
        else None
    )
    joint = JointInput(inputs) if rho is None else JointInput.from_pearson(inputs, rho)
    base = require_finite_real(constant, name="constant")
    totals = sorted(
        base + math.fsum(w * x for w, x in zip(a, draw, strict=True))
        for draw in joint.sample(samples, seed=seed)
    )
    mc_mean = math.fsum(totals) / samples
    mc_sd = math.sqrt(math.fsum((t - mc_mean) ** 2 for t in totals) / (samples - 1))
    point = (
        moments.mean
        if point_estimate is None
        else require_finite_real(point_estimate, name="point_estimate")
    )
    rows = []
    for q in checked_levels:
        analytic = float(law.quantile(q))
        independent = (
            float(independent_law.quantile(q))
            if independent_law is not None
            else moments.mean
        )
        empirical = _sample_quantile(totals, q)
        rows.append(
            ConfidenceLevelRow(
                q, analytic, independent, empirical, analytic - point, empirical - point
            )
        )
    notes = [
        f"Analytic S-curve assumes a {family} total with the exact mean and "
        "variance; its percentiles are approximate.",
        f"Monte Carlo percentiles carry sampling error (n = {samples}).",
    ]
    if rho is None:
        notes.append(
            "No correlation was supplied: elements are independent. Positive "
            "correlation between cost elements is typical, and omitting it "
            "understates the spread of the total and the reserve."
        )
    else:
        notes.append(
            "Ignoring the supplied correlation would shrink the standard "
            f"deviation by {100.0 * moments.spread_understatement:.1f} %."
        )
    if point_estimate is None:
        notes.append("Point estimate defaulted to the mean total.")
    return CostRiskResult(
        point,
        moments,
        family,
        law,
        independent_law,
        tuple(totals),
        mc_mean,
        mc_sd,
        mc_sd / math.sqrt(samples),
        tuple(rows),
        tuple(notes),
    )


def life_cycle_cost_risk(
    model: LifeCycleCostModel,
    correlation: Sequence[Sequence[float]] | None = None,
    *,
    present_value: bool = True,
    point_estimate: float | None = None,
    levels: Iterable[float] = DEFAULT_LEVELS,
    family: str = "lognormal",
    samples: int = 20_000,
    seed: int | RandomStream | None = None,
) -> CostRiskResult:
    """Run :func:`cost_risk_analysis` on a life-cycle cost model.

    Args:
        model: Cost breakdown structure with at least one random element.
        correlation: Pearson correlation of the random elements, in
            ``model.random_elements`` order.
        present_value: Use present-value coefficients (default) or the
            undiscounted multiplier sums.
        point_estimate: Deterministic estimate (default: mean total).
        levels: Confidence levels.
        family: Analytic family.
        samples: Monte Carlo sample size.
        seed: Seed or stream.

    Returns:
        Cost-risk result for the life-cycle total.

    Equation:
        [EQ-COST-LCC-001], [EQ-COST-MOM-001].

    References:
        [REF-NASA-CEH-2015], [REF-FABRYCKY-BLANCHARD-1991].

    """
    if not isinstance(model, LifeCycleCostModel):
        raise TypeError("model must be a LifeCycleCostModel")
    coefficients = (
        model.coefficients()
        if present_value
        else {e.name: e.multiplier_sum for e in model.elements}
    )
    randoms = model.random_elements
    if not randoms:
        raise ValueError("the model has no random cost elements")
    constant = math.fsum(
        coefficients[e.name] * e.mean for e in model.elements if not e.is_random
    )
    inputs = model.joint_input().inputs
    return cost_risk_analysis(
        inputs,
        correlation,
        weights=[coefficients[e.name] for e in randoms],
        constant=constant,
        point_estimate=point_estimate,
        levels=levels,
        family=family,
        samples=samples,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class JointConfidenceLevel:
    """Joint cost-schedule confidence level from paired samples.

    Args:
        cost_target: Cost target ``C``.
        schedule_target: Schedule target ``S``.
        joint: ``P(cost <= C and schedule <= S)`` (sample fraction).
        cost_only: ``P(cost <= C)``.
        schedule_only: ``P(schedule <= S)``.
        standard_error: Binomial standard error of ``joint``.
        sample_size: Number of paired samples.

    References:
        [REF-NASA-CEH-2015].

    """

    cost_target: float
    schedule_target: float
    joint: float
    cost_only: float
    schedule_only: float
    standard_error: float
    sample_size: int


def _paired(
    costs: Iterable[float], schedules: Iterable[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate paired cost and schedule samples.

    Args:
        costs: Finite cost samples.
        schedules: Finite schedule samples of equal length.

    Returns:
        Both samples as tuples.

    Raises:
        ValueError: If they are empty or of different lengths.

    References:
        Project validation policy. No equation is evaluated.

    """
    c = tuple(require_finite_real(v, name="cost") for v in costs)
    s = tuple(require_finite_real(v, name="schedule") for v in schedules)
    if not c or len(c) != len(s):
        raise ValueError("costs and schedules must be non-empty and paired")
    return c, s


def joint_confidence_level(
    costs: Iterable[float],
    schedules: Iterable[float],
    cost_target: float,
    schedule_target: float,
) -> JointConfidenceLevel:
    """Return the JCL ``P(cost <= C and schedule <= S)`` from joint samples.

    The samples must come from one joint simulation (for example a
    :class:`~probstat.modeling.ProbabilisticModel` with both outputs) so
    that their dependence is preserved.

    Args:
        costs: Cost samples.
        schedules: Schedule samples paired with ``costs``.
        cost_target: Cost target ``C``.
        schedule_target: Schedule target ``S``.

    Returns:
        Joint and marginal confidence levels with a standard error.

    Equation:
        [EQ-COST-JCL-001].

    References:
        [REF-NASA-CEH-2015].

    """
    c, s = _paired(costs, schedules)
    cap_c = require_finite_real(cost_target, name="cost_target")
    cap_s = require_finite_real(schedule_target, name="schedule_target")
    n = len(c)
    joint = sum(1 for x, y in zip(c, s, strict=True) if x <= cap_c and y <= cap_s) / n
    return JointConfidenceLevel(
        cap_c,
        cap_s,
        joint,
        sum(1 for x in c if x <= cap_c) / n,
        sum(1 for y in s if y <= cap_s) / n,
        math.sqrt(joint * (1.0 - joint) / n),
        n,
    )


def cost_for_joint_confidence(
    costs: Iterable[float],
    schedules: Iterable[float],
    schedule_target: float,
    level: float,
) -> float | None:
    """Return the smallest sampled cost ``C`` with ``JCL(C, S) >= level``.

    Args:
        costs: Cost samples.
        schedules: Paired schedule samples.
        schedule_target: Schedule target ``S``.
        level: Required joint confidence in ``(0, 1)``.

    Returns:
        The cost target, or ``None`` when even an unlimited budget cannot
        reach ``level`` because ``P(schedule <= S) < level``.

    Equation:
        [EQ-COST-JCL-001].

    References:
        [REF-NASA-CEH-2015].

    """
    c, s = _paired(costs, schedules)
    cap_s = require_finite_real(schedule_target, name="schedule_target")
    q = require_probability(level, name="level")
    needed = math.ceil(q * len(c) - 1e-10)
    feasible = sorted(x for x, y in zip(c, s, strict=True) if y <= cap_s)
    if needed > len(feasible):
        return None
    if needed <= 0:
        return min(c)
    return feasible[needed - 1]
