"""Cost of unreliability and reliability-cost trade-offs.

The *cost of unreliability* is the expected cost of failures: for failure
modes with constant frequencies ``lambda_i`` (homogeneous Poisson
occurrences) and consequence costs ``C_i``, the expected cost per unit time is
``sum_i lambda_i C_i`` and, with continuous discounting at rate ``r`` over a
horizon ``H``, its present value is
``sum_i lambda_i C_i (1 - exp(-r H)) / r`` (exact by Campbell's theorem).
For one-shot missions with failure probabilities ``p_i`` it is
``sum_i p_i C_i``.

Design alternatives are compared by expected life-cycle cost *plus* this
risk cost. Expected values are the right criterion for a risk-neutral
decision maker; when the spread matters, propagate the uncertainty with
:mod:`.cost_risk` or :mod:`probstat.risk` and compare percentiles or CVaR.
:func:`reliability_decision_problem` builds a
:class:`~probstat.risk.DecisionProblem` across scenarios so that the expected
value of perfect information (EVPI) about the scenario can be computed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations import require_finite_real, require_probability
from probstat.risk import DecisionProblem


def _non_negative(value: object, *, name: str) -> float:
    """Validate a finite non-negative real.

    Args:
        value: Candidate value.
        name: Field name for error messages.

    Returns:
        The value.

    Raises:
        ValueError: If it is negative.

    References:
        Project validation policy. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def discounted_rate_factor(horizon: float, discount_rate: float = 0.0) -> float:
    """Return ``int_0^H exp(-r t) dt``: present value of a unit cost rate.

    Args:
        horizon: Non-negative horizon ``H``.
        discount_rate: Continuous rate ``r`` (``0`` gives ``H``).

    Returns:
        ``(1 - exp(-r H)) / r`` or ``H``.

    Equation:
        [EQ-COST-UNREL-001].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    h = _non_negative(horizon, name="horizon")
    r = require_finite_real(discount_rate, name="discount_rate")
    if r == 0.0:
        return h
    return -math.expm1(-r * h) / r


def expected_failure_cost(
    probabilities: Iterable[float], consequences: Iterable[float]
) -> float:
    """Return ``sum_i p_i C_i`` for mission failure modes.

    Args:
        probabilities: Failure probabilities ``p_i`` in ``[0, 1]``.
        consequences: Consequence costs ``C_i`` aligned with them.

    Returns:
        Expected failure cost of one mission.

    Raises:
        ValueError: If the sequences differ in length or are empty.

    Equation:
        [EQ-COST-UNREL-001].

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    p = tuple(require_probability(v, name="probability") for v in probabilities)
    c = tuple(require_finite_real(v, name="consequence") for v in consequences)
    if not p or len(p) != len(c):
        raise ValueError("probabilities and consequences must be non-empty and aligned")
    return math.fsum(a * b for a, b in zip(p, c, strict=True))


@dataclass(frozen=True, slots=True)
class FailureCostItem:
    """One failure mode with a constant frequency and a consequence cost.

    Args:
        name: Failure-mode label.
        frequency: Non-negative occurrences per unit time.
        consequence_cost: Cost of each occurrence.

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    name: str
    frequency: float
    consequence_cost: float

    def __post_init__(self) -> None:
        """Validate the fields.

        Raises:
            TypeError: If the name is not text.
            ValueError: If the name is blank or the frequency negative.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        object.__setattr__(
            self, "frequency", _non_negative(self.frequency, name="frequency")
        )
        object.__setattr__(
            self,
            "consequence_cost",
            require_finite_real(self.consequence_cost, name="consequence_cost"),
        )


@dataclass(frozen=True)
class UnreliabilityCost:
    """Expected cost of failures over a horizon.

    Args:
        cost_rate: ``sum lambda_i C_i`` per unit time.
        expected_cost: ``cost_rate * H`` (undiscounted).
        present_value: ``cost_rate * (1 - exp(-r H)) / r``.
        expected_failures: ``sum lambda_i H``.
        contributions: Failure mode to present-value contribution.
        shares: Failure mode to fraction of the present value.

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    cost_rate: float
    expected_cost: float
    present_value: float
    expected_failures: float
    contributions: Mapping[str, float]
    shares: Mapping[str, float]


def cost_of_unreliability(
    items: Iterable[FailureCostItem], horizon: float, *, discount_rate: float = 0.0
) -> UnreliabilityCost:
    """Return the expected (and discounted) cost of failures over ``[0, H]``.

    Args:
        items: Failure modes with distinct names.
        horizon: Non-negative horizon ``H``.
        discount_rate: Continuous discount rate ``r``.

    Returns:
        Totals and per-mode contributions.

    Raises:
        ValueError: If no item is given or names repeat.

    Equation:
        [EQ-COST-UNREL-001].

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021], [REF-BLANK-TARQUIN-2012].

    """
    modes = tuple(items)
    if not modes:
        raise ValueError("at least one failure mode is required")
    names = [m.name for m in modes]
    if len(set(names)) != len(names):
        raise ValueError("failure-mode names must be distinct")
    h = _non_negative(horizon, name="horizon")
    factor = discounted_rate_factor(h, discount_rate)
    rate = math.fsum(m.frequency * m.consequence_cost for m in modes)
    contributions = {m.name: m.frequency * m.consequence_cost * factor for m in modes}
    present = math.fsum(contributions.values())
    shares = {
        k: (v / present if present != 0.0 else math.nan)
        for k, v in contributions.items()
    }
    return UnreliabilityCost(
        rate,
        rate * h,
        present,
        math.fsum(m.frequency for m in modes) * h,
        MappingProxyType(contributions),
        MappingProxyType(shares),
    )


def cost_effectiveness_ratio(cost: float, effectiveness: float) -> float:
    """Return ``cost / effectiveness`` (for example cost per unit availability).

    Args:
        cost: Finite cost.
        effectiveness: Positive effectiveness measure.

    Returns:
        Cost per unit of effectiveness (smaller is better).

    Equation:
        [EQ-COST-CER-001].

    References:
        [REF-FABRYCKY-BLANCHARD-1991].

    """
    e = require_finite_real(effectiveness, name="effectiveness")
    if e <= 0.0:
        raise ValueError("effectiveness must be positive")
    return require_finite_real(cost, name="cost") / e


def incremental_cost_effectiveness(
    cost_a: float, effectiveness_a: float, cost_b: float, effectiveness_b: float
) -> float:
    """Return the incremental ratio ``(C_b - C_a) / (E_b - E_a)``.

    Args:
        cost_a: Cost of the reference alternative.
        effectiveness_a: Its effectiveness.
        cost_b: Cost of the challenger.
        effectiveness_b: Its effectiveness (must differ from ``effectiveness_a``).

    Returns:
        Extra cost per extra unit of effectiveness.

    Raises:
        ValueError: If the effectiveness values are equal.

    Equation:
        [EQ-COST-CER-001].

    References:
        [REF-FABRYCKY-BLANCHARD-1991].

    """
    ca = require_finite_real(cost_a, name="cost_a")
    cb = require_finite_real(cost_b, name="cost_b")
    ea = require_finite_real(effectiveness_a, name="effectiveness_a")
    eb = require_finite_real(effectiveness_b, name="effectiveness_b")
    if eb == ea:
        raise ValueError("effectiveness values must differ")
    return (cb - ca) / (eb - ea)


@dataclass(frozen=True, slots=True)
class DesignAlternative:
    """A design alternative for a reliability-cost trade-off.

    Args:
        name: Distinct label.
        investment_cost: Present value of acquisition cost.
        operating_cost: Present value of operation and planned maintenance.
        failure_frequency: Constant failure frequency (per unit time).
        consequence_cost: Cost of each failure (repair, downtime, damage).

    References:
        [REF-FABRYCKY-BLANCHARD-1991], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    name: str
    investment_cost: float
    operating_cost: float = 0.0
    failure_frequency: float = 0.0
    consequence_cost: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fields.

        Raises:
            ValueError: If the name is blank or a value is invalid.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        for field_name in ("investment_cost", "operating_cost", "consequence_cost"):
            object.__setattr__(
                self,
                field_name,
                require_finite_real(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "failure_frequency",
            _non_negative(self.failure_frequency, name="failure_frequency"),
        )


@dataclass(frozen=True, slots=True)
class AlternativeEvaluation:
    """Expected life-cycle and risk cost of one alternative.

    Args:
        name: Alternative label.
        life_cycle_cost: Investment plus operating present value.
        risk_cost: Present value of expected failure cost.
        total_cost: Sum of the two.
        reliability: ``exp(-lambda H)``, probability of no failure in ``[0, H]``.
        cost_per_reliability: ``total_cost / reliability`` (``inf`` if zero).

    References:
        [REF-FABRYCKY-BLANCHARD-1991].

    """

    name: str
    life_cycle_cost: float
    risk_cost: float
    total_cost: float
    reliability: float
    cost_per_reliability: float


def compare_alternatives(
    alternatives: Iterable[DesignAlternative],
    horizon: float,
    *,
    discount_rate: float = 0.0,
) -> tuple[AlternativeEvaluation, ...]:
    """Rank alternatives by expected life-cycle cost plus risk cost.

    Args:
        alternatives: Distinctly named alternatives.
        horizon: Non-negative service horizon ``H``.
        discount_rate: Continuous discount rate for the failure costs (the
            other costs are already present values).

    Returns:
        Evaluations in ascending order of ``total_cost`` (stable on ties).

    Equation:
        [EQ-COST-UNREL-001], [EQ-COST-CER-001].

    References:
        [REF-FABRYCKY-BLANCHARD-1991], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    options = tuple(alternatives)
    if not options:
        raise ValueError("at least one alternative is required")
    if len({a.name for a in options}) != len(options):
        raise ValueError("alternative names must be distinct")
    h = _non_negative(horizon, name="horizon")
    factor = discounted_rate_factor(h, discount_rate)
    rows = []
    for option in options:
        lcc = option.investment_cost + option.operating_cost
        risk = option.failure_frequency * option.consequence_cost * factor
        total = lcc + risk
        reliability = math.exp(-option.failure_frequency * h)
        rows.append(
            AlternativeEvaluation(
                option.name,
                lcc,
                risk,
                total,
                reliability,
                total / reliability if reliability > 0.0 else math.inf,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.total_cost))


def reliability_decision_problem(
    alternatives: Sequence[DesignAlternative],
    scenarios: Sequence[str],
    scenario_probabilities: Sequence[float],
    scenario_frequencies: Mapping[str, Sequence[float]],
    horizon: float,
    *,
    discount_rate: float = 0.0,
) -> DecisionProblem:
    """Build a loss-minimising decision problem across failure-rate scenarios.

    The loss of alternative ``a`` in scenario ``s`` is its life-cycle cost
    plus ``lambda_{a,s} C_a (1 - exp(-r H)) / r``. The returned
    :class:`~probstat.risk.DecisionProblem` gives the Bayes alternative and
    the EVPI: the most it is worth paying to learn which scenario holds
    (for example by a test programme) before choosing.

    Args:
        alternatives: Distinctly named alternatives (their own
            ``failure_frequency`` is ignored).
        scenarios: Scenario labels.
        scenario_probabilities: Prior probabilities summing to one.
        scenario_frequencies: Alternative name to per-scenario frequencies.
        horizon: Service horizon ``H``.
        discount_rate: Continuous discount rate.

    Returns:
        Decision problem with ``maximize=False``.

    Raises:
        KeyError: If an alternative has no frequency row.
        ValueError: If a row has the wrong length.

    Equation:
        [EQ-COST-UNREL-001], [EQ-RISK-DEC-EVPI-001].

    References:
        [REF-RAIFFA-SCHLAIFER-1961, Chapter 4], [REF-FABRYCKY-BLANCHARD-1991].

    """
    options = tuple(alternatives)
    labels = tuple(scenarios)
    factor = discounted_rate_factor(horizon, discount_rate)
    payoffs = []
    for option in options:
        row = tuple(
            _non_negative(v, name="frequency")
            for v in scenario_frequencies[option.name]
        )
        if len(row) != len(labels):
            raise ValueError("each frequency row needs one entry per scenario")
        base = option.investment_cost + option.operating_cost
        payoffs.append(
            tuple(base + rate * option.consequence_cost * factor for rate in row)
        )
    return DecisionProblem(
        tuple(o.name for o in options),
        labels,
        tuple(scenario_probabilities),
        tuple(payoffs),
        maximize=False,
    )
