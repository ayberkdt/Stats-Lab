"""Spare-part provisioning under Poisson demand.

With ``n`` identical installed units, each failing at a constant rate
``lambda``, and a resupply (or mission) period ``T``, the demand for spares in
the period is ``D ~ Poisson(mu)`` with ``mu = lambda n T`` (failed parts are
not returned to stock within the period). Holding ``s`` spares:

- protection level (probability of no stock-out): ``P(D <= s)``;
- expected backorders: ``EBO(s) = E[(D - s)^+] = mu P(D >= s) - s P(D >= s+1)``,
  an exact identity that avoids an infinite sum;
- a simple plan cost: ``s c_unit + c_short EBO(s)``, convex in ``s``.

Assumptions: constant failure rates (no wear-out), independent failures, no
repair of failed parts within the period, single echelon, single item. For
multi-item, multi-echelon systems use METRIC-type models (Sherbrooke 2004).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.distributions import Poisson
from probstat.foundations import require_finite_real, require_probability
from probstat.foundations.validation import require_non_negative_integer


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


def spare_demand_mean(failure_rate: float, units: int, period: float) -> float:
    """Return the Poisson demand mean ``mu = lambda n T``.

    Args:
        failure_rate: Non-negative failure rate per unit per unit time.
        units: Positive number of installed units.
        period: Non-negative resupply or mission period.

    Returns:
        Expected demand in the period.

    Equation:
        [EQ-COST-SPARE-001].

    References:
        [REF-SHERBROOKE-2004], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    rate = _non_negative(failure_rate, name="failure_rate")
    n = require_non_negative_integer(units, name="units")
    if n == 0:
        raise ValueError("units must be positive")
    return rate * n * _non_negative(period, name="period")


def _survival_at(mean: float, k: int) -> float:
    """Return ``P(D > k)`` for ``D ~ Poisson(mean)`` (``1`` for ``k < 0``).

    Args:
        mean: Non-negative Poisson mean.
        k: Integer level.

    Returns:
        Upper-tail probability.

    Equation:
        [EQ-DIST-POIS-001].

    References:
        [REF-ROSS-2019, Section 4.8].

    """
    if k < 0:
        return 1.0
    if mean == 0.0:
        return 0.0
    return float(Poisson(mean).sf(k))


def protection_level(mean: float, spares: int) -> float:
    """Return ``P(D <= s)``, the probability that ``s`` spares suffice.

    Args:
        mean: Non-negative Poisson demand mean.
        spares: Non-negative stock ``s``.

    Returns:
        Probability of no stock-out in the period.

    Equation:
        [EQ-COST-SPARE-001].

    References:
        [REF-SHERBROOKE-2004].

    """
    mu = _non_negative(mean, name="mean")
    s = require_non_negative_integer(spares, name="spares")
    if mu == 0.0:
        return 1.0
    return float(Poisson(mu).cdf(s))


def expected_backorders(mean: float, spares: int) -> float:
    """Return ``EBO(s) = E[(D - s)^+] = mu P(D >= s) - s P(D >= s + 1)``.

    Args:
        mean: Non-negative Poisson demand mean.
        spares: Non-negative stock ``s``.

    Returns:
        Expected number of unfilled demands in the period.

    Equation:
        [EQ-COST-SPARE-002].

    References:
        [REF-SHERBROOKE-2004].

    """
    mu = _non_negative(mean, name="mean")
    s = require_non_negative_integer(spares, name="spares")
    value = mu * _survival_at(mu, s - 1) - s * _survival_at(mu, s)
    return max(value, 0.0)


def minimum_spares(mean: float, target: float) -> int:
    """Return the smallest ``s`` with ``P(D <= s) >= target``.

    Args:
        mean: Non-negative Poisson demand mean.
        target: Protection level in ``[0, 1)``.

    Returns:
        Minimum stock; found by bisection on the monotone CDF.

    Raises:
        ValueError: If ``target`` is one (no finite stock gives certainty for
            a positive mean).

    Equation:
        [EQ-COST-SPARE-001].

    References:
        [REF-SHERBROOKE-2004].

    """
    mu = _non_negative(mean, name="mean")
    level = require_probability(target, name="target")
    if level == 1.0 and mu > 0.0:
        raise ValueError("target must be below one for a positive mean")
    if mu == 0.0 or protection_level(mu, 0) >= level:
        return 0
    low, high = 0, max(1, int(mu + 10.0 * math.sqrt(mu) + 20.0))
    while protection_level(mu, high) < level:
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if protection_level(mu, middle) >= level:
            high = middle
        else:
            low = middle
    return high


@dataclass(frozen=True, slots=True)
class SparePlan:
    """Evaluation of one spare stock level.

    Args:
        spares: Stock ``s``.
        demand_mean: Poisson mean ``mu``.
        protection: ``P(D <= s)``.
        expected_backorders: ``E[(D - s)^+]``.
        purchase_cost: ``s c_unit``.
        shortage_cost: ``c_short EBO(s)``.
        total_cost: Sum of the two.

    References:
        [REF-SHERBROOKE-2004].

    """

    spares: int
    demand_mean: float
    protection: float
    expected_backorders: float
    purchase_cost: float
    shortage_cost: float
    total_cost: float


def spare_plan_cost(
    mean: float, spares: int, unit_cost: float, shortage_cost: float
) -> SparePlan:
    """Return the cost of holding ``s`` spares for one period.

    Args:
        mean: Poisson demand mean.
        spares: Stock ``s``.
        unit_cost: Cost of one spare (purchase plus holding for the period).
        shortage_cost: Cost of one backorder (unfilled demand).

    Returns:
        Plan evaluation with ``total = s c_unit + c_short EBO(s)``.

    Equation:
        [EQ-COST-SPARE-002], [EQ-COST-SPARE-003].

    References:
        [REF-SHERBROOKE-2004].

    """
    mu = _non_negative(mean, name="mean")
    s = require_non_negative_integer(spares, name="spares")
    unit = _non_negative(unit_cost, name="unit_cost")
    short = _non_negative(shortage_cost, name="shortage_cost")
    ebo = expected_backorders(mu, s)
    return SparePlan(
        s,
        mu,
        protection_level(mu, s),
        ebo,
        s * unit,
        short * ebo,
        s * unit + short * ebo,
    )


def optimal_spares(mean: float, unit_cost: float, shortage_cost: float) -> SparePlan:
    """Return the stock minimising ``s c_unit + c_short EBO(s)``.

    ``EBO(s) - EBO(s + 1) = P(D > s)`` is non-increasing, so the cost is
    convex and the optimum is the smallest ``s`` with
    ``c_short P(D > s) <= c_unit`` (a newsvendor critical fractile).

    Args:
        mean: Poisson demand mean.
        unit_cost: Positive cost per spare.
        shortage_cost: Non-negative cost per backorder.

    Returns:
        Optimal plan.

    Equation:
        [EQ-COST-SPARE-003].

    References:
        [REF-SHERBROOKE-2004].

    """
    mu = _non_negative(mean, name="mean")
    unit = _non_negative(unit_cost, name="unit_cost")
    short = _non_negative(shortage_cost, name="shortage_cost")
    if unit <= 0.0:
        raise ValueError("unit_cost must be positive")
    if short <= unit or mu == 0.0:
        return spare_plan_cost(mu, 0, unit, short)
    # Smallest s with P(D <= s) >= 1 - unit / short.
    return spare_plan_cost(mu, minimum_spares(mu, 1.0 - unit / short), unit, short)
