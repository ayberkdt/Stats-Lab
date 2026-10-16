"""Time value of money: discounting, annuities, inflation, and escalation.

All functions work in one consistent time unit (usually years). Two
compounding conventions are supported everywhere a rate appears:

- ``"discrete"``: one compounding per unit time, ``DF(t) = (1 + r)^-t``;
- ``"continuous"``: ``DF(t) = exp(-r t)``.

A *real* rate discounts constant-currency (base-year) amounts; a *nominal*
rate discounts then-year amounts that include inflation. Mixing the two is
the most common error in life-cycle cost work: discount constant-currency
cash flows with a real rate, or escalate them to then-year amounts first and
use the nominal rate. The Fisher relation converts between the two exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from probstat.foundations import require_finite_real
from probstat.foundations.validation import require_non_negative_integer

COMPOUNDING = ("discrete", "continuous")


def _rate(value: object, *, name: str, compounding: str) -> float:
    """Validate a per-period rate for a compounding convention.

    Args:
        value: Candidate rate (decimal, ``0.07`` for 7 %).
        name: Field name for error messages.
        compounding: ``"discrete"`` or ``"continuous"``.

    Returns:
        The validated rate.

    Raises:
        ValueError: If the convention is unknown or a discrete rate is not
            above ``-1``.

    References:
        Project validation policy. No equation is evaluated.

    """
    if compounding not in COMPOUNDING:
        raise ValueError("compounding must be 'discrete' or 'continuous'")
    rate = require_finite_real(value, name=name)
    if compounding == "discrete" and rate <= -1.0:
        raise ValueError(f"{name} must be greater than -1 for discrete compounding")
    return rate


def discount_factor(
    rate: float, time: float, *, compounding: str = "discrete"
) -> float:
    """Return the factor that converts an amount at ``time`` to time zero.

    Args:
        rate: Discount rate per unit time.
        time: Finite time of the cash flow (negative times compound forward).
        compounding: ``"discrete"`` or ``"continuous"``.

    Returns:
        ``(1 + r)^-t`` or ``exp(-r t)``.

    Raises:
        ValueError: If the rate or convention is invalid.

    Equation:
        [EQ-COST-DISC-001].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    checked = _rate(rate, name="rate", compounding=compounding)
    t = require_finite_real(time, name="time")
    if compounding == "continuous":
        return math.exp(-checked * t)
    return math.exp(-t * math.log1p(checked))


def present_value(
    amount: float, rate: float, time: float, *, compounding: str = "discrete"
) -> float:
    """Return the present value of one amount paid at ``time``.

    Args:
        amount: Finite cash amount.
        rate: Discount rate per unit time.
        time: Finite payment time.
        compounding: ``"discrete"`` or ``"continuous"``.

    Returns:
        ``amount * DF(time)``.

    Equation:
        [EQ-COST-DISC-001].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    value = require_finite_real(amount, name="amount")
    return value * discount_factor(rate, time, compounding=compounding)


def net_present_value(
    cash_flows: Iterable[float],
    rate: float,
    *,
    times: Sequence[float] | None = None,
    compounding: str = "discrete",
) -> float:
    """Return the net present value ``sum_k CF_k DF(t_k)`` of a stream.

    Args:
        cash_flows: Finite signed amounts (costs and benefits with opposite
            signs, or costs only).
        rate: Discount rate per unit time.
        times: Payment times; defaults to ``0, 1, 2, ...`` so the first
            amount is undiscounted (end-of-period convention for the rest).
        compounding: ``"discrete"`` or ``"continuous"``.

    Returns:
        Net present value, summed with ``math.fsum``.

    Raises:
        ValueError: If ``times`` has a different length or the stream is
            empty.

    Equation:
        [EQ-COST-DISC-002].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    flows = tuple(require_finite_real(v, name="cash flow") for v in cash_flows)
    if not flows:
        raise ValueError("cash_flows must not be empty")
    if times is None:
        moments: tuple[float, ...] = tuple(float(k) for k in range(len(flows)))
    else:
        moments = tuple(require_finite_real(t, name="time") for t in times)
        if len(moments) != len(flows):
            raise ValueError("times must have one entry per cash flow")
    return math.fsum(
        flow * discount_factor(rate, t, compounding=compounding)
        for flow, t in zip(flows, moments, strict=True)
    )


def annuity_factor(rate: float, periods: int) -> float:
    """Return the present-worth factor of a uniform end-of-period series.

    Args:
        rate: Discrete rate per period (greater than ``-1``).
        periods: Non-negative number of payments at ``t = 1, ..., n``.

    Returns:
        ``(1 - (1 + r)^-n) / r``, and ``n`` at ``r = 0``.

    Equation:
        [EQ-COST-ANN-001].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    r = _rate(rate, name="rate", compounding="discrete")
    n = require_non_negative_integer(periods, name="periods")
    if r == 0.0:
        return float(n)
    # -expm1(-n log1p(r)) keeps full accuracy for tiny rates.
    return -math.expm1(-n * math.log1p(r)) / r


def capital_recovery_factor(rate: float, periods: int) -> float:
    """Return the factor converting a present amount to an equal annual series.

    Args:
        rate: Discrete rate per period (greater than ``-1``).
        periods: Positive number of periods.

    Returns:
        ``1 / annuity_factor(r, n) = r / (1 - (1 + r)^-n)``.

    Raises:
        ValueError: If ``periods`` is zero.

    Equation:
        [EQ-COST-ANN-001].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    if require_non_negative_integer(periods, name="periods") == 0:
        raise ValueError("periods must be positive")
    return 1.0 / annuity_factor(rate, periods)


def equivalent_annual_cost(present_cost: float, rate: float, periods: int) -> float:
    """Return the equivalent uniform annual cost of a present cost.

    Use it to compare alternatives with unequal lives: the alternative with
    the smaller equivalent annual cost is cheaper under repeated renewal.

    Args:
        present_cost: Finite present value of all costs of one life cycle.
        rate: Discrete rate per period.
        periods: Positive life in periods.

    Returns:
        ``PV * r / (1 - (1 + r)^-n)``.

    Equation:
        [EQ-COST-ANN-001].

    References:
        [REF-BLANK-TARQUIN-2012], [REF-FABRYCKY-BLANCHARD-1991].

    """
    value = require_finite_real(present_cost, name="present_cost")
    return value * capital_recovery_factor(rate, periods)


def real_rate(nominal: float, inflation: float) -> float:
    """Return the real rate from a nominal rate and inflation (Fisher).

    Args:
        nominal: Nominal rate per period (greater than ``-1``).
        inflation: Inflation rate per period (greater than ``-1``).

    Returns:
        ``(1 + nominal) / (1 + inflation) - 1`` (exact, not the additive
        approximation ``nominal - inflation``).

    Equation:
        [EQ-COST-FISHER-001].

    References:
        [REF-FISHER-1930], [REF-BLANK-TARQUIN-2012].

    """
    n = _rate(nominal, name="nominal", compounding="discrete")
    i = _rate(inflation, name="inflation", compounding="discrete")
    return (n - i) / (1.0 + i)


def nominal_rate(real: float, inflation: float) -> float:
    """Return the nominal rate from a real rate and inflation (Fisher).

    Args:
        real: Real rate per period (greater than ``-1``).
        inflation: Inflation rate per period (greater than ``-1``).

    Returns:
        ``(1 + real)(1 + inflation) - 1``.

    Equation:
        [EQ-COST-FISHER-001].

    References:
        [REF-FISHER-1930], [REF-BLANK-TARQUIN-2012].

    """
    r = _rate(real, name="real", compounding="discrete")
    i = _rate(inflation, name="inflation", compounding="discrete")
    return r + i + r * i


def continuous_rate(discrete: float) -> float:
    """Return the continuous rate equivalent to a discrete rate.

    Args:
        discrete: Discrete rate per period (greater than ``-1``).

    Returns:
        ``log(1 + r)``, so both conventions give the same discount factors.

    Equation:
        [EQ-COST-DISC-001].

    References:
        [REF-BLANK-TARQUIN-2012].

    """
    return math.log1p(_rate(discrete, name="discrete", compounding="discrete"))


def escalate(amount: float, escalation_rate: float, time: float) -> float:
    """Return a base-year amount escalated to a later time.

    Args:
        amount: Finite base-year (constant-currency) amount.
        escalation_rate: Escalation rate per unit time (greater than ``-1``).
        time: Finite time since the base year.

    Returns:
        ``amount (1 + e)^t``, the then-year amount.

    Equation:
        [EQ-COST-ESC-001].

    References:
        [REF-BLANK-TARQUIN-2012], [REF-NASA-CEH-2015].

    """
    value = require_finite_real(amount, name="amount")
    e = _rate(escalation_rate, name="escalation_rate", compounding="discrete")
    t = require_finite_real(time, name="time")
    return value * math.exp(t * math.log1p(e))
