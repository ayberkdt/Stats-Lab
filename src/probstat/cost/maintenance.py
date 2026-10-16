"""Preventive-maintenance cost optimisation (Barlow-Proschan policies).

All policies minimise a *long-run expected cost per unit time* derived from
the renewal-reward theorem: expected cost per cycle divided by expected cycle
length.

- **Age replacement**: replace preventively at age ``T`` (cost ``c_p``) or at
  failure (cost ``c_f > c_p``), whichever comes first:
  ``C(T) = (c_p R(T) + c_f F(T)) / int_0^T R(t) dt``. As ``T -> inf`` this
  tends to the run-to-failure rate ``c_f / MTTF``. With a non-increasing
  hazard (for example the exponential law) preventive replacement never
  pays and no finite optimum exists; the result says so.
- **Block replacement**: replace at ``T, 2T, ...`` regardless of age and at
  every failure: ``C(T) = (c_p + c_f M(T)) / T`` with the renewal function
  ``M(t)`` obtained by solving the renewal equation numerically.
- **Minimal repair with periodic overhaul**: failures are minimally repaired
  (cost ``c_m``; the system is "as bad as old") and the system is overhauled
  to as-good-as-new every ``T`` (cost ``c_p``):
  ``C(T) = (c_p + c_m Lambda(T)) / T``. For the power-law process
  ``Lambda(t) = (t / eta)^beta`` with ``beta > 1`` the optimum is
  ``T* = eta (c_p / ((beta - 1) c_m))^(1/beta)``.

Assumptions shared by all policies: replacement and repair times are
negligible, costs are constant, and the lifetime law is known (feed
:mod:`probstat.reliability.data` posteriors through a
:class:`~probstat.modeling.ProbabilisticModel` to propagate parameter
uncertainty).
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from probstat.foundations import require_finite_real
from probstat.numerics import integrate

from ._scalar import brent_minimize


class LifetimeLaw(Protocol):
    """Structural interface of a lifetime law used by the policies.

    References:
        [REF-BARLOW-PROSCHAN-1965].

    """

    def cdf(self, x: float) -> float:
        """Return ``F(x) = P(lifetime <= x)``.

        References:
            [REF-BARLOW-PROSCHAN-1965].

        """
        ...

    def quantile(self, probability: float) -> float:
        """Return the inverse CDF.

        References:
            [REF-BARLOW-PROSCHAN-1965].

        """
        ...

    @property
    def mean(self) -> float:
        """Return the mean time to failure.

        References:
            [REF-BARLOW-PROSCHAN-1965].

        """
        ...


def _positive(value: object, *, name: str) -> float:
    """Validate a finite positive real.

    Args:
        value: Candidate value.
        name: Field name for error messages.

    Returns:
        The value as ``float``.

    Raises:
        ValueError: If it is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _survival(law: LifetimeLaw) -> Callable[[float], float]:
    """Return ``R(t)``, using ``law.sf`` when available for tail accuracy.

    Args:
        law: Lifetime law.

    Returns:
        Survival function.

    References:
        [REF-BARLOW-PROSCHAN-1965]. No new equation.

    """
    method = getattr(law, "sf", None)
    if method is not None:
        return lambda t: float(method(t))
    return lambda t: 1.0 - float(law.cdf(t))


def _costs(preventive: float, failure: float, *, name: str) -> tuple[float, float]:
    """Validate ``0 < c_p < c_f``.

    Args:
        preventive: Preventive-action cost.
        failure: Failure (corrective) cost.
        name: Policy name for error messages.

    Returns:
        The two costs.

    Raises:
        ValueError: If the ordering does not hold.

    References:
        [REF-BARLOW-PROSCHAN-1965]. No equation is evaluated.

    """
    cp = _positive(preventive, name="preventive_cost")
    cf = _positive(failure, name="failure_cost")
    if cf <= cp:
        raise ValueError(f"{name} needs failure_cost > preventive_cost")
    return cp, cf


@dataclass(frozen=True, slots=True)
class ReplacementPolicyResult:
    """Optimum of a preventive-replacement cost rate.

    Args:
        policy: ``"age"``, ``"block"`` or ``"minimal_repair"``.
        finite_optimum: Whether a finite interval beats running to failure.
        interval: Optimal interval ``T*`` (``inf`` when no finite optimum).
        cost_rate: Long-run cost per unit time at ``T*``.
        run_to_failure_cost_rate: Rate without preventive action (``c_f /
            MTTF``); ``nan`` when undefined (minimal repair).
        savings: ``1 - cost_rate / run_to_failure_cost_rate`` (``nan`` if
            undefined).
        evaluations: Cost-rate evaluations used.
        resolution: Grid spacing limiting the accuracy of ``interval`` for
            grid-based policies (``0`` when refined by Brent's method).
        message: Plain-language explanation.

    References:
        [REF-BARLOW-PROSCHAN-1965], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    policy: str
    finite_optimum: bool
    interval: float
    cost_rate: float
    run_to_failure_cost_rate: float
    savings: float
    evaluations: int
    resolution: float
    message: str


def age_replacement_cost_rate(
    law: LifetimeLaw, interval: float, preventive_cost: float, failure_cost: float
) -> float:
    """Return ``C(T) = (c_p R(T) + c_f F(T)) / int_0^T R(t) dt``.

    Args:
        law: Lifetime law with ``cdf`` (and optionally ``sf``).
        interval: Positive replacement age ``T``.
        preventive_cost: ``c_p``.
        failure_cost: ``c_f`` (greater than ``c_p``).

    Returns:
        Long-run expected cost per unit time.

    Equation:
        [EQ-COST-AGE-001].

    References:
        [REF-BARLOW-PROSCHAN-1965], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    cp, cf = _costs(preventive_cost, failure_cost, name="age replacement")
    t = _positive(interval, name="interval")
    survival = _survival(law)
    cycle = integrate(survival, 0.0, t, rel_tol=1e-11).value
    r = survival(t)
    return (cp * r + cf * (1.0 - r)) / cycle


def optimize_age_replacement(
    law: LifetimeLaw,
    preventive_cost: float,
    failure_cost: float,
    *,
    upper: float | None = None,
    grid_points: int = 48,
) -> ReplacementPolicyResult:
    """Return the cost-optimal age-replacement interval.

    A log-spaced grid on ``[upper 1e-4, upper]`` locates the best region and
    Brent's method refines it. If the best grid point is the last one, or the
    refined rate is not below ``c_f / MTTF``, no finite optimum exists within
    the range and run-to-failure is reported (always the case for a
    non-increasing hazard rate).

    Args:
        law: Lifetime law with ``cdf``, ``quantile`` and ``mean``.
        preventive_cost: ``c_p``.
        failure_cost: ``c_f > c_p``.
        upper: Largest interval considered; default ``F^-1(1 - 1e-6)``.
        grid_points: Number of coarse grid points (at least 8).

    Returns:
        Optimal policy, its cost rate, and the run-to-failure comparison.

    Equation:
        [EQ-COST-AGE-001], [EQ-COST-BRENTMIN-001].

    References:
        [REF-BARLOW-PROSCHAN-1965], [REF-RAUSAND-BARROS-HOYLAND-2021],
        [REF-BRENT-1973, Chapter 5].

    """
    cp, cf = _costs(preventive_cost, failure_cost, name="age replacement")
    if (
        isinstance(grid_points, bool)
        or not isinstance(grid_points, int)
        or grid_points < 8
    ):
        raise ValueError("grid_points must be an integer of at least 8")
    mttf = _positive(law.mean, name="mean lifetime")
    limit = (
        float(law.quantile(1.0 - 1e-6))
        if upper is None
        else _positive(upper, name="upper")
    )
    run_to_failure = cf / mttf
    evaluations = 0

    def rate(t: float) -> float:
        """Return the age-replacement cost rate at ``t``.

        References:
            [EQ-COST-AGE-001], [REF-BARLOW-PROSCHAN-1965].

        """
        nonlocal evaluations
        evaluations += 1
        return age_replacement_cost_rate(law, t, cp, cf)

    low = limit * 1e-4
    grid = [low * (limit / low) ** (k / (grid_points - 1)) for k in range(grid_points)]
    values = [rate(t) for t in grid]
    best = min(range(grid_points), key=values.__getitem__)
    if best == grid_points - 1:
        return ReplacementPolicyResult(
            "age",
            False,
            math.inf,
            run_to_failure,
            run_to_failure,
            0.0,
            evaluations,
            0.0,
            "Cost rate decreases over the whole range: no finite optimum; "
            "run to failure (expected for a non-increasing hazard rate).",
        )
    left = grid[max(best - 1, 0)]
    right = grid[best + 1]
    found = brent_minimize(rate, left, right, tolerance=1e-12 * limit)
    if found.value >= run_to_failure * (1.0 - 1e-12):
        return ReplacementPolicyResult(
            "age",
            False,
            math.inf,
            run_to_failure,
            run_to_failure,
            0.0,
            evaluations,
            0.0,
            "Preventive replacement does not beat run-to-failure.",
        )
    return ReplacementPolicyResult(
        "age",
        True,
        found.x,
        found.value,
        run_to_failure,
        1.0 - found.value / run_to_failure,
        evaluations,
        0.0,
        "Finite optimum found by grid search and Brent refinement."
        + ("" if found.converged else " Brent iteration limit reached."),
    )


@dataclass(frozen=True, slots=True)
class RenewalFunction:
    """Numerical renewal function ``M(t) = E[N(t)]`` on a uniform grid.

    Args:
        times: Grid ``0, h, 2h, ..., horizon``.
        values: ``M`` at the grid (``M(0) = 0``).
        error_estimate: ``max |M_h - M_{2h}|`` over the coarse grid, an
            estimate of the discretization error of the coarser solution
            (the returned finer solution is usually more accurate).
        step: Grid spacing ``h``.

    References:
        [REF-XIE-1989], [REF-ROSS-1996, Chapter 3].

    """

    times: tuple[float, ...]
    values: tuple[float, ...]
    error_estimate: float
    step: float

    def __call__(self, t: float) -> float:
        """Return ``M(t)`` by linear interpolation on the grid.

        Args:
            t: Time in ``[0, horizon]``.

        Returns:
            Interpolated renewal function.

        Raises:
            ValueError: If ``t`` is outside the grid.

        References:
            [REF-XIE-1989]. No new equation.

        """
        x = require_finite_real(t, name="t")
        if not 0.0 <= x <= self.times[-1] * (1.0 + 1e-12):
            raise ValueError("t lies outside the computed grid")
        position = min(x / self.step, len(self.times) - 1.0)
        index = min(int(position), len(self.times) - 2)
        weight = position - index
        return (1.0 - weight) * self.values[index] + weight * self.values[index + 1]


def _renewal_rs(cdf_values: list[float]) -> list[float]:
    """Solve the discretized renewal equation by Xie's RS scheme.

    Args:
        cdf_values: ``F(t_k)`` on a uniform grid with ``t_0 = 0``.

    Returns:
        ``M(t_k)`` from the trapezoidal Riemann-Stieltjes recursion.

    Equation:
        [EQ-COST-RENEW-001].

    References:
        [REF-XIE-1989].

    """
    size = len(cdf_values)
    increments = [0.0] + [cdf_values[k] - cdf_values[k - 1] for k in range(1, size)]
    m = [0.0] * size
    averages = [0.0] * size
    first = increments[1] if size > 1 else 0.0
    for n in range(1, size):
        # sum_{j=1}^{n-1} S_j dF_{n-j+1}, S_j = (M_j + M_{j-1}) / 2.
        tail = sum(map(operator.mul, averages[1:n], increments[n:1:-1]))
        m[n] = (cdf_values[n] + 0.5 * first * m[n - 1] + tail) / (1.0 - 0.5 * first)
        averages[n] = 0.5 * (m[n] + m[n - 1])
    return m


def renewal_function(
    law: LifetimeLaw, horizon: float, *, steps: int = 400
) -> RenewalFunction:
    """Return the renewal function ``M(t) = F(t) + int_0^t M(t - x) dF(x)``.

    The renewal equation is solved on ``2 * steps`` intervals with Xie's
    Riemann-Stieltjes (trapezoidal) scheme, which is second order for smooth
    ``F``. It is also solved on ``steps`` intervals and the largest
    difference is reported as an error estimate. Cost is ``O(steps^2)``.

    Args:
        law: Lifetime law with ``cdf``.
        horizon: Positive end of the grid.
        steps: Coarse number of intervals (at least 10).

    Returns:
        Renewal function on the fine grid with an error estimate.

    Equation:
        [EQ-COST-RENEW-001].

    References:
        [REF-XIE-1989], [REF-ROSS-1996, Chapter 3].

    """
    end = _positive(horizon, name="horizon")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 10:
        raise ValueError("steps must be an integer of at least 10")
    fine = 2 * steps
    h = end / fine
    times = [k * h for k in range(fine + 1)]
    cdf = [float(law.cdf(t)) for t in times]
    fine_values = _renewal_rs(cdf)
    coarse_values = _renewal_rs(cdf[::2])
    error = max(abs(fine_values[2 * k] - coarse_values[k]) for k in range(steps + 1))
    return RenewalFunction(tuple(times), tuple(fine_values), error, h)


def optimize_block_replacement(
    law: LifetimeLaw,
    preventive_cost: float,
    failure_cost: float,
    *,
    horizon: float | None = None,
    steps: int = 400,
) -> ReplacementPolicyResult:
    """Return the cost-optimal block-replacement interval.

    ``C(T) = (c_p + c_f M(T)) / T`` is evaluated on the renewal-function
    grid; the best grid point is refined by the vertex of the parabola
    through it and its neighbours. ``resolution`` is the grid step. As
    ``T -> inf``, ``C(T) -> c_f / MTTF``, the run-to-failure rate.

    Args:
        law: Lifetime law with ``cdf`` and ``mean``.
        preventive_cost: ``c_p``.
        failure_cost: ``c_f > c_p``.
        horizon: Largest interval considered; default ``3 * MTTF``.
        steps: Coarse renewal-grid intervals (the fine grid has twice as many).

    Returns:
        Optimal policy, or run-to-failure when the rate keeps decreasing.

    Equation:
        [EQ-COST-BLOCK-001], [EQ-COST-RENEW-001].

    References:
        [REF-BARLOW-PROSCHAN-1965], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    cp, cf = _costs(preventive_cost, failure_cost, name="block replacement")
    mttf = _positive(law.mean, name="mean lifetime")
    end = 3.0 * mttf if horizon is None else _positive(horizon, name="horizon")
    renewal = renewal_function(law, end, steps=steps)
    rates = [math.inf] + [
        (cp + cf * m) / t
        for t, m in zip(renewal.times[1:], renewal.values[1:], strict=True)
    ]
    run_to_failure = cf / mttf
    best = min(range(1, len(rates)), key=rates.__getitem__)
    evaluations = len(rates) - 1
    if best == len(rates) - 1 or rates[best] >= run_to_failure * (1.0 - 1e-12):
        return ReplacementPolicyResult(
            "block",
            False,
            math.inf,
            run_to_failure,
            run_to_failure,
            0.0,
            evaluations,
            renewal.step,
            "Cost rate does not fall below run-to-failure within the horizon: "
            "no finite optimum found.",
        )
    interval, value = renewal.times[best], rates[best]
    if best >= 2:
        y0, y1, y2 = rates[best - 1], rates[best], rates[best + 1]
        curvature = y0 - 2.0 * y1 + y2
        if curvature > 0.0:
            shift = 0.5 * (y0 - y2) / curvature
            interval = renewal.times[best] + shift * renewal.step
            value = y1 - 0.25 * (y0 - y2) * shift
    return ReplacementPolicyResult(
        "block",
        True,
        interval,
        value,
        run_to_failure,
        1.0 - value / run_to_failure,
        evaluations,
        renewal.step,
        "Finite optimum on the renewal grid refined by parabolic interpolation; "
        f"renewal-function error estimate {renewal.error_estimate:.2e}.",
    )


def minimal_repair_overhaul(
    beta: float, eta: float, overhaul_cost: float, repair_cost: float
) -> ReplacementPolicyResult:
    """Return the optimal overhaul period under minimal repair (power law).

    Failures follow a power-law NHPP with ``Lambda(t) = (t / eta)^beta`` and
    are minimally repaired at ``c_m`` each; an overhaul at cost ``c_p``
    renews the system every ``T``. For ``beta > 1``,
    ``T* = eta (c_p / ((beta - 1) c_m))^(1/beta)`` and
    ``C(T*) = beta c_p / ((beta - 1) T*)``. For ``beta <= 1`` the rate
    decreases in ``T``: overhauls never pay.

    Args:
        beta: Positive power-law shape.
        eta: Positive scale.
        overhaul_cost: ``c_p``.
        repair_cost: ``c_m`` per minimal repair.

    Returns:
        Closed-form optimum (``run_to_failure_cost_rate`` is ``nan``: without
        overhaul the long-run rate of a deteriorating system is unbounded).

    Equation:
        [EQ-COST-MINREP-001].

    References:
        [REF-BARLOW-PROSCHAN-1965], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    b = _positive(beta, name="beta")
    scale = _positive(eta, name="eta")
    cp = _positive(overhaul_cost, name="overhaul_cost")
    cm = _positive(repair_cost, name="repair_cost")
    if b <= 1.0:
        limit = cm / scale if b == 1.0 else 0.0
        return ReplacementPolicyResult(
            "minimal_repair",
            False,
            math.inf,
            limit,
            math.nan,
            math.nan,
            0,
            0.0,
            "beta <= 1: failure intensity is not increasing; no finite "
            "overhaul period is optimal.",
        )
    interval = scale * (cp / ((b - 1.0) * cm)) ** (1.0 / b)
    rate = b * cp / ((b - 1.0) * interval)
    return ReplacementPolicyResult(
        "minimal_repair",
        True,
        interval,
        rate,
        math.nan,
        math.nan,
        0,
        0.0,
        "Closed-form optimum for the power-law process.",
    )


def periodic_overhaul_cost_rate(
    cumulative_intensity: Callable[[float], float],
    interval: float,
    overhaul_cost: float,
    repair_cost: float,
) -> float:
    """Return ``(c_p + c_m Lambda(T)) / T`` for any cumulative intensity.

    Args:
        cumulative_intensity: ``Lambda(t)``, expected failures in ``[0, t]``.
        interval: Positive overhaul period ``T``.
        overhaul_cost: ``c_p``.
        repair_cost: ``c_m``.

    Returns:
        Long-run cost per unit time.

    Equation:
        [EQ-COST-MINREP-001].

    References:
        [REF-BARLOW-PROSCHAN-1965].

    """
    t = _positive(interval, name="interval")
    cp = _positive(overhaul_cost, name="overhaul_cost")
    cm = _positive(repair_cost, name="repair_cost")
    expected = require_finite_real(cumulative_intensity(t), name="Lambda(T)")
    return (cp + cm * expected) / t


def optimize_periodic_overhaul(
    cumulative_intensity: Callable[[float], float],
    overhaul_cost: float,
    repair_cost: float,
    *,
    upper: float,
    grid_points: int = 48,
) -> ReplacementPolicyResult:
    """Return the optimal overhaul period for a general cumulative intensity.

    Args:
        cumulative_intensity: ``Lambda(t)``.
        overhaul_cost: ``c_p``.
        repair_cost: ``c_m``.
        upper: Largest period considered.
        grid_points: Coarse log-grid size (at least 8).

    Returns:
        Grid-plus-Brent optimum; ``finite_optimum`` is false when the rate is
        smallest at ``upper``.

    Equation:
        [EQ-COST-MINREP-001], [EQ-COST-BRENTMIN-001].

    References:
        [REF-BARLOW-PROSCHAN-1965], [REF-BRENT-1973, Chapter 5].

    """
    limit = _positive(upper, name="upper")
    if (
        isinstance(grid_points, bool)
        or not isinstance(grid_points, int)
        or grid_points < 8
    ):
        raise ValueError("grid_points must be an integer of at least 8")
    evaluations = 0

    def rate(t: float) -> float:
        """Return the overhaul cost rate at ``t``.

        References:
            [EQ-COST-MINREP-001], [REF-BARLOW-PROSCHAN-1965].

        """
        nonlocal evaluations
        evaluations += 1
        return periodic_overhaul_cost_rate(
            cumulative_intensity, t, overhaul_cost, repair_cost
        )

    low = limit * 1e-4
    grid = [low * (limit / low) ** (k / (grid_points - 1)) for k in range(grid_points)]
    values = [rate(t) for t in grid]
    best = min(range(grid_points), key=values.__getitem__)
    if best == grid_points - 1:
        return ReplacementPolicyResult(
            "minimal_repair",
            False,
            math.inf,
            values[-1],
            math.nan,
            math.nan,
            evaluations,
            0.0,
            "Cost rate decreases up to the upper limit.",
        )
    found = brent_minimize(
        rate, grid[max(best - 1, 0)], grid[best + 1], tolerance=1e-12 * limit
    )
    return ReplacementPolicyResult(
        "minimal_repair",
        True,
        found.x,
        found.value,
        math.nan,
        math.nan,
        evaluations,
        0.0,
        "Finite optimum found by grid search and Brent refinement.",
    )


@dataclass(frozen=True, slots=True)
class CorrectiveCost:
    """Expected corrective-maintenance cost over a horizon.

    Args:
        expected_failures: ``int_0^H w(t) dt``.
        expected_cost: ``c * expected_failures`` (undiscounted).
        present_value: ``c int_0^H w(t) exp(-r t) dt`` (continuous discounting).
        error_estimate: Quadrature error estimate of the present value.

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    expected_failures: float
    expected_cost: float
    present_value: float
    error_estimate: float


def expected_corrective_cost(
    intensity: Callable[[float], float],
    horizon: float,
    cost_per_failure: float,
    *,
    discount_rate: float = 0.0,
) -> CorrectiveCost:
    """Return the expected number and cost of failures from an intensity.

    For a repairable system with rate of occurrence of failures ``w(t)``,
    ``E[N(H)] = int_0^H w(t) dt`` regardless of the repair model, provided
    ``w`` is the unconditional intensity under that model.

    Args:
        intensity: Non-negative failure intensity ``w(t)``.
        horizon: Positive horizon ``H``.
        cost_per_failure: Cost ``c`` of each corrective action.
        discount_rate: Continuous discount rate ``r`` (``0`` for none).

    Returns:
        Expected failures, cost, and present value.

    Equation:
        [EQ-COST-CORR-001].

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021], [REF-BLANK-TARQUIN-2012].

    """
    end = _positive(horizon, name="horizon")
    cost = require_finite_real(cost_per_failure, name="cost_per_failure")
    rate = require_finite_real(discount_rate, name="discount_rate")
    failures = integrate(intensity, 0.0, end, rel_tol=1e-11)
    if rate == 0.0:
        discounted = failures
    else:
        discounted = integrate(
            lambda t: intensity(t) * math.exp(-rate * t), 0.0, end, rel_tol=1e-11
        )
    return CorrectiveCost(
        failures.value,
        cost * failures.value,
        cost * discounted.value,
        abs(cost) * discounted.error_estimate,
    )
