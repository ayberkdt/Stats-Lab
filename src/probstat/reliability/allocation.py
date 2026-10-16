"""Reliability allocation (apportionment) to subsystems.

Weighting methods for *series* systems of constant-failure-rate units with a
system reliability target ``R*`` over a mission time ``t`` (so the system
failure rate budget is ``lambda* = -ln R* / t``):

- **Equal apportionment**: ``R_i = R*^(1/n)``.
- **ARINC**: weights proportional to current (predicted) failure rates,
  ``lambda_i* = (lambda_i / sum lambda_j) lambda*``.
- **AGREE**: unit ``i`` with ``n_i`` of ``N`` modules, importance factor
  ``E_i`` (probability that a unit failure causes system failure) and
  operating time ``t_i`` gets
  ``R_i(t_i) = 1 - (1 - R*^(n_i/N)) / E_i`` and
  ``lambda_i = -ln R_i(t_i) / t_i`` (exact form), or the classical
  approximation ``lambda_i = n_i (-ln R*) / (N E_i t_i)``.
- **Feasibility of objectives**: each unit is rated 1-10 on several
  factors (MIL-HDBK-338B uses intricacy, state of the art, performance time,
  and environment); weights are the products of the ratings, normalised.

**Cost-optimal allocation** minimises ``sum_i c_i(R_i)`` subject to
``R_sys(R_1, ..., R_n) >= R*`` and ``l_i <= R_i < u_i``:

- for series systems, exactly through the Lagrange (KKT) conditions
  ``R_i c_i'(R_i) = mu`` (one multiplier, found by Brent's method) when every
  ``c_i`` is convex and increasing;
- for any structure given as a callable ``R_sys``, by an augmented
  Lagrangian method with BFGS on a bounded reparametrisation, followed by a
  feasibility repair along the segment towards the upper bounds. The
  repair assumes ``R_sys`` is non-decreasing in each component (a coherent
  system).

The convex cost-of-reliability model of Mettas (2000),
``c(R) = exp((1 - f)(R - R_min) / (R_max - R))`` with feasibility
``f in [0, 1)``, is provided as :class:`MettasCost`.

Stability: this module is *experimental* (ADR-0063) inside a *provisional*
package; its names and signatures may change in any release.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations import require_finite_real, require_probability
from probstat.optimization import brent_root, minimize_bfgs

CostFunction = Callable[[float], float]


def _target(value: object) -> float:
    """Validate a reliability target strictly inside ``(0, 1)``.

    Args:
        value: Candidate target.

    Returns:
        The target.

    Raises:
        ValueError: If it is 0 or 1.

    References:
        Project validation policy. No equation is evaluated.

    """
    target = require_probability(value, name="target")
    if target in (0.0, 1.0):
        raise ValueError("target must lie strictly between zero and one")
    return target


def _names(names: Sequence[str] | None, size: int) -> tuple[str, ...]:
    """Return unit names, defaulting to ``unit_1, ..., unit_n``.

    Args:
        names: Optional distinct names.
        size: Number of units.

    Returns:
        Names tuple.

    Raises:
        ValueError: If the count is wrong or names repeat.

    References:
        Project validation policy. No equation is evaluated.

    """
    if names is None:
        return tuple(f"unit_{k + 1}" for k in range(size))
    labels = tuple(str(n) for n in names)
    if len(labels) != size or len(set(labels)) != size:
        raise ValueError("names must be distinct with one per unit")
    return labels


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Allocated unit reliabilities and how well they meet the target.

    Args:
        method: Allocation method label.
        names: Unit names.
        reliabilities: Allocated unit reliabilities (over each unit's
            operating time for AGREE, the mission time otherwise).
        failure_rates: Allocated constant failure rates, or ``None`` when no
            mission time applies.
        target: System reliability target ``R*``.
        system_reliability: System reliability achieved by the allocation.
        constraint_satisfied: ``system_reliability >= R* - tolerance``.
        cost: Total cost (``nan`` for weighting methods).
        message: Plain-language summary.

    References:
        [REF-EBELING-1997], [REF-MIL-HDBK-338B-1998].

    """

    method: str
    names: tuple[str, ...]
    reliabilities: tuple[float, ...]
    failure_rates: tuple[float, ...] | None
    target: float
    system_reliability: float
    constraint_satisfied: bool
    cost: float
    message: str


def _from_rates(
    method: str,
    names: tuple[str, ...],
    rates: Sequence[float],
    mission_time: float,
    target: float,
    message: str,
) -> AllocationResult:
    """Build a result from allocated failure rates of a series system.

    Args:
        method: Method label.
        names: Unit names.
        rates: Allocated failure rates.
        mission_time: Mission time ``t``.
        target: Target ``R*``.
        message: Summary text.

    Returns:
        Allocation result with ``R_i = exp(-lambda_i t)``.

    Equation:
        [EQ-REL-ALLOC-001].

    References:
        [REF-EBELING-1997].

    """
    reliabilities = tuple(math.exp(-r * mission_time) for r in rates)
    system = math.exp(-math.fsum(rates) * mission_time)
    return AllocationResult(
        method,
        names,
        reliabilities,
        tuple(rates),
        target,
        system,
        system >= target * (1.0 - 1e-12),
        math.nan,
        message,
    )


def equal_apportionment(
    target: float,
    units: int,
    *,
    mission_time: float | None = None,
    names: Sequence[str] | None = None,
) -> AllocationResult:
    """Return ``R_i = R*^(1/n)`` for ``n`` units in series.

    Args:
        target: System reliability target ``R*``.
        units: Positive number of units.
        mission_time: Optional mission time to report failure rates.
        names: Optional unit names.

    Returns:
        Equal allocation.

    Equation:
        [EQ-REL-ALLOC-001].

    References:
        [REF-EBELING-1997], [REF-MIL-HDBK-338B-1998].

    """
    goal = _target(target)
    if isinstance(units, bool) or not isinstance(units, int) or units < 1:
        raise ValueError("units must be a positive integer")
    labels = _names(names, units)
    share = goal ** (1.0 / units)
    rates = None
    if mission_time is not None:
        t = require_finite_real(mission_time, name="mission_time")
        if t <= 0.0:
            raise ValueError("mission_time must be positive")
        rates = tuple(-math.log(share) / t for _ in range(units))
    return AllocationResult(
        "equal",
        labels,
        tuple(share for _ in range(units)),
        rates,
        goal,
        share**units,
        True,
        math.nan,
        "Equal apportionment ignores differences in complexity and cost.",
    )


def _budget(target: float, mission_time: float) -> tuple[float, float]:
    """Return the validated target and the failure-rate budget ``-ln R* / t``.

    Args:
        target: Target ``R*``.
        mission_time: Positive mission time.

    Returns:
        ``(R*, lambda*)``.

    Equation:
        [EQ-REL-ALLOC-001].

    References:
        [REF-EBELING-1997].

    """
    goal = _target(target)
    t = require_finite_real(mission_time, name="mission_time")
    if t <= 0.0:
        raise ValueError("mission_time must be positive")
    return goal, -math.log(goal) / t


def arinc_allocation(
    target: float,
    failure_rates: Sequence[float],
    mission_time: float,
    *,
    names: Sequence[str] | None = None,
) -> AllocationResult:
    """Return the ARINC allocation ``lambda_i* = (lambda_i / sum lambda) lambda*``.

    Args:
        target: System reliability target ``R*``.
        failure_rates: Current (predicted or observed) positive unit rates.
        mission_time: Mission time ``t``.
        names: Optional unit names.

    Returns:
        Allocation meeting ``R*`` exactly.

    Equation:
        [EQ-REL-ALLOC-002].

    References:
        [REF-EBELING-1997], [REF-MIL-HDBK-338B-1998].

    """
    goal, budget = _budget(target, mission_time)
    rates = tuple(require_finite_real(v, name="failure rate") for v in failure_rates)
    if not rates or any(v <= 0.0 for v in rates):
        raise ValueError("failure_rates must be non-empty and positive")
    total = math.fsum(rates)
    allocated = tuple(budget * v / total for v in rates)
    current = math.exp(-total * float(mission_time))
    return _from_rates(
        "ARINC",
        _names(names, len(rates)),
        allocated,
        float(mission_time),
        goal,
        f"Current design reliability {current:.6g}; allocation scales every "
        f"rate by {budget / total:.6g}.",
    )


def agree_allocation(
    target: float,
    modules: Sequence[int],
    importance: Sequence[float],
    operating_times: Sequence[float],
    *,
    form: str = "exact",
    names: Sequence[str] | None = None,
) -> AllocationResult:
    """Return the AGREE allocation from complexity and importance.

    Args:
        target: System reliability target ``R*`` for the mission.
        modules: Positive module counts ``n_i`` (complexity).
        importance: Importance factors ``E_i`` in ``(0, 1]``.
        operating_times: Positive unit operating times ``t_i`` in the mission.
        form: ``"exact"`` (``R_i = 1 - (1 - R*^(n_i/N)) / E_i``) or
            ``"approximate"`` (``lambda_i = n_i (-ln R*) / (N E_i t_i)``).
        names: Optional unit names.

    Returns:
        Allocation; the achieved system reliability is
        ``prod_i [1 - E_i (1 - exp(-lambda_i t_i))]``, equal to ``R*`` for the
        exact form and close to it for the approximate one.

    Raises:
        ValueError: If inputs are invalid, or the exact form is infeasible
            (``E_i <= 1 - R*^(n_i/N)``).

    Equation:
        [EQ-REL-ALLOC-003].

    References:
        [REF-AGREE-1957], [REF-EBELING-1997].

    """
    goal = _target(target)
    counts = tuple(modules)
    for n in counts:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("modules must be positive integers")
    factors = tuple(require_probability(e, name="importance") for e in importance)
    times = tuple(
        require_finite_real(t, name="operating time") for t in operating_times
    )
    size = len(counts)
    if size == 0 or len(factors) != size or len(times) != size:
        raise ValueError("modules, importance, and operating_times must align")
    if any(e <= 0.0 for e in factors) or any(t <= 0.0 for t in times):
        raise ValueError("importance and operating times must be positive")
    if form not in ("exact", "approximate"):
        raise ValueError("form must be 'exact' or 'approximate'")
    total = sum(counts)
    rates = []
    for n, e, t in zip(counts, factors, times, strict=True):
        if form == "approximate":
            rates.append(n * -math.log(goal) / (total * e * t))
            continue
        unit = 1.0 - (1.0 - goal ** (n / total)) / e
        if unit <= 0.0:
            raise ValueError("exact AGREE form is infeasible for this importance")
        rates.append(-math.log(unit) / t)
    reliabilities = tuple(math.exp(-r * t) for r, t in zip(rates, times, strict=True))
    system = math.prod(
        1.0 - e * (1.0 - r) for e, r in zip(factors, reliabilities, strict=True)
    )
    return AllocationResult(
        f"AGREE ({form})",
        _names(names, size),
        reliabilities,
        tuple(rates),
        goal,
        system,
        system >= goal - 1e-12,
        math.nan,
        "Reliabilities are over each unit's operating time.",
    )


def feasibility_of_objectives(
    target: float,
    ratings: Sequence[Sequence[float]],
    mission_time: float,
    *,
    names: Sequence[str] | None = None,
) -> AllocationResult:
    """Return the feasibility-of-objectives allocation from factor ratings.

    Unit ``i`` gets ``lambda_i = (w_i / sum w) lambda*`` with ``w_i`` the
    product of its ratings (higher rating = harder to make reliable = larger
    share of the failure budget).

    Args:
        target: System reliability target ``R*``.
        ratings: Per unit, the ratings of the same factors (each 1-10).
        mission_time: Mission time ``t``.
        names: Optional unit names.

    Returns:
        Allocation meeting ``R*`` exactly.

    Equation:
        [EQ-REL-ALLOC-002].

    References:
        [REF-MIL-HDBK-338B-1998].

    """
    goal, budget = _budget(target, mission_time)
    rows = tuple(
        tuple(require_finite_real(v, name="rating") for v in r) for r in ratings
    )
    if not rows or any(len(r) != len(rows[0]) or not r for r in rows):
        raise ValueError("every unit needs the same non-empty set of ratings")
    if any(not 1.0 <= v <= 10.0 for r in rows for v in r):
        raise ValueError("ratings must lie in [1, 10]")
    weights = [math.prod(r) for r in rows]
    total = math.fsum(weights)
    return _from_rates(
        "feasibility of objectives",
        _names(names, len(rows)),
        tuple(budget * w / total for w in weights),
        float(mission_time),
        goal,
        "Weights are products of the factor ratings.",
    )


@dataclass(frozen=True, slots=True)
class MettasCost:
    """Cost of achieving reliability ``R``: ``exp((1 - f)(R - R_min)/(R_max - R))``.

    Args:
        feasibility: ``f`` in ``[0, 1)``; larger means cheaper to improve.
        r_min: Current (minimum) reliability, cost 1 there.
        r_max: Maximum achievable reliability (cost tends to infinity).
        scale: Positive multiplier of the cost (default 1).

    References:
        [REF-METTAS-2000].

    """

    feasibility: float
    r_min: float
    r_max: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If ``f`` is outside ``[0, 1)`` or not
                ``0 <= R_min < R_max <= 1``.

        References:
            [REF-METTAS-2000]. No equation is evaluated.

        """
        f = require_probability(self.feasibility, name="feasibility")
        low = require_probability(self.r_min, name="r_min")
        high = require_probability(self.r_max, name="r_max")
        scale = require_finite_real(self.scale, name="scale")
        if f >= 1.0 or not low < high or scale <= 0.0:
            raise ValueError("need 0 <= f < 1, r_min < r_max, scale > 0")
        object.__setattr__(self, "feasibility", f)
        object.__setattr__(self, "r_min", low)
        object.__setattr__(self, "r_max", high)
        object.__setattr__(self, "scale", scale)

    def __call__(self, reliability: float) -> float:
        """Return ``c(R)`` for ``R_min <= R < R_max``.

        Args:
            reliability: Unit reliability.

        Equation:
            [EQ-REL-ALLOC-004].

        References:
            [REF-METTAS-2000].

        """
        r = require_finite_real(reliability, name="reliability")
        if not self.r_min <= r < self.r_max:
            return math.inf
        exponent = (1.0 - self.feasibility) * (r - self.r_min) / (self.r_max - r)
        return self.scale * math.exp(min(exponent, 700.0))

    def derivative(self, reliability: float) -> float:
        """Return ``c'(R) = c(R) (1 - f)(R_max - R_min) / (R_max - R)^2``.

        Args:
            reliability: Unit reliability in ``[R_min, R_max)``.

        Equation:
            [EQ-REL-ALLOC-004].

        References:
            [REF-METTAS-2000].

        """
        r = require_finite_real(reliability, name="reliability")
        if not self.r_min <= r < self.r_max:
            return math.inf
        factor = (1.0 - self.feasibility) * (self.r_max - self.r_min)
        return self(r) * factor / (self.r_max - r) ** 2

    def log_derivative(self, reliability: float) -> float:
        """Return ``ln c'(R)`` without overflow near ``R_max``.

        Args:
            reliability: Unit reliability in ``[R_min, R_max)``.

        Returns:
            ``ln s + (1 - f)(R - R_min)/(R_max - R) + ln((1 - f)(R_max - R_min))
            - 2 ln(R_max - R)``.

        Equation:
            [EQ-REL-ALLOC-004].

        References:
            [REF-METTAS-2000].

        """
        r = require_finite_real(reliability, name="reliability")
        if not self.r_min <= r < self.r_max:
            return math.inf
        gap = self.r_max - r
        one_minus_f = 1.0 - self.feasibility
        return (
            math.log(self.scale)
            + one_minus_f * (r - self.r_min) / gap
            + math.log(one_minus_f * (self.r_max - self.r_min))
            - 2.0 * math.log(gap)
        )


def _log_marginal(cost: CostFunction) -> Callable[[float], float]:
    """Return ``r -> ln(r c'(r))``, clamped to ``[-1e300, 1e300]``.

    Uses ``cost.log_derivative`` when available, else the logarithm of the
    (analytic or finite-difference) derivative.

    Args:
        cost: Cost function.

    Returns:
        Log marginal-cost function used in the KKT condition.

    Equation:
        [EQ-REL-ALLOC-005].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 12].

    """
    log_method = getattr(cost, "log_derivative", None)
    slope = _derivative(cost)

    def value(r: float) -> float:
        """Return the clamped ``ln(r c'(r))``.

        References:
            [EQ-REL-ALLOC-005]. No new equation.

        """
        if r <= 0.0:
            return -1e300
        if log_method is not None:
            result = math.log(r) + float(log_method(r))
        else:
            d = slope(r)
            result = math.log(r) + math.log(d) if d > 0.0 else -math.inf
        if math.isnan(result):
            return 1e300
        return min(max(result, -1e300), 1e300)

    return value


def _derivative(cost: CostFunction) -> Callable[[float], float]:
    """Return the analytic derivative of a cost or a central difference.

    Args:
        cost: Cost function, optionally with a ``derivative`` method.

    Returns:
        Derivative function.

    Equation:
        [EQ-OPT-FD-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    method = getattr(cost, "derivative", None)
    if method is not None:
        return lambda r: float(method(r))

    def numeric(r: float) -> float:
        """Return a central-difference derivative of the cost.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        h = 1e-7 * max(1.0, abs(r))
        return (float(cost(r + h)) - float(cost(r - h))) / (2.0 * h)

    return numeric


def _bounds(
    costs: Sequence[CostFunction],
    lower: Sequence[float] | None,
    upper: Sequence[float] | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return unit bounds, defaulting to the Mettas ``R_min``/``R_max``.

    Args:
        costs: Cost functions.
        lower: Optional lower bounds.
        upper: Optional upper bounds (exclusive).

    Returns:
        Validated ``(lower, upper)``.

    Raises:
        ValueError: If a bound is missing for a non-Mettas cost or
            ``lower >= upper``.

    References:
        Project validation policy. No equation is evaluated.

    """
    size = len(costs)

    def default(index: int, attribute: str) -> float:
        """Return a Mettas bound or raise when none is available.

        References:
            [REF-METTAS-2000]. No equation is evaluated.

        """
        cost = costs[index]
        if isinstance(cost, MettasCost):
            return float(getattr(cost, attribute))
        raise ValueError("lower and upper bounds are required for non-Mettas costs")

    lows = (
        tuple(default(i, "r_min") for i in range(size))
        if lower is None
        else tuple(require_probability(v, name="lower") for v in lower)
    )
    highs = (
        tuple(default(i, "r_max") for i in range(size))
        if upper is None
        else tuple(require_probability(v, name="upper") for v in upper)
    )
    if len(lows) != size or len(highs) != size:
        raise ValueError("bounds need one entry per unit")
    if any(not a < b for a, b in zip(lows, highs, strict=True)):
        raise ValueError("every lower bound must be below its upper bound")
    return lows, highs


def cost_optimal_series_allocation(
    target: float,
    costs: Sequence[CostFunction],
    *,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> AllocationResult:
    """Return the exact cost-optimal allocation for a series system.

    Minimises ``sum c_i(R_i)`` subject to ``prod R_i >= R*`` and
    ``l_i <= R_i < u_i``. With ``x_i = ln R_i`` the constraint is linear and,
    for convex increasing costs, the KKT conditions are
    ``R_i c_i'(R_i) = mu`` for interior units and ``R_i = l_i`` when
    ``l_i c_i'(l_i) >= mu``. Each ``R_i(mu)`` is found by Brent's method and
    the multiplier ``mu`` by Brent's method on ``sum ln R_i(mu) = ln R*``.

    Args:
        target: System target ``R*``.
        costs: Convex increasing cost functions (optionally with a
            ``derivative`` method).
        lower: Lower bounds (default: ``MettasCost.r_min``).
        upper: Exclusive upper bounds (default: ``MettasCost.r_max``).
        names: Optional unit names.

    Returns:
        Optimal allocation with total cost and constraint check.

    Raises:
        ValueError: If the target is unreachable (``prod u_i <= R*``).

    Equation:
        [EQ-REL-ALLOC-005], [EQ-OPT-BRENT-001].

    References:
        [REF-METTAS-2000], [REF-NOCEDAL-WRIGHT-2006, Chapter 12],
        [REF-BRENT-1973, Chapter 4].

    """
    goal = _target(target)
    units = tuple(costs)
    if not units:
        raise ValueError("at least one cost function is required")
    lows, highs = _bounds(units, lower, upper)
    labels = _names(names, len(units))
    if math.fsum(math.log(u) for u in highs) <= math.log(goal):
        raise ValueError("target is not reachable below the upper bounds")
    marginals = [_log_marginal(c) for c in units]

    def unit_reliability(index: int, log_mu: float) -> float:
        """Return ``R_i(mu)`` solving ``ln(R c_i'(R)) = ln mu`` within the bounds.

        References:
            [EQ-REL-ALLOC-005], [REF-NOCEDAL-WRIGHT-2006, Chapter 12].

        """
        low, high = lows[index], highs[index]
        marginal = marginals[index]
        if marginal(low) >= log_mu:
            return low
        top = high - 1e-12 * max(1.0, high)
        if marginal(top) <= log_mu:
            return top
        return brent_root(
            lambda r: marginal(r) - log_mu, low, top, tolerance=1e-15
        ).root

    def excess(log_mu: float) -> float:
        """Return ``sum ln R_i(mu) - ln R*`` at ``mu = exp(log_mu)``.

        References:
            [EQ-REL-ALLOC-005]. No new equation.

        """
        return math.fsum(
            math.log(max(unit_reliability(i, log_mu), 1e-300))
            for i in range(len(units))
        ) - math.log(goal)

    if all(low > 0.0 for low in lows) and math.fsum(
        math.log(low) for low in lows
    ) >= math.log(goal):
        allocation = lows
    else:
        low_log, high_log = -40.0, 0.0
        while excess(high_log) < 0.0:
            high_log += 5.0
            if high_log > 1e6:
                raise ValueError("could not bracket the Lagrange multiplier")
        while excess(low_log) > 0.0 and low_log > -1e6:
            low_log -= 20.0
        root = brent_root(excess, low_log, high_log, tolerance=1e-13).root
        allocation = tuple(unit_reliability(i, root) for i in range(len(units)))
    system = math.prod(allocation)
    total = math.fsum(float(c(r)) for c, r in zip(units, allocation, strict=True))
    return AllocationResult(
        "cost-optimal series (Lagrange)",
        labels,
        tuple(allocation),
        None,
        goal,
        system,
        system >= goal * (1.0 - 1e-9),
        total,
        "Exact KKT solution for a series system with convex costs.",
    )


def cost_optimal_allocation(
    target: float,
    costs: Sequence[CostFunction],
    system_reliability: Callable[[Sequence[float]], float],
    *,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    start: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    tolerance: float = 1e-8,
    max_outer: int = 60,
) -> AllocationResult:
    """Minimise ``sum c_i(R_i)`` subject to ``R_sys(R) >= R*`` for any structure.

    An augmented Lagrangian for the inequality constraint
    ``h(R) = (R_sys(R) - R*) / (1 - R*) >= 0`` is minimised by BFGS over
    ``z`` with ``R_i = l_i + (u_i - l_i) / (1 + exp(-z_i))``; the multiplier
    is updated by ``mu <- max(0, mu - rho h)`` and the penalty increased
    when the violation does not shrink. A final repair moves ``R`` towards
    the upper bounds just far enough to satisfy the constraint (assumes
    ``R_sys`` is non-decreasing in every component). The result is a local
    optimum; for non-convex problems try several ``start`` points.

    Args:
        target: System target ``R*``.
        costs: Unit cost functions of reliability.
        system_reliability: ``R_sys`` as a function of the unit reliabilities.
        lower: Lower bounds (default: ``MettasCost.r_min``).
        upper: Exclusive upper bounds (default: ``MettasCost.r_max``).
        start: Optional interior starting reliabilities.
        names: Optional unit names.
        tolerance: Constraint tolerance on ``|h|`` for convergence.
        max_outer: Maximum multiplier updates.

    Returns:
        Allocation with total cost and the achieved system reliability.

    Raises:
        ValueError: If the target is unreachable at the upper bounds.

    Equation:
        [EQ-REL-ALLOC-006], [EQ-OPT-BFGS-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 17], [REF-METTAS-2000].

    """
    goal = _target(target)
    units = tuple(costs)
    if not units:
        raise ValueError("at least one cost function is required")
    lows, highs = _bounds(units, lower, upper)
    size = len(units)
    labels = _names(names, size)
    ceiling = tuple(h - 1e-9 * max(1.0, h) for h in highs)
    if float(system_reliability(ceiling)) < goal:
        raise ValueError("target is not reachable below the upper bounds")
    span = 1.0 - goal

    def to_reliability(z: Sequence[float]) -> tuple[float, ...]:
        """Map unconstrained ``z`` into the open boxes ``(l_i, u_i)``.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 17]. No new equation.

        """
        return tuple(
            low + (high - low) / (1.0 + math.exp(-max(min(v, 700.0), -700.0)))
            for v, low, high in zip(z, lows, highs, strict=True)
        )

    def total_cost(r: Sequence[float]) -> float:
        """Return ``sum c_i(R_i)``.

        References:
            [EQ-REL-ALLOC-006]. No new equation.

        """
        return math.fsum(float(c(v)) for c, v in zip(units, r, strict=True))

    def constraint(r: Sequence[float]) -> float:
        """Return the scaled constraint ``h(R)``.

        References:
            [EQ-REL-ALLOC-006]. No new equation.

        """
        return (float(system_reliability(r)) - goal) / span

    if start is None:
        initial = [0.0] * size
    else:
        points = tuple(require_probability(v, name="start") for v in start)
        if len(points) != size:
            raise ValueError("start needs one entry per unit")
        initial = []
        for v, low, high in zip(points, lows, highs, strict=True):
            fraction = min(max((v - low) / (high - low), 1e-9), 1.0 - 1e-9)
            initial.append(math.log(fraction / (1.0 - fraction)))
    scale = max(abs(total_cost(to_reliability(initial))), 1.0)
    mu, rho = 0.0, 10.0
    z = tuple(initial)
    previous_violation = math.inf
    converged = False
    for _ in range(max_outer):

        def merit(point: Sequence[float], mu: float = mu, rho: float = rho) -> float:
            """Return the augmented Lagrangian of the scaled problem.

            References:
                [EQ-REL-ALLOC-006], [REF-NOCEDAL-WRIGHT-2006, Chapter 17].

            """
            r = to_reliability(point)
            h = constraint(r)
            value = total_cost(r) / scale
            if not math.isfinite(value):
                return math.inf
            if h - mu / rho <= 0.0:
                return value - mu * h + 0.5 * rho * h * h
            return value - 0.5 * mu * mu / rho

        z = minimize_bfgs(merit, z, gradient_tolerance=1e-10, max_iterations=400).x
        h = constraint(to_reliability(z))
        violation = max(0.0, -h)
        mu = max(0.0, mu - rho * h)
        if violation <= tolerance and (mu == 0.0 or abs(h) <= tolerance):
            converged = True
            break
        if violation > 0.25 * previous_violation:
            rho *= 10.0
        previous_violation = violation
    allocation = to_reliability(z)
    repaired = False
    if constraint(allocation) < 0.0:
        base = allocation

        def along(s: float) -> float:
            """Return ``R_sys`` minus the target on the repair segment.

            References:
                [EQ-REL-ALLOC-006]. No new equation.

            """
            point = tuple(b + s * (c - b) for b, c in zip(base, ceiling, strict=True))
            return float(system_reliability(point)) - goal

        step = brent_root(along, 0.0, 1.0, tolerance=1e-15).root
        allocation = tuple(
            b + step * (c - b) for b, c in zip(base, ceiling, strict=True)
        )
        if along(step) < 0.0:
            step = min(1.0, step + 1e-12)
            allocation = tuple(
                b + step * (c - b) for b, c in zip(base, ceiling, strict=True)
            )
        repaired = True
    achieved = float(system_reliability(allocation))
    message = "Augmented Lagrangian " + ("converged" if converged else "stopped")
    if repaired:
        message += "; feasibility repair applied"
    return AllocationResult(
        "cost-optimal (augmented Lagrangian)",
        labels,
        tuple(allocation),
        None,
        goal,
        achieved,
        achieved >= goal * (1.0 - 1e-12),
        total_cost(allocation),
        message + ".",
    )
