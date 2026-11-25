"""Schedule risk analysis and joint cost-schedule confidence.

:mod:`probstat.cost.cost_risk` turns paired cost and schedule samples into a
joint confidence level, but nothing there produces the schedule samples. This
module does: it models a project as an *activity-on-node network*, runs the
deterministic critical path method, the classical PERT approximation, and a
Monte Carlo schedule risk simulation, and then loads cost onto the simulated
durations so that the paired (cost, completion) samples can be handed to
:func:`~probstat.cost.joint_confidence_level`.

**Network.** An :class:`Activity` has a duration, either a number or a law from
:mod:`probstat.distributions`, and a list of predecessors. A precedence is
finish-to-start (``"FS"``, the default), start-to-start (``"SS"``) or
finish-to-finish (``"FF"``), each with an optional lag (a negative lag is a
lead). Start-to-finish is not supported. Every activity starts at time zero
at the earliest; the graph must be acyclic, and a cycle is reported by name.

**Deterministic CPM.** :func:`critical_path` runs a forward pass (early
start and finish) and a backward pass (late start and finish) and reports the
total float ``LS - ES``, the free float (the smallest slack of the outgoing
relations), and every critical path (a chain of zero-float activities linked
by zero-slack relations).

**PERT.** :func:`pert_estimate` is the classical approximation of Malcolm et
al.: the mean and variance of the project duration are the sums along the
critical path of the mean-duration network, and the duration is taken as
normal. It ignores every path but one and so is *optimistic*: the expected
maximum of several path durations exceeds the maximum of their expectations
(merge-event bias; MacCrimmon and Ryavec, 1964). For a pure
series network of independent activities the mean and variance are exact; the
normal law (used by ``probability_by`` and ``quantile``) is exact only when
the activities are normal. It underestimates the mean as soon as parallel
paths compete.

**Simulation.** :func:`simulate_schedule` draws all activity durations,
optionally dependent through a Gaussian copula on the normal scores (the
:class:`~probstat.modeling.JointInput` machinery), adds optional discrete
risk events (a probability and an added-duration law on one activity, built on
:class:`~probstat.risk.Risk`), and runs the CPM on each draw. It reports the
completion-time distribution with distribution-free quantile intervals
(the ranks of
:func:`~probstat.propagation.order_statistic_interval`), the probability of
meeting a deadline with an exact Clopper-Pearson interval, and for each
activity:

- the *criticality index* (Van Slyke, 1963): the fraction of draws in which
  the activity has zero total float; activities tied on the critical path
  are all critical in that draw;
- the *cruciality* (Williams, 1992): the correlation of the activity's
  duration with the completion time; this module reports the Pearson
  coefficient and the Spearman rank coefficient;
- the *schedule sensitivity index*: criticality times the ratio of the
  activity's duration standard deviation to the completion standard
  deviation, the measure schedule risk software reports (the GAO Schedule
  Assessment Guide shows such an index without giving this formula; the
  definition is stated here, not quoted).

**Cost loading.** :func:`cost_schedule_samples` adds a time-independent cost
per activity, a burn rate times the activity's sampled duration, and a
project-level rate times the sampled completion time. Because the burn terms
grow with the same durations that delay completion, cost and completion are
positively dependent; :func:`joint_cost_schedule_confidence` and
:func:`cost_schedule_frontier` feed the pairs to the existing joint-confidence
functions.

**Joint confidence is lower than either marginal.** ``P(C <= c and S <= s)``
never exceeds ``min(P(C <= c), P(S <= s))``. A (cost, date) pair with a 70 %
cost confidence and a 70 % schedule confidence therefore has a joint
confidence of at most 70 %, and of about 49 % when cost and schedule are
independent. The iso-confidence frontier lists, for each deadline, the
smallest cost that reaches a stated *joint* confidence.

Assumptions: activity durations are non-negative (a negative draw raises an
error); durations of different activities are independent unless a
correlation is supplied; the dependence is the Gaussian copula, so tail
dependence is zero; risk events occur independently of one another and of the
durations; costs and burn rates are independent of durations apart from the
product with the sampled duration. Resources are not leveled, calendars are
not modeled, and a path through an activity of zero total float is the only
meaning of criticality.

Relation to neighbouring tools: the cost S-curve for element costs without a
schedule is :func:`~probstat.cost.cost_risk_analysis`; distribution-free
quantile intervals are :func:`~probstat.propagation.order_statistic_interval`;
discrete risk records are :class:`~probstat.risk.Risk`.

References:
    [REF-KELLEY-WALKER-1959], [REF-MALCOLM-1959], [REF-VAN-SLYKE-1963],
    [REF-MACCRIMMON-RYAVEC-1964], [REF-WILLIAMS-1992],
    [REF-ELMAGHRABY-2000], [REF-GAO-16-89G], [REF-NASA-CEH-2015].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from probstat.distributions import Binomial, Normal
from probstat.foundations import (
    RandomStream,
    require_finite_real,
    require_probability,
    resolve_stream,
)
from probstat.inference import ConfidenceInterval, clopper_pearson_interval
from probstat.modeling import (
    ContinuousMarginal,
    DiscreteMarginal,
    JointInput,
    RandomInput,
)
from probstat.propagation import QuantileInterval
from probstat.risk import DiscreteImpact, Risk
from probstat.statistics import pearson_correlation, quantile, spearman_correlation

from .cost_risk import (
    JointConfidenceLevel,
    cost_for_joint_confidence,
    joint_confidence_level,
)

RELATION_KINDS = ("FS", "SS", "FF")
CORRELATION_KINDS = ("copula", "pearson")
DEFAULT_CONFIDENCE = 0.95

DurationLaw = ContinuousMarginal | DiscreteMarginal


def _is_law(value: object) -> bool:
    """Return whether a value is a law rather than a number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    return not isinstance(value, bool | int | float)


def _name(value: object, *, what: str) -> str:
    """Return a validated non-empty name.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if not value.strip():
        raise ValueError(f"{what} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class Precedence:
    """One precedence relation to a predecessor activity.

    Args:
        activity: Name of the predecessor.
        kind: ``"FS"`` (successor starts after the predecessor finishes),
            ``"SS"`` (starts after the predecessor starts) or ``"FF"``
            (finishes after the predecessor finishes).
        lag: Finite lag in time units; negative means a lead.

    Raises:
        TypeError: If a field has an invalid type.
        ValueError: If the name is empty, the kind unknown, or the lag not
            finite.

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959].

    """

    activity: str
    kind: str = "FS"
    lag: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fields.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        _name(self.activity, what="predecessor name")
        if self.kind not in RELATION_KINDS:
            raise ValueError(f"kind must be one of {RELATION_KINDS}")
        object.__setattr__(self, "lag", require_finite_real(self.lag, name="lag"))


@dataclass(frozen=True)
class Activity:
    """One activity of an activity-on-node network.

    Args:
        name: Distinct non-empty label.
        duration: A non-negative finite number (deterministic) or a law from
            :mod:`probstat.distributions` with ``cdf``, ``quantile``, ``mean``
            and ``variance``. Sampled durations must be non-negative.
        predecessors: Names of predecessors (finish-to-start, no lag) or
            :class:`Precedence` records.
        description: Optional text.

    Raises:
        TypeError: If a field has an invalid type or the law lacks a method.
        ValueError: If the name is empty or a number is negative.

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959], [REF-MALCOLM-1959].

    """

    name: str
    duration: float | DurationLaw
    predecessors: tuple[str | Precedence, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the fields and normalize the predecessors.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        _name(self.name, what="name")
        if _is_law(self.duration):
            RandomInput(self.name, self.duration)  # type: ignore[arg-type]
        else:
            value = require_finite_real(self.duration, name="duration")
            if value < 0.0:
                raise ValueError("duration must be non-negative")
            object.__setattr__(self, "duration", value)
        relations: list[Precedence] = []
        for item in self.predecessors:
            relation = Precedence(item) if isinstance(item, str) else item
            if not isinstance(relation, Precedence):
                raise TypeError("predecessors must be names or Precedence records")
            relations.append(relation)
        object.__setattr__(self, "predecessors", tuple(relations))
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

    @property
    def is_random(self) -> bool:
        """Return whether the duration is a law.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        return _is_law(self.duration)

    @property
    def mean(self) -> float:
        """Return the expected duration.

        References:
            [REF-MALCOLM-1959].

        """
        if isinstance(self.duration, bool | int | float):
            return float(self.duration)
        return float(self.duration.mean)

    @property
    def variance(self) -> float:
        """Return the duration variance (zero for a number).

        References:
            [REF-MALCOLM-1959].

        """
        if isinstance(self.duration, bool | int | float):
            return 0.0
        return float(self.duration.variance)


def _find_cycle(
    names: Sequence[str], preds: Sequence[Sequence[int]], left: set[int]
) -> str:
    """Return a readable precedence cycle among the unsorted nodes.

    Every node left after a topological sort has a predecessor that is also
    left, so following predecessors from any of them must repeat a node.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    node = min(left)
    seen: dict[int, int] = {}
    walk: list[int] = []
    while node not in seen:
        seen[node] = len(walk)
        walk.append(node)
        node = next(p for p in preds[node] if p in left)
    cycle = list(reversed(walk[seen[node] :]))
    first = cycle.index(min(cycle))
    cycle = cycle[first:] + cycle[:first]
    labels = [names[i] for i in cycle]
    return " -> ".join([*labels, labels[0]])


@dataclass(frozen=True)
class ActivityNetwork:
    """A validated acyclic activity-on-node network.

    Args:
        activities: Activities with distinct names; every predecessor must be
            an activity of the network.

    Raises:
        TypeError: If an item is not an :class:`Activity`.
        ValueError: If the network is empty, a name repeats, a predecessor is
            unknown or the activity itself, or the precedence graph has a
            cycle (the message names it, for example ``a -> b -> a``).

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959].

    """

    activities: tuple[Activity, ...]
    _order: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _preds: tuple[tuple[tuple[int, str, float], ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _succs: tuple[tuple[tuple[int, str, float], ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the graph and cache the topological order.

        References:
            [REF-KELLEY-WALKER-1959].

        """
        items = tuple(self.activities)
        if not items:
            raise ValueError("a network needs at least one activity")
        for item in items:
            if not isinstance(item, Activity):
                raise TypeError("activities must be Activity instances")
        names = [a.name for a in items]
        index = {name: i for i, name in enumerate(names)}
        if len(index) != len(names):
            raise ValueError("activity names must be distinct")
        preds: list[list[tuple[int, str, float]]] = [[] for _ in items]
        succs: list[list[tuple[int, str, float]]] = [[] for _ in items]
        for i, item in enumerate(items):
            for relation in item.predecessors:
                assert isinstance(relation, Precedence)
                if relation.activity not in index:
                    raise ValueError(
                        f"activity {item.name!r} has unknown predecessor "
                        f"{relation.activity!r}"
                    )
                p = index[relation.activity]
                if p == i:
                    raise ValueError(f"activity {item.name!r} precedes itself")
                preds[i].append((p, relation.kind, relation.lag))
                succs[p].append((i, relation.kind, relation.lag))
        remaining = [len({p for p, _, _ in preds[i]}) for i in range(len(items))]
        order: list[int] = []
        ready = [i for i in range(len(items)) if remaining[i] == 0]
        while ready:
            node = ready.pop(0)
            order.append(node)
            for s in sorted({s for s, _, _ in succs[node]}):
                remaining[s] -= 1
                if remaining[s] == 0:
                    ready.append(s)
        if len(order) != len(items):
            left = set(range(len(items))) - set(order)
            plain = [[p for p, _, _ in row] for row in preds]
            raise ValueError(
                "the precedence graph has a cycle: " + _find_cycle(names, plain, left)
            )
        object.__setattr__(self, "activities", items)
        object.__setattr__(self, "_order", tuple(order))
        object.__setattr__(self, "_preds", tuple(tuple(r) for r in preds))
        object.__setattr__(self, "_succs", tuple(tuple(r) for r in succs))

    @property
    def names(self) -> tuple[str, ...]:
        """Return the activity names in input order.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        return tuple(a.name for a in self.activities)

    @property
    def topological_order(self) -> tuple[str, ...]:
        """Return the names in a valid precedence order.

        References:
            [REF-KELLEY-WALKER-1959].

        """
        return tuple(self.activities[i].name for i in self._order)

    @property
    def has_only_finish_to_start(self) -> bool:
        """Return whether every relation is finish-to-start.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        return all(kind == "FS" for row in self._preds for _, kind, _ in row)

    def index(self, name: str) -> int:
        """Return the position of an activity.

        Raises:
            ValueError: If the name is unknown.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(f"unknown activity {name!r}") from None


def _passes(
    network: ActivityNetwork, d: Sequence[float]
) -> tuple[list[float], list[float], list[float], list[float], float]:
    """Return ``(ES, EF, LS, LF, project duration)`` for given durations.

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959].

    """
    n = len(d)
    es = [0.0] * n
    ef = [0.0] * n
    for i in network._order:
        start = 0.0
        for p, kind, lag in network._preds[i]:
            if kind == "FS":
                bound = ef[p] + lag
            elif kind == "SS":
                bound = es[p] + lag
            else:
                bound = ef[p] + lag - d[i]
            if bound > start:
                start = bound
        es[i] = start
        ef[i] = start + d[i]
    finish = max(ef)
    lf = [finish] * n
    ls = [finish] * n
    for i in reversed(network._order):
        late = finish
        for s, kind, lag in network._succs[i]:
            if kind == "FS":
                bound = ls[s] - lag
            elif kind == "SS":
                bound = ls[s] - lag + d[i]
            else:
                bound = lf[s] - lag
            if bound < late:
                late = bound
        lf[i] = late
        ls[i] = late - d[i]
    return es, ef, ls, lf, finish


def _tolerance(finish: float) -> float:
    """Return the float tolerance that marks an activity as critical.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    return 1e-9 * max(1.0, abs(finish))


@dataclass(frozen=True, slots=True)
class ActivityTiming:
    """Early and late dates and floats of one activity.

    Args:
        name: Activity name.
        duration: Duration used.
        early_start: Earliest start ``ES``.
        early_finish: Earliest finish ``EF = ES + d``.
        late_start: Latest start ``LS`` that keeps the project duration.
        late_finish: Latest finish ``LF = LS + d``.
        total_float: ``LS - ES``.
        free_float: Smallest slack of the outgoing relations (for an
            activity with no successor, project duration minus ``EF``).
        critical: Whether the total float is zero.

    References:
        [REF-KELLEY-WALKER-1959].

    """

    name: str
    duration: float
    early_start: float
    early_finish: float
    late_start: float
    late_finish: float
    total_float: float
    free_float: float
    critical: bool


@dataclass(frozen=True)
class CriticalPathResult:
    """Deterministic critical path method result.

    Args:
        duration: Project duration (largest early finish).
        timings: One :class:`ActivityTiming` per activity in input order.
        critical_paths: Every critical path as a tuple of names, source to
            sink (at most ``max_paths``).
        paths_truncated: Whether more critical paths exist than reported.

    References:
        [REF-KELLEY-WALKER-1959].

    """

    duration: float
    timings: tuple[ActivityTiming, ...]
    critical_paths: tuple[tuple[str, ...], ...]
    paths_truncated: bool

    def timing(self, name: str) -> ActivityTiming:
        """Return the timing of one activity.

        Raises:
            ValueError: If the name is unknown.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        for row in self.timings:
            if row.name == name:
                return row
        raise ValueError(f"unknown activity {name!r}")

    def table(self) -> str:
        """Return the timings as a plain-text table.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        rows = [
            (
                t.name,
                _num(t.duration),
                _num(t.early_start),
                _num(t.early_finish),
                _num(t.late_start),
                _num(t.late_finish),
                _num(t.total_float),
                _num(t.free_float),
                "*" if t.critical else "",
            )
            for t in self.timings
        ]
        head = ("activity", "dur", "ES", "EF", "LS", "LF", "TF", "FF", "crit")
        lines = [_render(head, rows), f"project duration: {_num(self.duration)}"]
        for path in self.critical_paths:
            lines.append("critical path: " + " -> ".join(path))
        if self.paths_truncated:
            lines.append("(more critical paths exist)")
        return "\n".join(lines)


def _num(value: float) -> str:
    """Format a number compactly for tables.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    return f"{value:.4g}"


def _render(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a left-aligned plain-text table with a rule under the header.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    widths = [
        max(len(head[k]), *(len(r[k]) for r in rows)) if rows else len(head[k])
        for k in range(len(head))
    ]

    def line(cells: Sequence[str]) -> str:
        """Join padded cells.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        return "  ".join(
            c.ljust(w) for c, w in zip(cells, widths, strict=True)
        ).rstrip()

    return "\n".join([line(head), "  ".join("-" * w for w in widths), *map(line, rows)])


def _build_timings(
    network: ActivityNetwork, d: Sequence[float]
) -> tuple[float, tuple[ActivityTiming, ...], list[bool]]:
    """Return the project duration, the timings, and the criticality flags.

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959].

    """
    es, ef, ls, lf, finish = _passes(network, d)
    tol = _tolerance(finish)
    rows: list[ActivityTiming] = []
    flags: list[bool] = []
    for i, activity in enumerate(network.activities):
        slack = finish - ef[i]
        for s, kind, lag in network._succs[i]:
            if kind == "FS":
                value = es[s] - ef[i] - lag
            elif kind == "SS":
                value = es[s] - es[i] - lag
            else:
                value = ef[s] - ef[i] - lag
            slack = min(slack, value)
        total = ls[i] - es[i]
        critical = total <= tol
        flags.append(critical)
        rows.append(
            ActivityTiming(
                activity.name,
                d[i],
                es[i],
                ef[i],
                ls[i],
                lf[i],
                0.0 if critical else total,
                max(0.0, slack),
                critical,
            )
        )
    return finish, tuple(rows), flags


def _critical_paths(
    network: ActivityNetwork,
    d: Sequence[float],
    flags: Sequence[bool],
    limit: int,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    """Enumerate the chains of critical activities joined by tight relations.

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959].

    """
    es, ef, _, _, finish = _passes(network, d)
    tol = _tolerance(finish)
    tight: list[list[int]] = [[] for _ in d]
    has_tight_in = [False] * len(d)
    for i in range(len(d)):
        if not flags[i]:
            continue
        for s, kind, lag in network._succs[i]:
            if not flags[s]:
                continue
            if kind == "FS":
                value = es[s] - ef[i] - lag
            elif kind == "SS":
                value = es[s] - es[i] - lag
            else:
                value = ef[s] - ef[i] - lag
            if value <= tol and s not in tight[i]:
                tight[i].append(s)
                has_tight_in[s] = True
    starts = [i for i in network._order if flags[i] and not has_tight_in[i]]
    paths: list[tuple[str, ...]] = []
    truncated = False
    stack: list[tuple[int, tuple[int, ...]]] = [(i, (i,)) for i in reversed(starts)]
    while stack:
        node, trail = stack.pop()
        if not tight[node]:
            if len(paths) >= limit:
                truncated = True
                break
            paths.append(tuple(network.activities[k].name for k in trail))
            continue
        stack.extend((s, (*trail, s)) for s in reversed(tight[node]))
    return tuple(paths), truncated


def critical_path(
    network: ActivityNetwork,
    durations: Mapping[str, float] | None = None,
    *,
    max_paths: int = 100,
) -> CriticalPathResult:
    """Run the critical path method on fixed durations.

    Args:
        network: Activity network.
        durations: Optional override of activity durations by name; the
            others use their number, or their mean when the duration is a
            law (so the default is the CPM at the expected durations).
        max_paths: Largest number of critical paths to enumerate.

    Returns:
        Early and late dates, total and free float, and the critical paths.

    Raises:
        TypeError: If the network is not an :class:`ActivityNetwork`.
        ValueError: If an override names an unknown activity or is negative,
            or ``max_paths`` is not a positive integer.

    Equation:
        [EQ-COST-CPM-001].

    References:
        [REF-KELLEY-WALKER-1959].

    """
    if not isinstance(network, ActivityNetwork):
        raise TypeError("network must be an ActivityNetwork")
    if isinstance(max_paths, bool) or not isinstance(max_paths, int) or max_paths < 1:
        raise ValueError("max_paths must be a positive integer")
    d = [a.mean for a in network.activities]
    for name, value in (durations or {}).items():
        v = require_finite_real(value, name="duration")
        if v < 0.0:
            raise ValueError("durations must be non-negative")
        d[network.index(name)] = v
    finish, timings, flags = _build_timings(network, d)
    paths, truncated = _critical_paths(network, d, flags, max_paths)
    return CriticalPathResult(finish, timings, paths, truncated)


@dataclass(frozen=True)
class PertEstimate:
    """The classical PERT normal approximation of the project duration.

    Args:
        mean: Sum of the expected durations (and lags) along the path.
        variance: Sum of the duration variances along the path.
        path: The critical path used (of the mean-duration network, the one
            with the largest variance when several are critical).
        paths: Number of critical paths found.
        notes: Plain-language caveats.

    References:
        [REF-MALCOLM-1959], [REF-MACCRIMMON-RYAVEC-1964].

    """

    mean: float
    variance: float
    path: tuple[str, ...]
    paths: int
    notes: tuple[str, ...]

    @property
    def standard_deviation(self) -> float:
        """Return the square root of the variance.

        References:
            [REF-MALCOLM-1959].

        """
        return math.sqrt(self.variance)

    def probability_by(self, deadline: float) -> float:
        """Return the normal-approximation ``P(duration <= deadline)``.

        Equation:
            [EQ-COST-PERT-001].

        References:
            [REF-MALCOLM-1959].

        """
        t = require_finite_real(deadline, name="deadline")
        if self.variance <= 0.0:
            return 1.0 if t >= self.mean else 0.0
        return float(Normal(self.mean, self.standard_deviation).cdf(t))

    def quantile(self, probability: float) -> float:
        """Return the normal-approximation quantile of the duration.

        Equation:
            [EQ-COST-PERT-001].

        References:
            [REF-MALCOLM-1959].

        """
        p = require_probability(probability, name="probability")
        if not 0.0 < p < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        if self.variance <= 0.0:
            return self.mean
        return float(Normal(self.mean, self.standard_deviation).quantile(p))


def pert_three_point(
    optimistic: float, most_likely: float, pessimistic: float
) -> tuple[float, float]:
    """Return the classical PERT mean and variance of an activity duration.

    Args:
        optimistic: Optimistic duration ``a``.
        most_likely: Most likely duration ``m``.
        pessimistic: Pessimistic duration ``b``.

    Returns:
        ``((a + 4 m + b) / 6, ((b - a) / 6)^2)``, the approximations of the
        mean and variance of a beta-distributed duration (an approximation,
        not an identity; compare :class:`probstat.distributions.PERT`).

    Raises:
        ValueError: If ``a <= m <= b`` fails (``a == b`` is a point mass).

    Equation:
        [EQ-COST-PERT-001].

    References:
        [REF-MALCOLM-1959].

    """
    a = require_finite_real(optimistic, name="optimistic")
    m = require_finite_real(most_likely, name="most_likely")
    b = require_finite_real(pessimistic, name="pessimistic")
    if not a <= m <= b:
        raise ValueError(
            "estimates must satisfy optimistic <= most_likely <= pessimistic"
        )
    return (a + 4.0 * m + b) / 6.0, ((b - a) / 6.0) ** 2


def pert_estimate(network: ActivityNetwork) -> PertEstimate:
    """Return the classical PERT approximation for a network.

    The critical path is found with the expected durations; the project mean
    is the path length and the variance is the sum of the activity variances
    along it. When several paths are critical the one with the largest
    variance is used. Paths that are nearly critical, merge events, and
    correlation are ignored, so the mean is biased low: the true expected
    duration is at least the length of the longest expected path (Jensen's
    inequality for the maximum).

    Args:
        network: Activity network with finish-to-start relations only.

    Returns:
        Mean, variance, and the path used.

    Raises:
        TypeError: If the network is not an :class:`ActivityNetwork`.
        ValueError: If a start-to-start or finish-to-finish relation exists.

    Equation:
        [EQ-COST-PERT-001].

    References:
        [REF-MALCOLM-1959], [REF-MACCRIMMON-RYAVEC-1964].

    """
    if not isinstance(network, ActivityNetwork):
        raise TypeError("network must be an ActivityNetwork")
    if not network.has_only_finish_to_start:
        raise ValueError("the PERT approximation needs finish-to-start relations only")
    cpm = critical_path(network)
    variance = {a.name: a.variance for a in network.activities}
    best = max(
        cpm.critical_paths, key=lambda path: math.fsum(variance[n] for n in path)
    )
    return PertEstimate(
        cpm.duration,
        math.fsum(variance[n] for n in best),
        best,
        len(cpm.critical_paths),
        (
            "PERT keeps one path: the mean is biased low where paths merge "
            "(MacCrimmon and Ryavec 1964) and correlation is ignored.",
        ),
    )


@dataclass(frozen=True)
class ScheduleRiskEvent:
    """A discrete risk event that lengthens one activity when it occurs.

    Args:
        risk: A :class:`probstat.risk.Risk` whose ``probability`` is the
            chance of occurrence and whose ``impact`` (a number, a
            :class:`~probstat.risk.DiscreteImpact`, or a law with
            ``quantile``) is the added duration in the network's time unit.
        activity: Name of the affected activity.
        mitigated: Use the risk's mitigated probability and impact.

    Raises:
        TypeError: If ``risk`` is not a :class:`~probstat.risk.Risk`.
        ValueError: If the activity name is empty.

    Equation:
        [EQ-COST-SRA-003].

    References:
        [REF-GAO-16-89G], [REF-NASA-PRA-2011].

    """

    risk: Risk
    activity: str
    mitigated: bool = False

    def __post_init__(self) -> None:
        """Validate the fields.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        if not isinstance(self.risk, Risk):
            raise TypeError("risk must be a Risk")
        _name(self.activity, what="activity")

    def draw(self, uniform_occurrence: float, uniform_impact: float) -> float:
        """Return the added duration for two uniform draws (zero if absent).

        Equation:
            [EQ-COST-SRA-003].

        References:
            [REF-GAO-16-89G].

        """
        probability, impact = self.risk.state(mitigated=self.mitigated)
        if uniform_occurrence >= probability:
            return 0.0
        if isinstance(impact, float):
            return impact
        assert isinstance(impact, DiscreteImpact) or hasattr(impact, "quantile")
        return float(impact.quantile(uniform_impact))


@dataclass(frozen=True, slots=True)
class ActivityRisk:
    """Simulated risk measures of one activity.

    Args:
        name: Activity name.
        mean_duration: Sample mean of the simulated duration (including any
            risk events on it).
        standard_deviation: Sample standard deviation of the duration.
        criticality: Fraction of draws with zero total float.
        criticality_interval: Clopper-Pearson interval for the criticality.
        cruciality: Pearson correlation of duration with completion time
            (zero when either is constant).
        rank_cruciality: Spearman correlation of the same pair.
        sensitivity: Schedule sensitivity index: ``criticality *
            standard_deviation / completion standard deviation``.

    References:
        [REF-VAN-SLYKE-1963], [REF-WILLIAMS-1992], [REF-ELMAGHRABY-2000].

    """

    name: str
    mean_duration: float
    standard_deviation: float
    criticality: float
    criticality_interval: ConfidenceInterval
    cruciality: float
    rank_cruciality: float
    sensitivity: float


@dataclass(frozen=True, slots=True)
class DeadlineProbability:
    """Probability of finishing by a deadline with an exact interval.

    Args:
        deadline: The date or duration target.
        successes: Draws with completion at or before the deadline.
        trials: Number of draws.
        probability: ``successes / trials``.
        interval: Clopper-Pearson interval.
        standard_error: ``sqrt(p (1 - p) / n)``.

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-GAO-16-89G].

    """

    deadline: float
    successes: int
    trials: int
    probability: float
    interval: ConfidenceInterval
    standard_error: float


def _quantile_interval(
    ordered: Sequence[float], probability: float, confidence: float
) -> QuantileInterval:
    """Return the equal-tailed distribution-free interval of a quantile.

    The ranks are those of
    :func:`~probstat.propagation.order_statistic_interval`, ``l = max{k :
    B(k - 1) <= alpha / 2}`` and ``u = min{k : B(k - 1) >= 1 - alpha / 2}``
    with ``B`` the ``Binomial(n, p)`` distribution function, found here by
    bisection on ``B`` (monotone), because the tabulating routine is
    quadratic in the sample size and a schedule simulation has thousands of
    draws. A test checks the two agree.

    Raises:
        ValueError: If a level is outside ``(0, 1)`` or the sample is too
            small for the requested confidence.

    Equation:
        [EQ-COST-SRA-002], [EQ-PROP-QCI-001].

    References:
        [REF-DAVID-NAGARAJA-2003, Section 7.1].

    """
    p = require_finite_real(probability, name="probability")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    n = len(ordered)
    if n < 2:
        raise ValueError("a quantile interval needs at least two values")
    tail = 0.5 * (1.0 - level)
    law = Binomial(n, p)

    def last_rank(holds: Callable[[int], bool]) -> int:
        """Return the largest rank in ``1..n`` where a monotone test holds.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        low, high = 0, n
        while low < high:
            mid = (low + high + 1) // 2
            if holds(mid):
                low = mid
            else:
                high = mid - 1
        return low

    lower = last_rank(lambda k: law.cdf(k - 1) <= tail)
    upper = n + 1 - last_rank(lambda k: law.cdf(n - k) >= 1.0 - tail)
    if lower == 0 or upper > n or upper <= lower:
        raise ValueError(
            f"{n} values are too few for a {level:g} distribution-free "
            f"interval of the {p:g} quantile"
        )
    return QuantileInterval(
        p,
        quantile(ordered, p),
        ordered[lower - 1],
        ordered[upper - 1],
        lower,
        upper,
        level,
        law.cdf(upper - 1) - law.cdf(lower - 1),
    )


@dataclass(frozen=True)
class ScheduleRiskResult:
    """Completion-time distribution and activity risk measures.

    Args:
        network: The simulated network.
        deterministic: CPM at the expected durations.
        pert: Classical PERT approximation (``None`` unless every relation is
            finish-to-start).
        sample_size: Number of draws.
        completion: Completion times in draw order.
        durations: Per-draw activity durations in network order (risk-event
            additions included), in draw order.
        mean: Sample mean completion time.
        standard_deviation: Sample standard deviation.
        standard_error: Standard error of the mean.
        activities: One :class:`ActivityRisk` per activity in input order.
        confidence: Confidence of the reported intervals.
        notes: Assumptions in plain language.

    References:
        [REF-VAN-SLYKE-1963], [REF-GAO-16-89G].

    """

    network: ActivityNetwork
    deterministic: CriticalPathResult
    pert: PertEstimate | None
    sample_size: int
    completion: tuple[float, ...]
    durations: tuple[tuple[float, ...], ...]
    mean: float
    standard_deviation: float
    standard_error: float
    activities: tuple[ActivityRisk, ...]
    confidence: float
    notes: tuple[str, ...]
    _ordered: tuple[float, ...] = field(repr=False, compare=False, default=())

    @property
    def ordered(self) -> tuple[float, ...]:
        """Return the completion times in ascending order.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        return self._ordered

    def activity(self, name: str) -> ActivityRisk:
        """Return the risk measures of one activity.

        Raises:
            ValueError: If the name is unknown.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        for row in self.activities:
            if row.name == name:
                return row
        raise ValueError(f"unknown activity {name!r}")

    def quantile(
        self, probability: float, confidence: float | None = None
    ) -> QuantileInterval:
        """Return a completion-time quantile with its order-statistic interval.

        Args:
            probability: Level strictly inside ``(0, 1)``.
            confidence: Interval confidence (default: the run's).

        Returns:
            Distribution-free interval (see
            :func:`~probstat.propagation.order_statistic_interval`).

        Equation:
            [EQ-COST-SRA-002].

        References:
            [REF-DAVID-NAGARAJA-2003, Section 7.1].

        """
        level = self.confidence if confidence is None else confidence
        return _quantile_interval(self.ordered, probability, level)

    def probability_by(
        self, deadline: float, confidence: float | None = None
    ) -> DeadlineProbability:
        """Return ``P(completion <= deadline)`` with a Clopper-Pearson interval.

        Args:
            deadline: Target completion time.
            confidence: Interval confidence (default: the run's).

        Equation:
            [EQ-COST-SRA-002].

        References:
            [REF-CLOPPER-PEARSON-1934], [REF-GAO-16-89G].

        """
        t = require_finite_real(deadline, name="deadline")
        level = self.confidence if confidence is None else confidence
        count = sum(1 for c in self.completion if c <= t)
        n = self.sample_size
        p = count / n
        return DeadlineProbability(
            t,
            count,
            n,
            p,
            clopper_pearson_interval(count, n, confidence=level),
            math.sqrt(p * (1.0 - p) / n),
        )

    def s_curve(self, points: int = 21) -> tuple[tuple[float, float], ...]:
        """Return ``(completion time, empirical CDF)`` on an equally spaced grid.

        Args:
            points: Number of grid points (at least two).

        Equation:
            [EQ-COST-SRA-002].

        References:
            [REF-GAO-16-89G].

        """
        if isinstance(points, bool) or not isinstance(points, int) or points < 2:
            raise ValueError("points must be an integer of at least two")
        low, high = self.ordered[0], self.ordered[-1]
        width = (high - low) or 1.0
        grid = [low + width * k / (points - 1) for k in range(points)]
        return tuple((t, self.probability_by(t).probability) for t in grid)

    def table(self) -> str:
        """Return the completion summary and activity measures as plain text.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        head = ("activity", "mean", "sd", "crit", "crit CI", "cruc", "SSI")
        rows = [
            (
                a.name,
                _num(a.mean_duration),
                _num(a.standard_deviation),
                f"{a.criticality:.3f}",
                f"[{a.criticality_interval.lower:.3f}, "
                f"{a.criticality_interval.upper:.3f}]",
                f"{a.cruciality:.3f}",
                f"{a.sensitivity:.3f}",
            )
            for a in self.activities
        ]
        lines = [
            f"completion: mean {_num(self.mean)}, sd {_num(self.standard_deviation)}, "
            f"n = {self.sample_size}; CPM at expected durations "
            f"{_num(self.deterministic.duration)}",
        ]
        if self.pert is not None:
            lines.append(
                f"PERT: mean {_num(self.pert.mean)}, "
                f"sd {_num(self.pert.standard_deviation)}"
            )
        for p in (0.5, 0.8, 0.9):
            q = self.quantile(p)
            lines.append(
                f"P{round(100 * p)}: {_num(q.estimate)} "
                f"[{_num(q.lower)}, {_num(q.upper)}]"
            )
        lines.append(_render(head, rows))
        return "\n".join(lines)


def _correlation_matrix(
    network: ActivityNetwork,
    random_names: Sequence[str],
    correlation: Mapping[tuple[str, str], float] | float | None,
) -> tuple[tuple[float, ...], ...] | None:
    """Return the correlation matrix over the random activities.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if correlation is None:
        return None
    if not isinstance(correlation, Mapping) and len(random_names) < 2:
        require_finite_real(correlation, name="correlation")
        return None
    k = len(random_names)
    matrix = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    if isinstance(correlation, Mapping):
        for (first, second), value in correlation.items():
            r = require_finite_real(value, name="correlation")
            if not -1.0 <= r <= 1.0:
                raise ValueError("correlation entries must lie in [-1, 1]")
            for name in (first, second):
                network.index(name)
                if name not in random_names:
                    raise ValueError(f"activity {name!r} has a fixed duration")
            if first == second:
                raise ValueError("a correlation pair needs two distinct activities")
            i, j = random_names.index(first), random_names.index(second)
            matrix[i][j] = matrix[j][i] = r
    else:
        r = require_finite_real(correlation, name="correlation")
        if not -1.0 < r < 1.0:
            raise ValueError("a common correlation must lie strictly inside (-1, 1)")
        for i in range(k):
            for j in range(k):
                if i != j:
                    matrix[i][j] = r
    return tuple(tuple(row) for row in matrix) if k >= 2 else None


def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the Pearson correlation, or zero when a sample is constant.

    Equation:
        [EQ-COST-SRA-001].

    References:
        [REF-WILLIAMS-1992].

    """
    try:
        return pearson_correlation(x, y)
    except ValueError:
        return 0.0


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the Spearman correlation, or zero when a sample is constant.

    Equation:
        [EQ-COST-SRA-001].

    References:
        [REF-WILLIAMS-1992].

    """
    try:
        return spearman_correlation(x, y)
    except ValueError:
        return 0.0


def simulate_schedule(
    network: ActivityNetwork,
    *,
    samples: int = 10_000,
    correlation: Mapping[tuple[str, str], float] | float | None = None,
    correlation_kind: str = "copula",
    risk_events: Iterable[ScheduleRiskEvent] = (),
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | RandomStream | None = None,
) -> ScheduleRiskResult:
    """Simulate the schedule and return completion and activity risk measures.

    Each draw samples every random duration (dependent through a Gaussian
    copula when ``correlation`` is given), adds the realized risk events to
    their activities, runs the CPM, and records the completion time and which
    activities have zero total float.

    Args:
        network: Activity network.
        samples: Number of draws (at least 100).
        correlation: ``None`` (independent), a common correlation for all
            random activities, or a mapping from name pairs to a correlation.
        correlation_kind: ``"copula"`` reads the values as the latent
            correlation of the normal scores (the Gaussian copula parameter,
            the default); ``"pearson"`` solves for the latent correlation that
            reproduces the stated Pearson correlations of the durations
            (Nataf; needs continuous laws and attainable values).
        risk_events: Discrete risk events adding duration to activities.
        confidence: Confidence of the reported intervals, in ``(0, 1)``.
        seed: Integer seed or stream; ``None`` is not reproducible.

    Returns:
        A :class:`ScheduleRiskResult` that also holds the per-draw durations
        for cost loading.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is out of range, a correlation is not
            valid for the activities, or a sampled duration is negative.

    Equation:
        [EQ-COST-SRA-001], [EQ-COST-SRA-002], [EQ-COST-SRA-003],
        [EQ-MOD-NATAF-002].

    References:
        [REF-VAN-SLYKE-1963], [REF-WILLIAMS-1992], [REF-GAO-16-89G],
        [REF-LIU-DERKIUREGHIAN-1986].

    """
    if not isinstance(network, ActivityNetwork):
        raise TypeError("network must be an ActivityNetwork")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("samples must be an integer of at least 100")
    if correlation_kind not in CORRELATION_KINDS:
        raise ValueError(f"correlation_kind must be one of {CORRELATION_KINDS}")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    events = tuple(risk_events)
    for event in events:
        if not isinstance(event, ScheduleRiskEvent):
            raise TypeError("risk_events must be ScheduleRiskEvent instances")
    event_index = [network.index(e.activity) for e in events]
    stream = resolve_stream(seed)
    n = len(network.activities)
    activities = network.activities
    random_index = [i for i, a in enumerate(activities) if a.is_random]
    random_names = [activities[i].name for i in random_index]
    matrix = _correlation_matrix(network, random_names, correlation)
    base = [a.mean if not a.is_random else 0.0 for a in activities]
    draws: Sequence[Sequence[float]] = ()
    if random_index:
        inputs = [
            RandomInput(activities[i].name, activities[i].duration)  # type: ignore[arg-type]
            for i in random_index
        ]
        if matrix is None:
            joint = JointInput(tuple(inputs))
        elif correlation_kind == "copula":
            joint = JointInput(tuple(inputs), matrix)
        else:
            joint = JointInput.from_pearson(inputs, matrix)
        draws = joint.sample(samples, seed=stream.child("durations"))
    generators = [stream.child("risk", k).generator() for k in range(len(events))]

    counts = [0] * n
    completion: list[float] = []
    kept: list[tuple[float, ...]] = []
    for t in range(samples):
        d = list(base)
        if random_index:
            row = draws[t]
            for slot, i in enumerate(random_index):
                d[i] = row[slot]
        for event, i, generator in zip(events, event_index, generators, strict=True):
            d[i] += event.draw(generator.random(), generator.random())
        for i, value in enumerate(d):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(
                    f"activity {activities[i].name!r} drew the invalid duration "
                    f"{value}; use a law with non-negative support"
                )
        es, ef, ls, _, finish = _passes(network, d)
        tol = _tolerance(finish)
        for i in range(n):
            if ls[i] - es[i] <= tol:
                counts[i] += 1
        completion.append(finish)
        kept.append(tuple(d))

    mean = math.fsum(completion) / samples
    sd = math.sqrt(math.fsum((c - mean) ** 2 for c in completion) / (samples - 1))
    rows: list[ActivityRisk] = []
    for i, activity in enumerate(activities):
        column = [row[i] for row in kept]
        m = math.fsum(column) / samples
        s = math.sqrt(math.fsum((v - m) ** 2 for v in column) / (samples - 1))
        index = counts[i] / samples
        rows.append(
            ActivityRisk(
                activity.name,
                m,
                s,
                index,
                clopper_pearson_interval(counts[i], samples, confidence=level),
                _safe_pearson(column, completion),
                _safe_spearman(column, completion),
                index * s / sd if sd > 0.0 else 0.0,
            )
        )
    notes = [
        f"Monte Carlo with {samples} draws; criticality counts zero total float, "
        "so tied paths are all critical.",
        "Dependence between durations is a Gaussian copula."
        if matrix is not None
        else "Durations are independent; positive correlation between related "
        "activities is typical and widens the completion distribution.",
    ]
    if events:
        notes.append(
            f"{len(events)} discrete risk event(s) added to activity durations."
        )
    pert = pert_estimate(network) if network.has_only_finish_to_start else None
    return ScheduleRiskResult(
        network,
        critical_path(network),
        pert,
        samples,
        tuple(completion),
        tuple(kept),
        mean,
        sd,
        sd / math.sqrt(samples),
        tuple(rows),
        level,
        tuple(notes),
        tuple(sorted(completion)),
    )


@dataclass(frozen=True)
class ScheduleCostSamples:
    """Paired cost and completion-time samples from one simulation.

    Args:
        costs: Total cost per draw, in draw order.
        completions: Completion time of the same draw.
        correlation: Pearson correlation of the two (zero when one is constant).

    References:
        [REF-NASA-CEH-2015].

    """

    costs: tuple[float, ...]
    completions: tuple[float, ...]
    correlation: float

    @property
    def size(self) -> int:
        """Return the number of pairs.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        return len(self.costs)


def _cost_value(
    spec: float | DurationLaw, generator: random.Random, *, name: str
) -> float:
    """Return one draw of a cost or rate given as a number or a law.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No mathematical equation is evaluated.

    """
    if _is_law(spec):
        return float(spec.quantile(generator.random()))  # type: ignore[union-attr]
    return require_finite_real(spec, name=name)


def cost_schedule_samples(
    result: ScheduleRiskResult,
    *,
    fixed_costs: Mapping[str, float | DurationLaw] | None = None,
    burn_rates: Mapping[str, float | DurationLaw] | None = None,
    project_rate: float | DurationLaw = 0.0,
    constant: float = 0.0,
    seed: int | RandomStream | None = None,
) -> ScheduleCostSamples:
    """Load cost onto simulated durations and return paired samples.

    The cost of draw ``k`` is ``constant + sum_i F_i + sum_i r_i d_i^(k) +
    r_p T^(k)``: time-independent activity costs ``F_i``, burn rates ``r_i``
    times the sampled duration ``d_i`` of that draw (risk-event additions
    included), and a project-level rate ``r_p`` times the sampled completion
    time ``T``. A cost or rate given as a law is drawn once per sample,
    independently of the durations and of every other cost.

    Args:
        result: Output of :func:`simulate_schedule`.
        fixed_costs: Time-independent cost per activity name.
        burn_rates: Cost per time unit per activity name.
        project_rate: Project-level cost per time unit (support cost).
        constant: Deterministic cost added to every draw.
        seed: Seed or stream for the cost laws.

    Returns:
        Paired cost and completion samples for
        :func:`joint_cost_schedule_confidence`.

    Raises:
        TypeError: If ``result`` is not a :class:`ScheduleRiskResult`.
        ValueError: If a name is unknown.

    Equation:
        [EQ-COST-LOAD-001].

    References:
        [REF-NASA-CEH-2015], [REF-GAO-16-89G].

    """
    if not isinstance(result, ScheduleRiskResult):
        raise TypeError("result must be a ScheduleRiskResult")
    network = result.network
    fixed = dict(fixed_costs or {})
    burn = dict(burn_rates or {})
    for name in (*fixed, *burn):
        network.index(name)
    base = require_finite_real(constant, name="constant")
    stream = resolve_stream(seed)
    generators = {
        key: stream.child(*key).generator()
        for key in [
            *(("fixed", name) for name in fixed),
            *(("burn", name) for name in burn),
            ("project",),
        ]
    }
    burn_index = {name: network.index(name) for name in burn}
    costs: list[float] = []
    for row, finish in zip(result.durations, result.completion, strict=True):
        total = base
        for name in fixed:
            total += _cost_value(fixed[name], generators[("fixed", name)], name=name)
        for name in burn:
            rate = _cost_value(burn[name], generators[("burn", name)], name=name)
            total += rate * row[burn_index[name]]
        total += (
            _cost_value(project_rate, generators[("project",)], name="project_rate")
            * finish
        )
        costs.append(total)
    try:
        rho = pearson_correlation(costs, result.completion)
    except ValueError:
        rho = 0.0
    return ScheduleCostSamples(tuple(costs), result.completion, rho)


def joint_cost_schedule_confidence(
    samples: ScheduleCostSamples, cost_target: float, deadline: float
) -> JointConfidenceLevel:
    """Return the joint confidence of a (cost, deadline) pair.

    ``P(cost <= C and completion <= D)`` is at most the smaller of the two
    marginal confidences; it equals their product only when cost and
    completion are independent. This hands the paired samples to
    :func:`~probstat.cost.joint_confidence_level`.

    Args:
        samples: Paired samples from :func:`cost_schedule_samples`.
        cost_target: Cost target ``C``.
        deadline: Completion-time target ``D``.

    Returns:
        The joint and marginal confidence levels.

    Equation:
        [EQ-COST-JCL-001].

    References:
        [REF-NASA-CEH-2015].

    """
    if not isinstance(samples, ScheduleCostSamples):
        raise TypeError("samples must be ScheduleCostSamples")
    return joint_confidence_level(
        samples.costs, samples.completions, cost_target, deadline
    )


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One point of the iso-confidence cost-schedule frontier.

    Args:
        deadline: Completion-time target.
        cost: Smallest sampled cost with joint confidence at least the level.
        schedule_confidence: Marginal ``P(completion <= deadline)``.
        cost_confidence: Marginal ``P(cost <= cost)``.

    References:
        [REF-NASA-CEH-2015].

    """

    deadline: float
    cost: float
    schedule_confidence: float
    cost_confidence: float


@dataclass(frozen=True)
class IsoConfidenceFrontier:
    """Cost-schedule pairs that all have the same joint confidence.

    Args:
        level: The joint confidence of every point.
        points: Frontier points by increasing deadline.

    References:
        [REF-NASA-CEH-2015].

    """

    level: float
    points: tuple[FrontierPoint, ...]

    def table(self) -> str:
        """Return the frontier as a plain-text table.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No mathematical equation is evaluated.

        """
        head = ("deadline", "cost", "P(schedule)", "P(cost)")
        rows = [
            (
                _num(p.deadline),
                _num(p.cost),
                f"{p.schedule_confidence:.3f}",
                f"{p.cost_confidence:.3f}",
            )
            for p in self.points
        ]
        return f"joint confidence {self.level:g}\n" + _render(head, rows)


def cost_schedule_frontier(
    samples: ScheduleCostSamples,
    level: float,
    deadlines: Iterable[float] | None = None,
    *,
    points: int = 10,
) -> IsoConfidenceFrontier:
    """Return the iso-confidence frontier of (deadline, cost) pairs.

    For each deadline the cost is the smallest sampled cost whose joint
    confidence with that deadline reaches ``level``
    (:func:`~probstat.cost.cost_for_joint_confidence`); deadlines whose own
    confidence is below ``level`` are infeasible and omitted. Both marginal
    confidences of every point are at least ``level``.

    Args:
        samples: Paired samples.
        level: Required joint confidence in ``(0, 1)``.
        deadlines: Deadlines to evaluate; default are ``points`` completion
            quantiles between ``level`` and 0.99.
        points: Number of default deadlines (at least two).

    Returns:
        The frontier.

    Equation:
        [EQ-COST-JCL-001].

    References:
        [REF-NASA-CEH-2015].

    """
    if not isinstance(samples, ScheduleCostSamples):
        raise TypeError("samples must be ScheduleCostSamples")
    q = require_probability(level, name="level")
    if not 0.0 < q < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    ordered = sorted(samples.completions)
    n = len(ordered)
    if deadlines is None:
        if isinstance(points, bool) or not isinstance(points, int) or points < 2:
            raise ValueError("points must be an integer of at least two")
        top = max(q, 0.99)
        grid = [q + (top - q) * k / (points - 1) for k in range(points)]
        targets = sorted({ordered[min(n, max(1, math.ceil(g * n))) - 1] for g in grid})
    else:
        targets = sorted(require_finite_real(t, name="deadline") for t in deadlines)
    costs_sorted = sorted(samples.costs)
    found: list[FrontierPoint] = []
    for t in targets:
        cost = cost_for_joint_confidence(samples.costs, samples.completions, t, q)
        if cost is None:
            continue
        found.append(
            FrontierPoint(
                t,
                cost,
                sum(1 for c in ordered if c <= t) / n,
                sum(1 for c in costs_sorted if c <= cost) / n,
            )
        )
    return IsoConfidenceFrontier(q, tuple(found))
