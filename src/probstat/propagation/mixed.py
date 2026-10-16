"""Nested (double-loop) propagation of mixed aleatory and epistemic uncertainty.

Aleatory uncertainty (variability) and epistemic uncertainty (lack of
knowledge) are kept apart, as in the framework of Roy and Oberkampf (2011):

- the **outer loop** visits epistemic values: inputs marked
  ``nature=EPISTEMIC`` are sampled from their (second-order) distribution,
  and :class:`~probstat.modeling.IntervalInput` quantities are explored by a
  design over their box (all corners plus space-filling points, optionally
  refined by a bounded compass search);
- the **inner loop** propagates the aleatory inputs, conditional on the
  outer values, and records an output CDF.

The result is a family of output CDFs, the probability box (pointwise lower
and upper CDF envelopes) and, for each requested statistic (mean, exceedance
probability, quantile), its values over the outer points with the smallest and
largest observed value. Envelopes over a finite design are *inner*
approximations of the true bounds: the true extremes may lie between the
design points (exact at the corners when the response is monotone in each
interval input), and inner-loop sampling error adds noise. With common random
numbers (the default) every outer point reuses the same inner design, so
differences between outer points are not masked by sampling noise.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Vector
from probstat.modeling import IntervalInput, JointInput, ProbabilisticModel
from probstat.statistics import quantile as sample_quantile

from .study import DESIGNS, make_design

_CORNER_LIMIT = 12


@dataclass(frozen=True, slots=True)
class EpistemicStatistic:
    """One output statistic as a function of the epistemic values.

    Args:
        name: Statistic label (``"mean"``, ``"P(Y>t)"``, ``"q0.95"``).
        values: Statistic at every outer point, in outer-point order.
        lower: Smallest observed value (an inner estimate of the lower bound).
        upper: Largest observed value (an inner estimate of the upper bound).
        argmin: Outer-point index attaining ``lower``.
        argmax: Outer-point index attaining ``upper``.

    References:
        [REF-OBERKAMPF-ROY-2010], [REF-ROY-OBERKAMPF-2011].

    """

    name: str
    values: tuple[float, ...]
    lower: float
    upper: float
    argmin: int
    argmax: int

    @property
    def width(self) -> float:
        """Return ``upper - lower``.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        return self.upper - self.lower

    def quantile(self, probability: float) -> float:
        """Return a type-7 quantile of the statistic over the outer points.

        Meaningful as an epistemic (second-order) distribution only when the
        outer points are a sample from epistemic distributions, not a design
        over intervals.

        Args:
            probability: Level in ``[0, 1]``.

        Returns:
            Quantile of :attr:`values`.

        Equation:
            [EQ-STAT-QUANT-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        return sample_quantile(self.values, probability)


@dataclass(frozen=True, slots=True)
class ProbabilityBox:
    """Lower and upper CDF envelopes of a family of empirical output CDFs.

    ``lower_cdf(y) = min_e F_e(y)`` and ``upper_cdf(y) = max_e F_e(y)``
    over the outer points ``e``, tabulated on ``levels``; :meth:`bounds`
    evaluates them exactly at any ``y``.

    Args:
        levels: Increasing output levels spanning every inner output.
        lower_cdf: Lower envelope at each level.
        upper_cdf: Upper envelope at each level.
        sorted_outputs: Sorted inner outputs of every outer point.

    References:
        [REF-FERSON-ET-AL-2003], [REF-ROY-OBERKAMPF-2011].

    """

    levels: tuple[float, ...]
    lower_cdf: tuple[float, ...]
    upper_cdf: tuple[float, ...]
    sorted_outputs: tuple[tuple[float, ...], ...]

    def bounds(self, y: float) -> tuple[float, float]:
        """Return ``(min_e F_e(y), max_e F_e(y))`` from the empirical CDFs.

        Args:
            y: Finite output level.

        Returns:
            Lower and upper CDF bounds at ``y``.

        Equation:
            [EQ-PROP-PBOX-001].

        References:
            [REF-FERSON-ET-AL-2003].

        """
        level = require_finite_real(y, name="y")
        values = [
            bisect.bisect_right(outputs, level) / len(outputs)
            for outputs in self.sorted_outputs
        ]
        return min(values), max(values)

    def exceedance_bounds(self, threshold: float) -> tuple[float, float]:
        """Return the bounds on ``P(Y > t)`` implied by the box.

        Args:
            threshold: Finite level ``t``.

        Returns:
            ``(1 - upper_cdf(t), 1 - lower_cdf(t))``.

        Equation:
            [EQ-PROP-PBOX-001].

        References:
            [REF-FERSON-ET-AL-2003].

        """
        low, high = self.bounds(threshold)
        return 1.0 - high, 1.0 - low


@dataclass(frozen=True, slots=True)
class MixedPropagationResult:
    """Outcome of a nested aleatory-epistemic propagation.

    Args:
        quantity: Propagated QoI name.
        epistemic_names: Epistemic input and interval names.
        aleatory_names: Aleatory input names.
        epistemic_points: Epistemic values of every outer point.
        point_kinds: ``"corner"``, ``"design"``, ``"sample"``, or ``"refine"``.
        outputs: Inner-loop outputs of every outer point, in inner order.
        pbox: Probability box over all outer points.
        statistics: Statistic name to :class:`EpistemicStatistic`.
        inner_size: Inner-loop sample size.
        inner_design: Inner design name.
        common_random_numbers: Whether all outer points share one inner design.
        model_evaluations: Forward-model calls charged to the model.

    References:
        [REF-OBERKAMPF-ROY-2010], [REF-ROY-OBERKAMPF-2011].

    """

    quantity: str
    epistemic_names: tuple[str, ...]
    aleatory_names: tuple[str, ...]
    epistemic_points: tuple[Mapping[str, float], ...]
    point_kinds: tuple[str, ...]
    outputs: tuple[tuple[float, ...], ...]
    pbox: ProbabilityBox
    statistics: Mapping[str, EpistemicStatistic]
    inner_size: int
    inner_design: str
    common_random_numbers: bool
    model_evaluations: int

    @property
    def outer_size(self) -> int:
        """Return the number of outer points (including refinement points).

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        return len(self.epistemic_points)

    def cdf(self, index: int, y: float) -> float:
        """Return the empirical output CDF of one outer point.

        Args:
            index: Outer-point index.
            y: Finite output level.

        Returns:
            Fraction of inner outputs ``<= y``.

        Equation:
            [EQ-STAT-005].

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        outputs = self.pbox.sorted_outputs[index]
        level = require_finite_real(y, name="y")
        return bisect.bisect_right(outputs, level) / len(outputs)


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer of at least ``minimum``.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is below ``minimum``.

    References:
        Project validation policy. No equation is evaluated.

    """
    count = require_non_negative_integer(value, name=name)
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def _split_inputs(
    joint: JointInput,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split input names into aleatory and epistemic groups.

    Raises:
        ValueError: If the two groups are dependent (non-zero Gaussian-copula
            cross correlation, or any non-Gaussian copula linking them).

    References:
        [REF-OBERKAMPF-ROY-2010]. No equation is evaluated.

    """
    aleatory = tuple(item.name for item in joint.inputs if not item.is_epistemic)
    epistemic = tuple(item.name for item in joint.inputs if item.is_epistemic)
    if aleatory and epistemic and not joint.is_independent:
        if joint.correlation is None:
            raise ValueError(
                "aleatory and epistemic inputs must be independent; a "
                f"{joint.copula_family} copula couples them"
            )
        index = {name: k for k, name in enumerate(joint.names)}
        for a in aleatory:
            for e in epistemic:
                if joint.correlation[index[a]][index[e]] != 0.0:
                    raise ValueError(
                        "aleatory and epistemic inputs must be independent; "
                        f"{a!r} and {e!r} are correlated"
                    )
    return aleatory, epistemic


def _statistic_names(
    thresholds: Sequence[float], probabilities: Sequence[float]
) -> tuple[str, ...]:
    """Return labels of the requested statistics.

    References:
        Project validation policy. No equation is evaluated.

    """
    return (
        ("mean",)
        + tuple(f"P(Y>{t:g})" for t in thresholds)
        + tuple(f"q{p:g}" for p in probabilities)
    )


def _statistics_of(
    outputs: Sequence[float],
    thresholds: Sequence[float],
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    """Return the mean, exceedance fractions, and type-7 quantiles.

    Equation:
        [EQ-MC-MEAN-001], [EQ-PROP-THRESH-001], [EQ-STAT-QUANT-001].

    References:
        [REF-SMITH-UQ-2024, Chapter 4], [REF-HYNDMAN-FAN-1996].

    """
    count = len(outputs)
    mean = math.fsum(outputs) / count
    exceedances = tuple(
        sum(1 for value in outputs if value > t) / count for t in thresholds
    )
    quantiles = tuple(sample_quantile(outputs, p) for p in probabilities)
    return (mean,) + exceedances + quantiles


def nested_propagation(
    model: ProbabilisticModel,
    quantity: str,
    *,
    intervals: Sequence[IntervalInput] = (),
    outer_size: int = 20,
    inner_size: int = 1000,
    inner_design: str = "lhs",
    corners: bool = True,
    thresholds: Sequence[float] = (),
    probabilities: Sequence[float] = (0.05, 0.5, 0.95),
    refine: int = 0,
    common_random_numbers: bool = True,
    levels: int = 200,
    seed: int | RandomStream | None = None,
) -> MixedPropagationResult:
    """Propagate aleatory inputs inside a loop over epistemic values.

    Outer points: a Latin hypercube of ``outer_size`` points over the
    epistemic dimensions (epistemic random inputs through their joint
    ``from_unit``, intervals linearly), on the child stream ``("outer",)``;
    when ``corners`` is true and intervals exist, all ``2^k`` corners of the
    interval box are added (epistemic random inputs at their median). With
    ``refine > 0`` a bounded compass search over the interval box spends up
    to ``refine`` extra outer points on each bound (lower and upper) of each
    statistic, starting from the best outer point so far; epistemic random
    inputs stay at that point's values.

    Inner loop at outer point ``i``: the aleatory sub-law (exact margin of the
    joint law) is sampled with ``inner_design`` on the child stream
    ``("inner", i)``, or ``("inner", 0)`` for every point when
    ``common_random_numbers`` is true; the epistemic values are passed to the
    forward model as fixed parameters. Aleatory and epistemic inputs must be
    independent.

    Args:
        model: Probabilistic model; inputs with ``nature=EPISTEMIC`` form the
            outer loop together with ``intervals``.
        quantity: QoI name.
        intervals: Interval-valued epistemic quantities passed to the model by
            name (they must not clash with inputs or fixed parameters).
        outer_size: Space-filling outer points (zero allowed with corners).
        inner_size: Inner-loop sample size, at least two.
        inner_design: Inner design name (see :data:`DESIGNS`).
        corners: Whether to add the corners of the interval box.
        thresholds: Levels ``t`` of the exceedance statistics ``P(Y > t)``.
        probabilities: Levels of the quantile statistics.
        refine: Extra outer points per statistic bound for the compass search.
        common_random_numbers: Whether all outer points reuse one inner design.
        levels: Number of tabulated p-box levels, at least two.
        seed: Root seed or stream.

    Returns:
        CDF family, probability box, statistic intervals, and the cost.

    Raises:
        TypeError: If arguments have the wrong type.
        KeyError: If the quantity is unknown.
        ValueError: If there is no aleatory or no epistemic input, names
            clash, the groups are dependent, or settings are invalid.

    Equation:
        [EQ-PROP-MIXED-001], [EQ-PROP-PBOX-001].

    References:
        [REF-OBERKAMPF-ROY-2010], [REF-ROY-OBERKAMPF-2011],
        [REF-FERSON-ET-AL-2003], [REF-MCKAY-BECKMAN-CONOVER-1979].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    model.quantity(quantity)
    boxes = tuple(intervals)
    for index, item in enumerate(boxes):
        if not isinstance(item, IntervalInput):
            raise TypeError(f"intervals[{index}] must be an IntervalInput")
    aleatory, epistemic = _split_inputs(model.inputs)
    if not aleatory:
        raise ValueError("nested propagation needs at least one aleatory input")
    interval_names = tuple(item.name for item in boxes)
    if len(set(interval_names)) != len(interval_names):
        raise ValueError("interval names must be distinct")
    clash = set(interval_names) & (set(model.inputs.names) | set(model.fixed))
    if clash:
        raise ValueError(f"interval names clash with model inputs: {sorted(clash)}")
    if not epistemic and not boxes:
        raise ValueError("nested propagation needs an epistemic input or interval")
    n_outer = _require_count(outer_size, "outer_size", 0)
    n_inner = _require_count(inner_size, "inner_size", 2)
    budget = _require_count(refine, "refine", 0)
    n_levels = _require_count(levels, "levels", 2)
    if inner_design not in DESIGNS:
        raise ValueError(f"inner_design must be one of {DESIGNS}")
    if not isinstance(corners, bool) or not isinstance(common_random_numbers, bool):
        raise TypeError("corners and common_random_numbers must be booleans")
    levels_t = tuple(require_finite_real(t, name="threshold") for t in thresholds)
    levels_p = tuple(require_finite_real(p, name="probability") for p in probabilities)
    if any(not 0.0 <= p <= 1.0 for p in levels_p):
        raise ValueError("probabilities must lie in [0, 1]")
    if corners and len(boxes) > _CORNER_LIMIT:
        raise ValueError(f"corners are limited to {_CORNER_LIMIT} intervals")
    if n_outer == 0 and not (corners and boxes):
        raise ValueError("outer_size must be positive unless interval corners are used")
    root = resolve_stream(seed)
    aleatory_law = model.inputs.subset(aleatory)
    epistemic_law = model.inputs.subset(epistemic) if epistemic else None
    names = epistemic + interval_names
    points: list[dict[str, float]] = []
    kinds: list[str] = []

    def epistemic_values(unit: Sequence[float]) -> dict[str, float]:
        """Map an outer unit point to named epistemic values.

        References:
            [REF-MCKAY-BECKMAN-CONOVER-1979]. No new equation.

        """
        values: dict[str, float] = {}
        r = len(epistemic)
        if epistemic_law is not None:
            values.update(
                zip(epistemic, epistemic_law.from_unit(unit[:r]), strict=True)
            )
        for item, t in zip(boxes, unit[r:], strict=True):
            values[item.name] = item.from_unit(t)
        return values

    if n_outer:
        design = make_design("lhs", n_outer, len(names), seed=root.child("outer"))
        for point in design.points:
            points.append(epistemic_values(point))
            kinds.append("design" if boxes else "sample")
    if corners and boxes:
        median = (
            dict(
                zip(
                    epistemic,
                    epistemic_law.from_unit((0.5,) * len(epistemic)),
                    strict=True,
                )
            )
            if epistemic_law is not None
            else {}
        )
        for mask in range(1 << len(boxes)):
            values = dict(median)
            for k, item in enumerate(boxes):
                values[item.name] = item.upper if mask >> k & 1 else item.lower
            points.append(values)
            kinds.append("corner")
    before = model.evaluations
    shared: tuple[Vector, ...] | None = None

    def inner_inputs(index: int) -> tuple[Vector, ...]:
        """Return the aleatory inner sample of an outer point.

        References:
            [REF-OWEN-2013, Chapter 10]. No new equation.

        """
        nonlocal shared
        if common_random_numbers and shared is not None:
            return shared
        stream = root.child("inner", 0 if common_random_numbers else index)
        sample = aleatory_law.sample_design(
            make_design(inner_design, n_inner, aleatory_law.dimension, seed=stream)
        )
        if common_random_numbers:
            shared = sample
        return sample

    outputs: list[tuple[float, ...]] = []
    table: list[tuple[float, ...]] = []

    def run(values: Mapping[str, float]) -> tuple[float, ...]:
        """Run the inner loop at one epistemic point; return its statistics.

        Equation:
            [EQ-PROP-MIXED-001].

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        index = len(outputs)
        conditional = replace(
            model, inputs=aleatory_law, fixed={**dict(model.fixed), **values}
        )
        result = tuple(conditional.evaluate(x)[quantity] for x in inner_inputs(index))
        outputs.append(result)
        statistics = _statistics_of(result, levels_t, levels_p)
        table.append(statistics)
        return statistics

    for values in points:
        run(values)
    labels = _statistic_names(levels_t, levels_p)
    if budget and boxes:
        _refine_bounds(points, kinds, table, run, boxes, len(labels), budget)
    sorted_outputs = tuple(tuple(sorted(result)) for result in outputs)
    low = min(values[0] for values in sorted_outputs)
    high = max(values[-1] for values in sorted_outputs)
    grid = tuple(low + (high - low) * k / (n_levels - 1) for k in range(n_levels))
    lower_cdf: list[float] = []
    upper_cdf: list[float] = []
    for level in grid:
        cdfs = [
            bisect.bisect_right(values, level) / len(values)
            for values in sorted_outputs
        ]
        lower_cdf.append(min(cdfs))
        upper_cdf.append(max(cdfs))
    summaries: dict[str, EpistemicStatistic] = {}
    for column, label in enumerate(labels):
        column_values = tuple(row[column] for row in table)
        argmin = min(range(len(column_values)), key=lambda k: column_values[k])
        argmax = max(range(len(column_values)), key=lambda k: column_values[k])
        summaries[label] = EpistemicStatistic(
            label,
            column_values,
            column_values[argmin],
            column_values[argmax],
            argmin,
            argmax,
        )
    return MixedPropagationResult(
        quantity=quantity,
        epistemic_names=names,
        aleatory_names=aleatory,
        epistemic_points=tuple(MappingProxyType(dict(p)) for p in points),
        point_kinds=tuple(kinds),
        outputs=tuple(outputs),
        pbox=ProbabilityBox(grid, tuple(lower_cdf), tuple(upper_cdf), sorted_outputs),
        statistics=MappingProxyType(summaries),
        inner_size=n_inner,
        inner_design=inner_design,
        common_random_numbers=common_random_numbers,
        model_evaluations=model.evaluations - before,
    )


def _refine_bounds(
    points: list[dict[str, float]],
    kinds: list[str],
    table: list[tuple[float, ...]],
    run: Callable[[Mapping[str, float]], tuple[float, ...]],
    boxes: tuple[IntervalInput, ...],
    statistic_count: int,
    budget: int,
) -> None:
    """Refine each statistic's bounds by a compass search over the interval box.

    For each statistic and direction, start at the best outer point, try
    ``+-step`` along every interval coordinate (normalized to ``[0, 1]``,
    clipped to the box), move on improvement, halve the step otherwise, and
    stop after ``budget`` new inner loops or when the step falls below
    ``1e-3``. Points are appended to the outer list with kind ``"refine"``.

    Equation:
        [EQ-PROP-MIXED-001].

    References:
        [REF-KOLDA-LEWIS-TORCZON-2003].

    """
    names = tuple(item.name for item in boxes)

    def unit_of(values: Mapping[str, float]) -> list[float]:
        """Return normalized interval coordinates of an outer point.

        References:
            [REF-KOLDA-LEWIS-TORCZON-2003]. No equation is evaluated.

        """
        return [
            0.0 if item.width == 0.0 else (values[item.name] - item.lower) / item.width
            for item in boxes
        ]

    for column in range(statistic_count):
        for sign in (-1.0, 1.0):
            best = max(range(len(table)), key=lambda k: sign * table[k][column])
            base = dict(points[best])
            current = unit_of(base)
            score = sign * table[best][column]
            step = 0.25
            spent = 0
            while spent < budget and step >= 1e-3:
                improved = False
                for k in range(len(names)):
                    for direction in (1.0, -1.0):
                        if spent >= budget:
                            break
                        trial = list(current)
                        trial[k] = min(1.0, max(0.0, trial[k] + direction * step))
                        if trial[k] == current[k]:
                            continue
                        values = dict(base)
                        for item, t in zip(boxes, trial, strict=True):
                            values[item.name] = item.from_unit(t)
                        statistics = run(values)
                        points.append(values)
                        kinds.append("refine")
                        spent += 1
                        if sign * statistics[column] > score:
                            score = sign * statistics[column]
                            current = trial
                            base = values
                            improved = True
                            break
                    if improved or spent >= budget:
                        break
                if not improved:
                    step *= 0.5


__all__ = [
    "EpistemicStatistic",
    "MixedPropagationResult",
    "ProbabilityBox",
    "nested_propagation",
]
