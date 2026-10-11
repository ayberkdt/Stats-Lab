"""Sensitivity analysis that runs directly on a ProbabilisticModel.

The estimators reuse the pick-freeze formulas of
:func:`probstat.sensitivity.estimate_sobol_indices` but evaluate the model
exactly once per design row and keep every output, so bootstrap confidence
intervals cost no further model evaluations. Inputs are drawn from the
model's joint law through a unit-hypercube design (Sobol, Halton, LHS, MC).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import ProbabilisticModel
from probstat.propagation import make_design
from probstat.sampling import SOBOL_MAX_DIMENSION
from probstat.statistics import quantile

from .analysis import SobolResult


@dataclass(frozen=True, slots=True)
class IndexInterval:
    """Point estimate and bootstrap percentile interval for one index.

    Args:
        estimate: Index estimated from the full design.
        lower: Lower percentile bound (``nan`` without bootstrap).
        upper: Upper percentile bound (``nan`` without bootstrap).

    References:
        [REF-ARCHER-SALTELLI-SOBOL-1997], [REF-EFRON-TIBSHIRANI-1993].

    """

    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ModelSobolResult:
    """Sobol indices of one QoI of a probabilistic model, with intervals.

    Args:
        quantity: QoI name.
        input_names: Input names in model order.
        first_order: First-order index intervals.
        total_effect: Total-effect index intervals.
        result: Plain :class:`SobolResult` (second-order entries are zero
            unless requested).
        design: Unit-hypercube design name.
        bootstrap: Number of bootstrap replicates (0 if none).
        confidence: Nominal bootstrap interval level.

    References:
        [REF-SALTELLI-2010], [REF-ARCHER-SALTELLI-SOBOL-1997].

    """

    quantity: str
    input_names: tuple[str, ...]
    first_order: tuple[IndexInterval, ...]
    total_effect: tuple[IndexInterval, ...]
    result: SobolResult
    design: str
    bootstrap: int
    confidence: float

    @property
    def evaluations(self) -> int:
        """Return the number of forward-model evaluations.

        References:
            [REF-SALTELLI-2010].

        """
        return self.result.evaluations

    def as_mapping(self) -> Mapping[str, tuple[float, float]]:
        """Return ``input -> (first-order, total-effect)`` point estimates.

        References:
            [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [REF-SALTELLI-2010].

        """
        return MappingProxyType(
            {
                name: (first.estimate, total.estimate)
                for name, first, total in zip(
                    self.input_names, self.first_order, self.total_effect, strict=True
                )
            }
        )

    def ranking(self) -> tuple[str, ...]:
        """Return input names in descending total-effect order.

        References:
            [EQ-SENS-TOTAL-001], [REF-SALTELLI-2010].

        """
        return tuple(self.input_names[index] for index in self.result.ranking())

    def negligible(self, threshold: float = 0.01) -> tuple[str, ...]:
        """Return inputs whose total-effect upper bound is below ``threshold``.

        Without bootstrap the point estimate is used. Fixing such inputs at a
        nominal value removes at most about ``threshold`` of output variance
        each (factor fixing).

        Args:
            threshold: Total-effect screening threshold.

        Returns:
            Names of inputs that can be fixed.

        Equation:
            [EQ-SENS-DECISION-001].

        References:
            [REF-SALTELLI-2008, Chapter 1].

        """
        names: list[str] = []
        for name, total in zip(self.input_names, self.total_effect, strict=True):
            bound = total.upper if math.isfinite(total.upper) else total.estimate
            if bound < threshold:
                names.append(name)
        return tuple(names)


def _indices(
    ya: Sequence[float],
    yb: Sequence[float],
    yab: Sequence[Sequence[float]],
    rows: Sequence[int],
) -> tuple[list[float], list[float], float]:
    """Compute first and total indices on a subset of design rows.

    Args:
        ya: Outputs on matrix A.
        yb: Outputs on matrix B.
        yab: Outputs on A with column i taken from B, per input.
        rows: Row indices (with repetition for bootstrap).

    Returns:
        First-order list, total-effect list, and pooled output variance.

    Raises:
        ValueError: If the pooled variance is zero.

    Equation:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    References:
        [REF-SALTELLI-2010], [REF-JANSEN-1999].

    """
    n = len(rows)
    pooled = [ya[k] for k in rows] + [yb[k] for k in rows]
    mean = math.fsum(pooled) / len(pooled)
    variance = math.fsum((value - mean) ** 2 for value in pooled) / len(pooled)
    if variance <= 0.0:
        raise ValueError("Sobol indices require positive output variance")
    first: list[float] = []
    total: list[float] = []
    for column in yab:
        first.append(
            math.fsum(yb[k] * (column[k] - ya[k]) for k in rows) / n / variance
        )
        total.append(
            0.5 * math.fsum((ya[k] - column[k]) ** 2 for k in rows) / n / variance
        )
    return first, total, variance


def sobol_analysis(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    design: str = "sobol",
    seed: int | RandomStream | None = None,
    bootstrap: int = 0,
    confidence: float = 0.95,
) -> ModelSobolResult:
    """Estimate first-order and total Sobol indices of a model QoI.

    The A and B matrices are the two halves of one ``2d``-dimensional design,
    so quasi-random designs remain low-discrepancy across both. Cost is
    ``size (d + 2)`` model evaluations; bootstrap intervals reuse them.

    Args:
        model: Probabilistic model with mutually independent inputs.
        quantity: QoI name.
        size: Number of design rows (powers of two suit Sobol designs).
        design: ``"sobol"``, ``"halton"``, ``"lhs"``, or ``"mc"``; Sobol falls
            back to scrambled Halton when ``2d`` exceeds its dimension limit.
        seed: Root seed or stream.
        bootstrap: Number of bootstrap replicates (0 disables intervals).
        confidence: Nominal percentile-interval level.

    Returns:
        Indices with optional bootstrap intervals and evaluation count.

    Raises:
        ValueError: If inputs are dependent, ``size < 2``, or the QoI has zero
            variance.

    Equation:
        [EQ-SENS-ANOVA-001], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001],
        [EQ-SENS-BOOT-001].

    References:
        [REF-SALTELLI-2010], [REF-JANSEN-1999], [REF-ARCHER-SALTELLI-SOBOL-1997].

    """
    if not model.inputs.is_independent:
        raise ValueError(
            "variance-based Sobol indices require independent inputs; "
            "the model's joint law has a non-identity copula"
        )
    model.quantity(quantity)
    if size < 2:
        raise ValueError("size must be at least two")
    if bootstrap < 0:
        raise ValueError("bootstrap must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    dimension = model.dimension
    root = resolve_stream(seed)
    method = design
    if design == "sobol" and 2 * dimension > SOBOL_MAX_DIMENSION:
        method = "halton"
    unit = make_design(method, size, 2 * dimension, seed=root.child("sobol-design"))
    a_rows = tuple(point[:dimension] for point in unit.points)
    b_rows = tuple(point[dimension:] for point in unit.points)
    function = model.unit_function(quantity)
    before = model.evaluations
    ya = tuple(function(row) for row in a_rows)
    yb = tuple(function(row) for row in b_rows)
    yab = tuple(
        tuple(
            function(a_row[:i] + (b_row[i],) + a_row[i + 1 :])
            for a_row, b_row in zip(a_rows, b_rows, strict=True)
        )
        for i in range(dimension)
    )
    evaluations = model.evaluations - before
    all_rows = tuple(range(size))
    first, total, variance = _indices(ya, yb, yab, all_rows)
    first_intervals: list[IndexInterval] = []
    total_intervals: list[IndexInterval] = []
    if bootstrap:
        generator = root.child("sobol-bootstrap").generator()
        first_draws: list[list[float]] = [[] for _ in range(dimension)]
        total_draws: list[list[float]] = [[] for _ in range(dimension)]
        for _ in range(bootstrap):
            rows = [generator.randrange(size) for _ in range(size)]
            try:
                replicate_first, replicate_total, _ = _indices(ya, yb, yab, rows)
            except ValueError:
                continue
            for i in range(dimension):
                first_draws[i].append(replicate_first[i])
                total_draws[i].append(replicate_total[i])
        tail = 0.5 * (1.0 - confidence)
        for i in range(dimension):
            first_intervals.append(
                IndexInterval(
                    first[i],
                    quantile(first_draws[i], tail),
                    quantile(first_draws[i], 1.0 - tail),
                )
            )
            total_intervals.append(
                IndexInterval(
                    total[i],
                    quantile(total_draws[i], tail),
                    quantile(total_draws[i], 1.0 - tail),
                )
            )
    else:
        first_intervals = [IndexInterval(v, math.nan, math.nan) for v in first]
        total_intervals = [IndexInterval(v, math.nan, math.nan) for v in total]
    result = SobolResult(
        tuple(first),
        tuple(total),
        tuple(tuple(0.0 for _ in range(dimension)) for _ in range(dimension)),
        variance,
        size,
        evaluations,
    )
    return ModelSobolResult(
        quantity,
        model.inputs.names,
        tuple(first_intervals),
        tuple(total_intervals),
        result,
        method,
        bootstrap,
        confidence,
    )


@dataclass(frozen=True, slots=True)
class MorrisResult:
    """Elementary-effect screening summary.

    Args:
        input_names: Input names in model order.
        mu_star: Mean absolute elementary effect per input.
        sigma: Standard deviation of elementary effects per input.
        trajectories: Number of one-at-a-time trajectories.
        evaluations: Forward-model evaluations, ``r (d + 1)``.

    References:
        [REF-MORRIS-1991], [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007].

    """

    input_names: tuple[str, ...]
    mu_star: tuple[float, ...]
    sigma: tuple[float, ...]
    trajectories: int
    evaluations: int

    def ranking(self) -> tuple[str, ...]:
        """Return input names in descending ``mu*`` order.

        References:
            [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007].

        """
        order = sorted(range(len(self.mu_star)), key=lambda i: -self.mu_star[i])
        return tuple(self.input_names[i] for i in order)


def morris_screening(
    model: ProbabilisticModel,
    quantity: str,
    trajectories: int,
    *,
    levels: int = 4,
    seed: int | RandomStream | None = None,
) -> MorrisResult:
    """Screen inputs with Morris elementary effects in the unit hypercube.

    Each trajectory starts at a random grid point and changes one input at a
    time by ``Delta = p / (2 (p - 1))`` in random order and direction. Effects
    are measured on the unit (probability) scale of each input, so ``mu*``
    ranks inputs regardless of their physical units.

    Caveat: grid levels are mapped into the open unit interval, so one step
    moves an input by exactly half of its quantile range. A response that is
    periodic with that period in an input (for example ``sin(x)**2`` on
    ``[-pi, pi]``) shows zero elementary effects for it; cross-check with
    :func:`sobol_analysis`.

    Args:
        model: Probabilistic model with independent inputs.
        quantity: QoI name.
        trajectories: Number of trajectories ``r`` (at least two).
        levels: Even number of grid levels ``p`` (at least two).
        seed: Root seed or stream.

    Returns:
        ``mu*`` and ``sigma`` per input with evaluation count ``r (d + 1)``.

    Raises:
        ValueError: If arguments are invalid or inputs are dependent.

    Equation:
        [EQ-SENS-MORRIS-001].

    References:
        [REF-MORRIS-1991], [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007].

    """
    if not model.inputs.is_independent:
        raise ValueError("Morris screening here assumes independent inputs")
    if trajectories < 2:
        raise ValueError("trajectories must be at least two")
    if levels < 2 or levels % 2:
        raise ValueError("levels must be an even integer of at least two")
    function = model.unit_function(quantity)
    dimension = model.dimension
    delta = levels / (2.0 * (levels - 1))
    grid = [k / (levels - 1) for k in range(levels)]
    generator = resolve_stream(seed).child("morris").generator()
    effects: list[list[float]] = [[] for _ in range(dimension)]
    before = model.evaluations

    def to_open(value: float) -> float:
        """Map a grid level in ``[0, 1]`` into the open unit interval.

        References:
            [REF-MORRIS-1991]. No equation is evaluated.

        """
        margin = 0.5 / levels
        return margin + (1.0 - 2.0 * margin) * value

    for _ in range(trajectories):
        start = [
            generator.choice([g for g in grid if g + delta <= 1.0 + 1e-12])
            for _ in range(dimension)
        ]
        point = list(start)
        current = function(tuple(to_open(v) for v in point))
        for index in generator.sample(range(dimension), dimension):
            step = delta if point[index] + delta <= 1.0 + 1e-12 else -delta
            if generator.random() < 0.5 and point[index] - delta >= -1e-12:
                step = -delta
            point[index] += step
            following = function(tuple(to_open(v) for v in point))
            scaled_step = step * (1.0 - 1.0 / levels)
            effects[index].append((following - current) / scaled_step)
            current = following
    mu_star = tuple(
        math.fsum(abs(e) for e in column) / len(column) for column in effects
    )
    sigma = tuple(
        math.sqrt(
            math.fsum((e - math.fsum(column) / len(column)) ** 2 for e in column)
            / (len(column) - 1)
        )
        for column in effects
    )
    return MorrisResult(
        model.inputs.names, mu_star, sigma, trajectories, model.evaluations - before
    )
