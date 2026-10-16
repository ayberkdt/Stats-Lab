"""Sensitivity analysis that runs directly on a ProbabilisticModel.

The estimators reuse the pick-freeze formulas of
:func:`probstat.sensitivity.estimate_sobol_indices` but evaluate the model
exactly once per design row and keep every output of every QoI, so several
quantities share one design (:func:`sobol_analysis_all`) and error bars cost
no further model evaluations. Inputs are drawn from the model's joint law
through a unit-hypercube design (Sobol, Halton, LHS, MC).

Error bars depend on the design: rows of a plain Monte Carlo design are
i.i.d. and may be bootstrapped; rows of quasi-random and Latin-hypercube
designs are not, so their intervals come from independent randomizations
(replicates) of the design instead (ADR-0065).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.distributions import StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import ProbabilisticModel
from probstat.propagation import DESIGNS, make_design
from probstat.sampling import SOBOL_MAX_DIMENSION
from probstat.statistics import quantile

from .analysis import SobolResult

DEFAULT_REPLICATES = 8


@dataclass(frozen=True, slots=True)
class IndexInterval:
    """Point estimate and confidence interval for one index.

    The interval is a bootstrap percentile interval (plain Monte Carlo
    designs) or a Student-t interval over independent randomizations of a
    quasi-random design; ``ModelSobolResult.interval_method`` says which.

    Args:
        estimate: Index estimate (mean of the replicates for replicate
            intervals).
        lower: Lower bound (``nan`` without an interval).
        upper: Upper bound (``nan`` without an interval).

    References:
        [REF-ARCHER-SALTELLI-SOBOL-1997], [REF-EFRON-TIBSHIRANI-1993],
        [REF-L-ECUYER-2018].

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
        result: Plain :class:`SobolResult`; ``second_order`` is ``None``
            because second-order indices are not estimated here.
        design: Unit-hypercube design name actually used.
        bootstrap: Number of row-bootstrap resamples (0 unless
            ``interval_method == "bootstrap"``).
        confidence: Nominal interval level.
        interval_method: ``"bootstrap"`` (i.i.d. rows of a Monte Carlo
            design), ``"replicates"`` (independent randomizations), or
            ``"none"``.
        replicates: Number of independent randomizations (0 unless
            ``interval_method == "replicates"``).

    References:
        [REF-SALTELLI-2010], [REF-ARCHER-SALTELLI-SOBOL-1997],
        [REF-L-ECUYER-2018].

    """

    quantity: str
    input_names: tuple[str, ...]
    first_order: tuple[IndexInterval, ...]
    total_effect: tuple[IndexInterval, ...]
    result: SobolResult
    design: str
    bootstrap: int
    confidence: float
    interval_method: str = "none"
    replicates: int = 0

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

        Without an interval the point estimate is used. Fixing such inputs at a
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


@dataclass(frozen=True, slots=True)
class MultiQuantitySobolResult:
    """Sobol indices of several QoIs computed from one shared design.

    Args:
        results: ``QoI name -> ModelSobolResult``. Each entry's
            ``evaluations`` is the *shared* total, not an additive share.
        evaluations: Model evaluations of the whole analysis.

    References:
        [REF-SALTELLI-2010].

    """

    results: Mapping[str, ModelSobolResult]
    evaluations: int

    def __getitem__(self, quantity: str) -> ModelSobolResult:
        """Return the result of one QoI.

        Raises:
            KeyError: If the QoI was not analysed.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.results[quantity]

    @property
    def quantities(self) -> tuple[str, ...]:
        """Return the analysed QoI names in order.

        References:
            Project API policy. No equation is evaluated.

        """
        return tuple(self.results)


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


_PickFreeze = tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    dict[str, tuple[tuple[float, ...], ...]],
]


def _pick_freeze_outputs(
    model: ProbabilisticModel,
    names: Sequence[str],
    method: str,
    size: int,
    stream: RandomStream,
) -> _PickFreeze:
    """Evaluate one pick-freeze design and keep every requested QoI.

    The A and B matrices are the two halves of one ``2d``-dimensional unit
    design; each row costs one model call that returns every QoI, so all
    quantities share the ``size (d + 2)`` evaluations.

    References:
        [REF-SALTELLI-2010]. Equation [EQ-SENS-FIRST-001].

    """
    dimension = model.dimension
    unit = make_design(method, size, 2 * dimension, seed=stream)
    a_rows = tuple(point[:dimension] for point in unit.points)
    b_rows = tuple(point[dimension:] for point in unit.points)

    def outputs(row: Sequence[float]) -> Mapping[str, float]:
        """Evaluate every QoI at one unit-hypercube row.

        References:
            [EQ-SAMP-INV-001], [REF-SALTELLI-2010].

        """
        return model.evaluate(model.inputs.from_unit(row))

    out_a = [outputs(row) for row in a_rows]
    out_b = [outputs(row) for row in b_rows]
    out_ab = [
        [
            outputs(a_row[:i] + (b_row[i],) + a_row[i + 1 :])
            for a_row, b_row in zip(a_rows, b_rows, strict=True)
        ]
        for i in range(dimension)
    ]
    ya = {name: tuple(o[name] for o in out_a) for name in names}
    yb = {name: tuple(o[name] for o in out_b) for name in names}
    yab = {
        name: tuple(tuple(o[name] for o in column) for column in out_ab)
        for name in names
    }
    return ya, yb, yab


def _percentile_intervals(
    ya: Sequence[float],
    yb: Sequence[float],
    yab: Sequence[Sequence[float]],
    first: Sequence[float],
    total: Sequence[float],
    bootstrap: int,
    confidence: float,
    stream: RandomStream,
) -> tuple[list[IndexInterval], list[IndexInterval]]:
    """Return row-bootstrap percentile intervals (valid for i.i.d. rows only).

    References:
        [EQ-SENS-BOOT-001], [REF-ARCHER-SALTELLI-SOBOL-1997],
        [REF-EFRON-TIBSHIRANI-1993].

    """
    size = len(ya)
    dimension = len(yab)
    generator = stream.generator()
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
    first_intervals = [
        IndexInterval(
            first[i],
            quantile(first_draws[i], tail),
            quantile(first_draws[i], 1.0 - tail),
        )
        for i in range(dimension)
    ]
    total_intervals = [
        IndexInterval(
            total[i],
            quantile(total_draws[i], tail),
            quantile(total_draws[i], 1.0 - tail),
        )
        for i in range(dimension)
    ]
    return first_intervals, total_intervals


def _replicate_interval(values: Sequence[float], confidence: float) -> IndexInterval:
    """Return ``mean +- t_(R-1) s / sqrt(R)`` over independent replicates.

    References:
        [EQ-SENS-RQMC-001], [REF-L-ECUYER-2018], [REF-OWEN-2013].

    """
    count = len(values)
    center = math.fsum(values) / count
    spread = math.sqrt(math.fsum((v - center) ** 2 for v in values) / (count - 1))
    factor = StudentT(float(count - 1)).quantile(0.5 + 0.5 * confidence)
    half = factor * spread / math.sqrt(count)
    return IndexInterval(center, center - half, center + half)


def _validated_settings(
    model: ProbabilisticModel,
    size: int,
    bootstrap: int,
    confidence: float,
    replicates: int | None,
    design: str,
) -> tuple[str, int]:
    """Validate shared Sobol settings; return the interval method and ``R``.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not model.inputs.is_independent:
        raise ValueError(
            "variance-based Sobol indices require independent inputs; "
            "the model's joint law has a non-identity copula"
        )
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least two")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    if replicates is not None:
        if (
            isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or replicates < 2
        ):
            raise ValueError("replicates must be an integer of at least two")
        if size // replicates < 2:
            raise ValueError("size must give at least two rows per replicate")
        return "replicates", replicates
    if bootstrap == 0:
        return "none", 1
    if design == "mc":
        return "bootstrap", 1
    count = DEFAULT_REPLICATES
    while count > 2 and size // count < 2:
        count //= 2
    if size // count < 2:
        raise ValueError("size is too small for replicate intervals")
    return "replicates", count


def sobol_analysis_all(
    model: ProbabilisticModel,
    size: int,
    *,
    quantities: Sequence[str] | None = None,
    design: str = "sobol",
    seed: int | RandomStream | None = None,
    bootstrap: int = 0,
    confidence: float = 0.95,
    replicates: int | None = None,
) -> MultiQuantitySobolResult:
    """Estimate Sobol indices of several QoIs from one shared pick-freeze design.

    Every design row is one model call that returns every QoI, so ``k``
    quantities cost ``size (d + 2)`` evaluations in total instead of
    ``k size (d + 2)``. The per-quantity results are identical to calling
    :func:`sobol_analysis` with the same seed for each quantity.

    Args:
        model: Probabilistic model with mutually independent inputs.
        size: Total number of design rows.
        quantities: QoI names (default: every QoI of the model).
        design: ``"sobol"``, ``"halton"``, ``"lhs"``, or ``"mc"``.
        seed: Root seed or stream.
        bootstrap: Request intervals; see :func:`sobol_analysis`.
        confidence: Nominal interval level.
        replicates: Independent randomizations for replicate intervals.

    Returns:
        One :class:`ModelSobolResult` per QoI and the shared evaluation count.

    Raises:
        ValueError: For invalid settings, dependent inputs, or a QoI with zero
            variance.
        KeyError: For an unknown QoI name.

    Equation:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [EQ-SENS-BOOT-001],
        [EQ-SENS-RQMC-001].

    References:
        [REF-SALTELLI-2010], [REF-L-ECUYER-2018].

    """
    names = model.quantity_names if quantities is None else tuple(quantities)
    if not names:
        raise ValueError("at least one quantity is required")
    if len(set(names)) != len(names):
        raise ValueError("quantities must be distinct")
    for name in names:
        model.quantity(name)
    interval, count = _validated_settings(
        model, size, bootstrap, confidence, replicates, design
    )
    dimension = model.dimension
    root = resolve_stream(seed)
    method = design
    if design == "sobol" and 2 * dimension > SOBOL_MAX_DIMENSION:
        method = "halton"
    before = model.evaluations
    results: dict[str, ModelSobolResult] = {}
    if interval != "replicates":
        ya, yb, yab = _pick_freeze_outputs(
            model, names, method, size, root.child("sobol-design")
        )
        evaluations = model.evaluations - before
        for name in names:
            first, total, variance = _indices(
                ya[name], yb[name], yab[name], tuple(range(size))
            )
            if interval == "bootstrap":
                first_intervals, total_intervals = _percentile_intervals(
                    ya[name],
                    yb[name],
                    yab[name],
                    first,
                    total,
                    bootstrap,
                    confidence,
                    root.child("sobol-bootstrap"),
                )
            else:
                first_intervals = [IndexInterval(v, math.nan, math.nan) for v in first]
                total_intervals = [IndexInterval(v, math.nan, math.nan) for v in total]
            results[name] = ModelSobolResult(
                name,
                model.inputs.names,
                tuple(first_intervals),
                tuple(total_intervals),
                SobolResult(
                    tuple(first), tuple(total), None, variance, size, evaluations
                ),
                method,
                bootstrap if interval == "bootstrap" else 0,
                confidence,
                interval,
                0,
            )
        return MultiQuantitySobolResult(MappingProxyType(results), evaluations)
    rows = size // count
    replicate_values: dict[str, list[tuple[list[float], list[float], float]]] = {
        name: [] for name in names
    }
    for r in range(count):
        ya, yb, yab = _pick_freeze_outputs(
            model, names, method, rows, root.child("sobol-design", r)
        )
        for name in names:
            replicate_values[name].append(
                _indices(ya[name], yb[name], yab[name], tuple(range(rows)))
            )
    evaluations = model.evaluations - before
    for name in names:
        values = replicate_values[name]
        first_intervals = [
            _replicate_interval([v[0][i] for v in values], confidence)
            for i in range(dimension)
        ]
        total_intervals = [
            _replicate_interval([v[1][i] for v in values], confidence)
            for i in range(dimension)
        ]
        variance = math.fsum(v[2] for v in values) / count
        results[name] = ModelSobolResult(
            name,
            model.inputs.names,
            tuple(first_intervals),
            tuple(total_intervals),
            SobolResult(
                tuple(item.estimate for item in first_intervals),
                tuple(item.estimate for item in total_intervals),
                None,
                variance,
                rows * count,
                evaluations,
            ),
            method,
            0,
            confidence,
            "replicates",
            count,
        )
    return MultiQuantitySobolResult(MappingProxyType(results), evaluations)


def sobol_analysis(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    design: str = "sobol",
    seed: int | RandomStream | None = None,
    bootstrap: int = 0,
    confidence: float = 0.95,
    replicates: int | None = None,
) -> ModelSobolResult:
    """Estimate first-order and total Sobol indices of a model QoI.

    The A and B matrices are the two halves of one ``2d``-dimensional design,
    so quasi-random designs remain low-discrepancy across both. Cost is
    ``size (d + 2)`` model evaluations; to analyse several QoIs at that cost
    use :func:`sobol_analysis_all`.

    Error bars (``interval_method`` of the result):

    * ``"bootstrap"``: rows resampled with replacement, percentile intervals
      ([EQ-SENS-BOOT-001]); used for ``design="mc"`` with ``bootstrap > 0``.
      Only plain Monte Carlo rows are i.i.d.; resampling the rows of a
      quasi-random or Latin-hypercube design destroys its structure and gives
      grossly wrong (typically far too wide) intervals, so it is never done.
    * ``"replicates"``: for ``"sobol"``, ``"halton"`` and ``"lhs"`` (and for
      ``"mc"`` when ``replicates`` is given) the ``size`` rows are split into
      ``R`` independent randomizations (scramblings) of ``size // R`` rows;
      the estimate is the mean of the ``R`` replicate estimates and the
      interval is ``mean +- t_(R-1) s / sqrt(R)`` ([EQ-SENS-RQMC-001]).
      ``R`` is ``replicates``, or 8 (halved while fewer than two rows per
      replicate remain) when only ``bootstrap > 0`` asks for intervals; the
      ``bootstrap`` count is then not used. Powers of two for ``size // R``
      suit Sobol designs.
    * ``"none"``: no interval (``lower`` and ``upper`` are ``nan``).

    Second-order indices are not estimated here; ``result.second_order`` is
    ``None`` (use :func:`estimate_sobol_indices` for them).

    Args:
        model: Probabilistic model with mutually independent inputs.
        quantity: QoI name.
        size: Number of design rows (powers of two suit Sobol designs).
        design: ``"sobol"``, ``"halton"``, ``"lhs"``, or ``"mc"``; Sobol falls
            back to scrambled Halton when ``2d`` exceeds its dimension limit.
        seed: Root seed or stream.
        bootstrap: Number of bootstrap replicates for ``"mc"``; for other
            designs any positive value requests replicate intervals.
        confidence: Nominal interval level.
        replicates: Number ``R >= 2`` of independent randomizations for
            replicate intervals (any design).

    Returns:
        Indices with optional intervals, the interval method, and the
        evaluation count.

    Raises:
        ValueError: If inputs are dependent, ``size < 2``, settings are
            invalid, or the QoI has zero variance.

    Equation:
        [EQ-SENS-ANOVA-001], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001],
        [EQ-SENS-BOOT-001], [EQ-SENS-RQMC-001].

    References:
        [REF-SALTELLI-2010], [REF-JANSEN-1999], [REF-ARCHER-SALTELLI-SOBOL-1997],
        [REF-L-ECUYER-2018].

    """
    model.quantity(quantity)
    return sobol_analysis_all(
        model,
        size,
        quantities=(quantity,),
        design=design,
        seed=seed,
        bootstrap=bootstrap,
        confidence=confidence,
        replicates=replicates,
    ).results[quantity]


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
