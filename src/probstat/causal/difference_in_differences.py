"""Difference-in-differences for 2x2 designs and staggered adoption.

**2x2.** With one pre and one post period and a treated and a control
group, ``DiD = (mean post - mean pre)_treated - (mean post - mean
pre)_control`` (EQ-CAUSAL-DID-001). Under parallel trends (the treated
group's untreated outcome would have changed, on average, like the control
group's) and no anticipation it estimates the average treatment effect on
the treated (ATT).

**Staggered adoption.** Units start treatment at different periods ``G``
(their *group*) and stay treated. The building block is the group-time
effect (EQ-CAUSAL-DID-002)

``ATT(g, t) = E[Y_t - Y_{g-1} | G = g] - E[Y_t - Y_{g-1} | C]``, ``t >= g``,

with ``C`` the never-treated units or the units not yet treated at ``t``.
Each cell is a 2x2 comparison with a clean comparison group, so
heterogeneous or dynamic effects cannot contaminate it. This is the
unconditional (no covariates) version. Cells are then averaged with
explicit weights: by exposure time (event study, EQ-CAUSAL-DID-004) or
over all post-treatment cells (EQ-CAUSAL-DID-005). Standard errors come
from the influence function of each estimate (EQ-CAUSAL-DID-003) and are
asymptotic in the number of units; periods are treated as fixed.

The static two-way fixed-effects regression coefficient is *not* such an
average: under staggered timing with dynamic effects it also uses
already-treated units as comparisons, and can be far from every sensible
average of the ``ATT(g, t)`` (the tests show a case).

Only balanced panels are supported; pre-treatment (placebo) cells
``t < g - 1`` are not computed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from probstat.linalg import Matrix, Vector, as_matrix, as_vector

from .double_ml import _wald_interval
from .effects import _binary_treatment

Comparison = Literal["never", "notyet"]


@dataclass(frozen=True, slots=True)
class DifferenceInDifferencesResult:
    """Canonical 2x2 difference-in-differences estimate.

    Attributes:
        estimate: Treated mean change minus control mean change.
        standard_error: ``sqrt(s_1^2 / n_1 + s_0^2 / n_0)`` of the
            unit-level changes (sample variances, unequal-variance form).
        treated_change: Mean of ``Y_post - Y_pre`` among treated units.
        control_change: Mean of ``Y_post - Y_pre`` among control units.
        n_treated: Number of treated units.
        n_control: Number of control units.

    References:
        [REF-ROTH-ET-AL-2023].

    """

    estimate: float
    standard_error: float
    treated_change: float
    control_change: float
    n_treated: int
    n_control: int

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the normal-theory interval ``estimate +- z se``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            Lower and upper bounds.

        Raises:
            ValueError: If the level is invalid.

        References:
            [EQ-CAUSAL-DID-001].

        """
        return _wald_interval(self.estimate, self.standard_error, level)


def did_two_by_two(
    outcome_pre: Iterable[object],
    outcome_post: Iterable[object],
    treated: Iterable[object],
) -> DifferenceInDifferencesResult:
    """Estimate the 2x2 difference-in-differences from unit-level data.

    Each unit contributes its change ``d_i = Y_i,post - Y_i,pre``; the
    estimate is ``mean(d | treated) - mean(d | control)``, which equals
    the textbook four-means formula. Differencing within units removes
    unit-level serial correlation from the standard error.

    Args:
        outcome_pre: ``n`` pre-period outcomes.
        outcome_post: ``n`` post-period outcomes of the same units.
        treated: ``n`` binary indicators of the treated group.

    Returns:
        :class:`DifferenceInDifferencesResult`. The estimate is the ATT
        under parallel trends and no anticipation; the standard error
        assumes independent units.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If lengths disagree or a group has fewer than two
            units.

    Equation:
        [EQ-CAUSAL-DID-001].

    References:
        [REF-ROTH-ET-AL-2023].

    """
    pre = as_vector(outcome_pre, name="outcome_pre")
    post = as_vector(outcome_post, name="outcome_post")
    if len(pre) != len(post):
        raise ValueError("outcome_pre and outcome_post must have equal length")
    flags = _binary_treatment(treated, len(pre))
    changes = [b - a for a, b in zip(pre, post, strict=True)]
    arm1 = [c for c, f in zip(changes, flags, strict=True) if f == 1]
    arm0 = [c for c, f in zip(changes, flags, strict=True) if f == 0]
    if min(len(arm1), len(arm0)) < 2:
        raise ValueError("each group needs at least two units")
    mean1, mean0 = statistics.fmean(arm1), statistics.fmean(arm0)
    error = math.sqrt(
        statistics.variance(arm1) / len(arm1) + statistics.variance(arm0) / len(arm0)
    )
    return DifferenceInDifferencesResult(
        estimate=mean1 - mean0,
        standard_error=error,
        treated_change=mean1,
        control_change=mean0,
        n_treated=len(arm1),
        n_control=len(arm0),
    )


def panel_from_long(
    units: Sequence[Hashable],
    periods: Sequence[float],
    outcomes: Sequence[float],
) -> tuple[tuple[Hashable, ...], Vector, Matrix]:
    """Reshape long-format panel records into a balanced unit-by-period matrix.

    Args:
        units: Unit label of each record.
        periods: Period (a number) of each record.
        outcomes: Outcome of each record.

    Returns:
        Unit labels in order of first appearance, the sorted distinct
        periods, and the ``n_units x n_periods`` outcome matrix.

    Raises:
        TypeError: If a period or an outcome is not real.
        ValueError: If lengths disagree, a (unit, period) pair is repeated,
            or the panel is unbalanced (some unit lacks a period).

    References:
        [REF-CALLAWAY-SANTANNA-2021] (balanced panel data setting). No
        equation is evaluated.

    """
    times = as_vector(periods, name="periods")
    values = as_vector(outcomes, name="outcomes")
    labels = tuple(units)
    if not len(labels) == len(times) == len(values):
        raise ValueError("units, periods and outcomes must have equal length")
    grid = tuple(sorted(set(times)))
    table: dict[Hashable, dict[float, float]] = {}
    for label, time, value in zip(labels, times, values, strict=True):
        cell = table.setdefault(label, {})
        if time in cell:
            raise ValueError(f"unit {label!r} has period {time} more than once")
        cell[time] = value
    for label, cell in table.items():
        if len(cell) != len(grid):
            raise ValueError(
                f"unbalanced panel: unit {label!r} is observed in {len(cell)} "
                f"of {len(grid)} periods"
            )
    return (
        tuple(table),
        grid,
        tuple(tuple(cell[time] for time in grid) for cell in table.values()),
    )


@dataclass(frozen=True, slots=True)
class GroupTimeEffect:
    """One group-time average treatment effect ``ATT(g, t)``.

    Attributes:
        group: First treated period ``g``.
        period: Evaluation period ``t >= g``.
        exposure: Number of periods since adoption (position of ``t``
            minus position of ``g`` in the period grid).
        estimate: ``ATT(g, t)`` estimate.
        standard_error: Influence-function standard error.
        n_treated: Units in group ``g``.
        n_comparison: Comparison units used for this cell.

    References:
        [REF-CALLAWAY-SANTANNA-2021].

    """

    group: float
    period: float
    exposure: int
    estimate: float
    standard_error: float
    n_treated: int
    n_comparison: int

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the pointwise interval ``estimate +- z se``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            Lower and upper bounds (pointwise, not simultaneous over
            cells).

        Raises:
            ValueError: If the level is invalid.

        References:
            [EQ-CAUSAL-DID-003].

        """
        return _wald_interval(self.estimate, self.standard_error, level)


@dataclass(frozen=True, slots=True)
class AggregatedEffect:
    """A weighted average of group-time effects.

    Attributes:
        kind: ``"event-study"`` or ``"overall"``.
        exposure: Exposure time ``e`` of an event-study average, else
            ``None``.
        estimate: ``sum_c w_c ATT_c``.
        standard_error: Influence-function standard error that includes
            the estimation of the group-size weights.
        weights: ``(group, period, weight)`` of every averaged cell; the
            weights sum to one.

    References:
        [REF-CALLAWAY-SANTANNA-2021].

    """

    kind: str
    exposure: int | None
    estimate: float
    standard_error: float
    weights: tuple[tuple[float, float, float], ...]

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the pointwise interval ``estimate +- z se``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            Lower and upper bounds.

        Raises:
            ValueError: If the level is invalid.

        References:
            [EQ-CAUSAL-DID-003].

        """
        return _wald_interval(self.estimate, self.standard_error, level)


@dataclass(frozen=True, slots=True)
class GroupTimeEffects:
    """All estimable ``ATT(g, t)`` of a staggered design and their averages.

    Attributes:
        effects: Cells ordered by group, then period.
        comparison: ``"never"`` or ``"notyet"``.
        n_units: Number of units.
        periods: Period grid.
        group_sizes: ``(group, number of units)`` for each treated group.
        n_never_treated: Number of never-treated units.
        skipped_cells: ``(group, period)`` cells with ``t >= g`` that have
            no comparison units (only possible with ``"notyet"``); they
            are excluded from every aggregate.
        influence: One row per cell: the influence-function value of every
            unit (``estimate = ATT + mean(influence)`` to first order).
        unit_groups: Group of each unit (``inf`` for never treated).

    References:
        [REF-CALLAWAY-SANTANNA-2021].

    """

    effects: tuple[GroupTimeEffect, ...]
    comparison: str
    n_units: int
    periods: Vector
    group_sizes: tuple[tuple[float, int], ...]
    n_never_treated: int
    skipped_cells: tuple[tuple[float, float], ...]
    influence: Matrix = field(repr=False)
    unit_groups: Vector = field(repr=False)

    def effect(self, group: float, period: float) -> GroupTimeEffect:
        """Return the cell ``ATT(group, period)``.

        Args:
            group: First treated period.
            period: Evaluation period.

        Returns:
            The matching :class:`GroupTimeEffect`.

        Raises:
            KeyError: If the cell was not estimated.

        References:
            [EQ-CAUSAL-DID-002].

        """
        for cell in self.effects:
            if cell.group == group and cell.period == period:
                return cell
        raise KeyError(f"no ATT estimate for group {group}, period {period}")

    def _aggregate(
        self, indices: Sequence[int], kind: str, exposure: int | None
    ) -> AggregatedEffect:
        """Average cells with weights proportional to their group sizes.

        ``w_c = n_{g(c)} / sum_{c'} n_{g(c')}``. The influence function is
        ``sum_c w_c psi_c,i + sum_c (ATT_c - theta) 1{G_i = g(c)} / P``
        with ``P = sum_c n_{g(c)} / n``; the second term is the delta
        method for the estimated group shares.

        Args:
            indices: Positions of the cells in ``effects``.
            kind: Label of the aggregate.
            exposure: Exposure time or ``None``.

        Returns:
            :class:`AggregatedEffect`.

        References:
            [EQ-CAUSAL-DID-003], [EQ-CAUSAL-DID-004], [EQ-CAUSAL-DID-005].

        """
        cells = [self.effects[i] for i in indices]
        total = sum(cell.n_treated for cell in cells)
        weights = [cell.n_treated / total for cell in cells]
        theta = math.fsum(
            w * cell.estimate for w, cell in zip(weights, cells, strict=True)
        )
        n = self.n_units
        share = total / n
        by_group: dict[float, float] = {}
        for cell in cells:
            by_group[cell.group] = (
                by_group.get(cell.group, 0.0) + (cell.estimate - theta) / share
            )
        squares = 0.0
        for unit in range(n):
            value = math.fsum(
                w * self.influence[i][unit]
                for w, i in zip(weights, indices, strict=True)
            )
            value += by_group.get(self.unit_groups[unit], 0.0)
            squares += value * value
        return AggregatedEffect(
            kind=kind,
            exposure=exposure,
            estimate=theta,
            standard_error=math.sqrt(squares) / n,
            weights=tuple(
                (cell.group, cell.period, w)
                for cell, w in zip(cells, weights, strict=True)
            ),
        )

    def event_study(self) -> tuple[AggregatedEffect, ...]:
        """Average the effects by time since adoption.

        ``theta(e) = sum_g w_g ATT(g, g + e)`` over the groups whose cell
        at exposure ``e`` was estimated, with ``w_g`` proportional to the
        group size. Different exposures average different sets of groups,
        so the path mixes dynamics with composition changes unless the
        groups are balanced in exposure.

        Returns:
            One :class:`AggregatedEffect` per exposure ``e = 0, 1, ...``
            in increasing order.

        Equation:
            [EQ-CAUSAL-DID-004], [EQ-CAUSAL-DID-003].

        References:
            [REF-CALLAWAY-SANTANNA-2021].

        """
        exposures = sorted({cell.exposure for cell in self.effects})
        return tuple(
            self._aggregate(
                [i for i, cell in enumerate(self.effects) if cell.exposure == e],
                "event-study",
                e,
            )
            for e in exposures
        )

    def overall(self) -> AggregatedEffect:
        """Average all estimated post-treatment cells.

        ``theta = sum_{(g,t)} w_{g,t} ATT(g, t)`` with
        ``w_{g,t} = n_g / sum_{(g',t')} n_{g'}`` over every estimated cell
        with ``t >= g``: each cell is weighted by its group size, so a
        group counts in proportion to its size times its number of
        post-treatment periods (early adopters weigh more).

        Returns:
            :class:`AggregatedEffect` with ``kind="overall"``.

        Equation:
            [EQ-CAUSAL-DID-005], [EQ-CAUSAL-DID-003].

        References:
            [REF-CALLAWAY-SANTANNA-2021].

        """
        return self._aggregate(range(len(self.effects)), "overall", None)


def _groups(
    first_treated: Iterable[object], grid: Vector, size: int
) -> tuple[float, ...]:
    """Validate adoption periods and code never-treated units as ``inf``.

    Args:
        first_treated: Per unit the first treated period, or ``None`` (or
            ``inf``) for never-treated units.
        grid: Period grid.
        size: Number of units.

    Returns:
        Group of each unit.

    Raises:
        ValueError: If the length is wrong, a value is not a period of the
            grid, or a unit is treated in the first period.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    groups: list[float] = []
    for index, value in enumerate(first_treated):
        if value is None:
            groups.append(math.inf)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"first_treated[{index}] must be a period or None")
        number = float(value)
        if number == math.inf:
            groups.append(math.inf)
        elif number not in grid:
            raise ValueError(
                f"first_treated[{index}] = {value!r} is not one of the periods; "
                "code units first treated after the last period as None"
            )
        elif number == grid[0]:
            raise ValueError(
                f"unit {index} is treated in the first period: it has no "
                "pre-treatment period and must be removed"
            )
        else:
            groups.append(number)
    if len(groups) != size:
        raise ValueError("first_treated must have one entry per unit")
    return tuple(groups)


def group_time_effects(
    outcomes: Iterable[Iterable[object]],
    first_treated: Iterable[object],
    *,
    periods: Iterable[object] | None = None,
    comparison: Comparison = "never",
) -> GroupTimeEffects:
    """Estimate every ``ATT(g, t)``, ``t >= g``, of a staggered design.

    For a group ``g`` (units first treated in period ``g``) and a period
    ``t >= g`` the estimate is the mean of ``Y_t - Y_{g-1}`` in the group
    minus the mean of the same long difference among the comparison units
    (``g - 1`` is the last period before adoption). With
    ``comparison="never"`` these are the never-treated units; with
    ``"notyet"`` all units with ``G > t`` including the never treated.

    The standard error of a cell is ``sqrt(sum_i psi_i^2) / n`` with
    ``psi_i = (n / n_g) 1{G_i = g}(d_i - mean_g) - (n / n_C) 1{i in C}
    (d_i - mean_C)``, i.e. ``sqrt(v_g / n_g + v_C / n_C)`` with
    divisor-``n`` variances.

    Args:
        outcomes: ``n_units x n_periods`` balanced panel (one row per
            unit, columns in period order).
        first_treated: Per unit the first treated period (a value of
            ``periods``), or ``None`` for never treated. Treatment is
            absorbing.
        periods: Strictly increasing period labels; default
            ``0, 1, ..., n_periods - 1``.
        comparison: ``"never"`` or ``"notyet"``.

    Returns:
        :class:`GroupTimeEffects`. Each cell identifies ``ATT(g, t)``
        under parallel trends between group ``g`` and the comparison
        units from ``g - 1`` to ``t`` and no anticipation.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the panel is ragged (unbalanced), has fewer than
            two periods, the periods are not increasing, an adoption
            period is invalid, no unit is ever treated, ``comparison`` is
            unknown, or ``comparison="never"`` without never-treated
            units.

    Equation:
        [EQ-CAUSAL-DID-002], [EQ-CAUSAL-DID-003].

    References:
        [REF-CALLAWAY-SANTANNA-2021], [REF-ROTH-ET-AL-2023].

    """
    if comparison not in ("never", "notyet"):
        raise ValueError("comparison must be 'never' or 'notyet'")
    try:
        panel = as_matrix(outcomes, name="outcomes")
    except ValueError as error:
        raise ValueError(
            f"outcomes must be a balanced unit-by-period matrix: {error}"
        ) from error
    n, width = len(panel), len(panel[0])
    if width < 2:
        raise ValueError("at least two periods are required")
    if periods is None:
        grid: Vector = tuple(float(j) for j in range(width))
    else:
        grid = as_vector(periods, name="periods")
        if len(grid) != width:
            raise ValueError("periods must have one entry per outcome column")
        if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
            raise ValueError("periods must be strictly increasing")
    groups = _groups(first_treated, grid, n)
    treated_groups = sorted({g for g in groups if g != math.inf})
    if not treated_groups:
        raise ValueError("no unit is ever treated")
    never = sum(1 for g in groups if g == math.inf)
    if comparison == "never" and never == 0:
        raise ValueError("comparison='never' needs never-treated units; use 'notyet'")
    position = {time: j for j, time in enumerate(grid)}
    cells: list[GroupTimeEffect] = []
    influence: list[tuple[float, ...]] = []
    skipped: list[tuple[float, float]] = []
    for g in treated_groups:
        base = position[g] - 1
        members = [i for i in range(n) if groups[i] == g]
        for j in range(position[g], width):
            time = grid[j]
            if comparison == "never":
                others = [i for i in range(n) if groups[i] == math.inf]
            else:
                others = [i for i in range(n) if groups[i] > time]
            if not others:
                skipped.append((g, time))
                continue
            mean_g = statistics.fmean(panel[i][j] - panel[i][base] for i in members)
            mean_c = statistics.fmean(panel[i][j] - panel[i][base] for i in others)
            psi = [0.0] * n
            for i in members:
                psi[i] = n / len(members) * (panel[i][j] - panel[i][base] - mean_g)
            for i in others:
                psi[i] = -n / len(others) * (panel[i][j] - panel[i][base] - mean_c)
            cells.append(
                GroupTimeEffect(
                    group=g,
                    period=time,
                    exposure=j - position[g],
                    estimate=mean_g - mean_c,
                    standard_error=math.sqrt(math.fsum(v * v for v in psi)) / n,
                    n_treated=len(members),
                    n_comparison=len(others),
                )
            )
            influence.append(tuple(psi))
    if not cells:
        raise ValueError("no group-time cell has comparison units")
    return GroupTimeEffects(
        effects=tuple(cells),
        comparison=comparison,
        n_units=n,
        periods=grid,
        group_sizes=tuple(
            (g, sum(1 for value in groups if value == g)) for g in treated_groups
        ),
        n_never_treated=never,
        skipped_cells=tuple(skipped),
        influence=tuple(influence),
        unit_groups=groups,
    )
