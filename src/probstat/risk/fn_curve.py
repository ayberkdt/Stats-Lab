"""F-N curves: frequency of N or more fatalities (or other consequences).

For scenarios ``i`` with frequencies ``f_i`` (per year) and consequences
``N_i``, the F-N curve is the complementary cumulative frequency

    F(N) = sum_{i : N_i >= N} f_i,

a non-increasing step function of ``N`` that is constant on each interval
``(N_(k-1), N_(k)]`` between consecutive distinct consequence values. A
societal-risk criterion line is commonly ``F_c(N) = C / N^a`` with an anchor
frequency ``C`` at ``N = 1`` and an aversion slope ``a`` (``a = 1`` neutral,
``a = 2`` risk-averse). Because ``F_c`` decreases, the worst point of each
step is its right end, so comparing ``F`` with ``F_c`` at the distinct
``N_i`` decides compliance for every ``N`` exactly.

Scenario lists can come from :meth:`EventTree.sequences` (consequence =
fatalities per sequence) or be supplied directly. Scenarios with ``N <= 0``
do not contribute.

Stability: this module is *experimental* (ADR-0063) inside a *provisional*
package; its names and signatures may change in any release.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from probstat.foundations import require_finite_real

from .event_tree import EventTree


@dataclass(frozen=True, slots=True)
class FNCriterion:
    """Criterion line ``F_c(N) = C (N_0 / N)^a``.

    Args:
        anchor_frequency: ``C``, the tolerable frequency at ``N = N_0``.
        slope: Aversion exponent ``a`` (non-negative).
        anchor_consequence: ``N_0`` (default 1).

    References:
        [REF-JONKMAN-VANGELDER-VRIJLING-2003].

    """

    anchor_frequency: float
    slope: float = 1.0
    anchor_consequence: float = 1.0

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a parameter is outside its domain.

        References:
            Project validation policy. No equation is evaluated.

        """
        c = require_finite_real(self.anchor_frequency, name="anchor_frequency")
        a = require_finite_real(self.slope, name="slope")
        n0 = require_finite_real(self.anchor_consequence, name="anchor_consequence")
        if c <= 0.0 or a < 0.0 or n0 <= 0.0:
            raise ValueError("need anchor_frequency > 0, slope >= 0, anchor > 0")
        object.__setattr__(self, "anchor_frequency", c)
        object.__setattr__(self, "slope", a)
        object.__setattr__(self, "anchor_consequence", n0)

    def __call__(self, consequence: float) -> float:
        """Return ``F_c(N)``.

        Args:
            consequence: Positive ``N``.

        Equation:
            [EQ-RISK-FN-002].

        References:
            [REF-JONKMAN-VANGELDER-VRIJLING-2003].

        """
        n = require_finite_real(consequence, name="consequence")
        if n <= 0.0:
            raise ValueError("consequence must be positive")
        ratio = (self.anchor_consequence / n) ** self.slope
        return self.anchor_frequency * float(ratio)


@dataclass(frozen=True, slots=True)
class FNPoint:
    """One step of an F-N curve and its comparison with a criterion.

    Args:
        consequence: Distinct consequence value ``N_(k)``.
        frequency: ``F(N_(k))``, frequency of ``N_(k)`` or more.
        criterion: Criterion value at ``N_(k)`` (``nan`` without criterion).
        ratio: ``frequency / criterion`` (``nan`` without criterion).

    References:
        [REF-JONKMAN-VANGELDER-VRIJLING-2003].

    """

    consequence: float
    frequency: float
    criterion: float
    ratio: float


@dataclass(frozen=True)
class FNCurve:
    """An F-N curve with an optional criterion comparison.

    Args:
        points: Steps in ascending order of consequence.
        expected_consequence: ``sum f_i N_i`` over scenarios with ``N > 0``
            (the expected number of fatalities per year, PLL).
        max_ratio: Largest ``F / F_c`` (``nan`` without criterion).
        compliant: ``max_ratio <= 1`` (``None`` without criterion).

    References:
        [REF-JONKMAN-VANGELDER-VRIJLING-2003].

    """

    points: tuple[FNPoint, ...]
    expected_consequence: float
    max_ratio: float
    compliant: bool | None

    def frequency(self, consequence: float) -> float:
        """Return ``F(N)`` for any ``N > 0``.

        Args:
            consequence: Positive ``N``.

        Equation:
            [EQ-RISK-FN-001].

        References:
            [REF-JONKMAN-VANGELDER-VRIJLING-2003].

        """
        n = require_finite_real(consequence, name="consequence")
        if n <= 0.0:
            raise ValueError("consequence must be positive")
        for point in self.points:
            if point.consequence >= n:
                return point.frequency
        return 0.0

    def exceedances(self) -> tuple[FNPoint, ...]:
        """Return the steps whose frequency exceeds the criterion.

        References:
            [REF-JONKMAN-VANGELDER-VRIJLING-2003]. No new equation.

        """
        return tuple(
            p for p in self.points if not math.isnan(p.ratio) and p.ratio > 1.0
        )


def fn_curve(
    scenarios: Iterable[tuple[float, float]],
    criterion: FNCriterion | Callable[[float], float] | None = None,
) -> FNCurve:
    """Return the F-N curve of ``(frequency, consequence)`` scenarios.

    Args:
        scenarios: Pairs of non-negative frequency and consequence ``N``.
        criterion: Optional non-increasing criterion line ``F_c(N)``. For a
            general callable, compliance is checked at the step ends, which is
            exact only for non-increasing criteria.

    Returns:
        Curve, expected consequence, and the criterion comparison.

    Raises:
        ValueError: If a frequency is negative or no scenario has ``N > 0``.

    Equation:
        [EQ-RISK-FN-001], [EQ-RISK-FN-002].

    References:
        [REF-JONKMAN-VANGELDER-VRIJLING-2003].

    """
    pairs = []
    for frequency, consequence in scenarios:
        f = require_finite_real(frequency, name="frequency")
        n = require_finite_real(consequence, name="consequence")
        if f < 0.0:
            raise ValueError("frequencies must be non-negative")
        if n > 0.0:
            pairs.append((n, f))
    if not pairs:
        raise ValueError("at least one scenario with positive consequence is required")
    grouped: dict[float, list[float]] = {}
    for n, f in pairs:
        grouped.setdefault(n, []).append(f)
    distinct = sorted(grouped)
    totals: dict[float, float] = {}
    tail: list[float] = []
    for n in reversed(distinct):
        tail.extend(grouped[n])
        totals[n] = math.fsum(tail)
    points = []
    for n in distinct:
        total = totals[n]
        if criterion is None:
            limit = ratio = math.nan
        else:
            limit = float(criterion(n))
            ratio = total / limit if limit > 0.0 else math.inf
        points.append(FNPoint(n, total, limit, ratio))
    expected = math.fsum(n * f for n, f in pairs)
    if criterion is None:
        return FNCurve(tuple(points), expected, math.nan, None)
    worst = max(p.ratio for p in points)
    return FNCurve(tuple(points), expected, worst, worst <= 1.0)


def fn_curve_from_event_trees(
    trees: Iterable[EventTree],
    criterion: FNCriterion | Callable[[float], float] | None = None,
) -> FNCurve:
    """Return the F-N curve of all sequences of one or more event trees.

    Args:
        trees: Event trees whose outcome consequences are counts ``N``.
        criterion: Optional criterion line.

    Returns:
        Combined F-N curve.

    Equation:
        [EQ-RISK-FN-001], [EQ-RISK-ET-001].

    References:
        [REF-JONKMAN-VANGELDER-VRIJLING-2003], [REF-RAUSAND-HOYLAND-2004].

    """
    scenarios = [
        (sequence.frequency, sequence.consequence)
        for tree in trees
        for sequence in tree.sequences()
    ]
    return fn_curve(scenarios, criterion)
