"""Effect measures and residuals for contingency tables.

Two-by-two tables are laid out with exposure (or treatment) in rows and the
outcome in columns::

                 event   no event
    exposed        a        b
    unexposed      c        d

so the risks are ``p1 = a / (a + b)`` and ``p2 = c / (c + d)``. Paired binary
data (before/after, two raters) use the same layout for McNemar's test, where
only the discordant cells ``b`` and ``c`` carry information.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.validation import require_probability

from .confidence import ConfidenceInterval, wilson_score_interval
from .hypothesis_tests import _table, binomial_test
from .sampling_distributions import chi_square_sf


def _two_by_two(table: Iterable[Iterable[object]]) -> tuple[int, int, int, int]:
    """Validate a 2 x 2 table of non-negative integer counts.

    Raises:
        TypeError: If a count is not a real number.
        ValueError: If the table is not 2 x 2 or a count is not a
            non-negative integer.

    References:
        [REF-AGRESTI-2013, Chapter 2]. No equation is evaluated.

    """
    rows = [list(row) for row in table]
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError("a 2 x 2 table is required")
    counts: list[int] = []
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"table[{i}][{j}] must be a real number")
            number = float(value)
            if not (math.isfinite(number) and number >= 0.0 and number.is_integer()):
                raise ValueError(f"table[{i}][{j}] must be a non-negative integer")
            counts.append(int(number))
    a, b, c, d = counts
    return a, b, c, d


def _critical(confidence: float) -> float:
    """Return the two-sided standard-normal critical value.

    Raises:
        ValueError: If ``confidence`` is not strictly inside ``(0, 1)``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9]. No new equation is evaluated.

    """
    level = require_probability(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    return NormalDist().inv_cdf(0.5 + 0.5 * level)


@dataclass(frozen=True, slots=True)
class TwoByTwoMeasures:
    """Risk difference, risk ratio, and odds ratio with confidence intervals.

    Attributes:
        table: Counts ``(a, b, c, d)``.
        risk_exposed: ``p1 = a / (a + b)``.
        risk_unexposed: ``p2 = c / (c + d)``.
        risk_difference: ``p1 - p2``.
        risk_difference_interval: Newcombe hybrid-score interval.
        risk_ratio: ``p1 / p2`` (``inf`` if ``c = 0 < a``, ``nan`` if both
            risks are zero).
        risk_ratio_interval: Katz log interval, or ``None`` when a risk is
            zero.
        odds_ratio: ``ad / bc`` on the corrected table when a cell is zero.
        odds_ratio_interval: Woolf log interval.
        corrected: Whether 0.5 was added to every cell for the odds ratio
            (Haldane-Anscombe) because a cell was zero.
        confidence: Nominal two-sided level.

    References:
        [REF-NEWCOMBE-1998], [REF-KATZ-ETAL-1978], [REF-WOOLF-1955],
        [REF-AGRESTI-2013, Chapter 3].

    """

    table: tuple[int, int, int, int]
    risk_exposed: float
    risk_unexposed: float
    risk_difference: float
    risk_difference_interval: ConfidenceInterval
    risk_ratio: float
    risk_ratio_interval: ConfidenceInterval | None
    odds_ratio: float
    odds_ratio_interval: ConfidenceInterval
    corrected: bool
    confidence: float


def two_by_two_measures(
    table: Iterable[Iterable[object]], *, confidence: float = 0.95
) -> TwoByTwoMeasures:
    """Return the three standard effect measures of a 2 x 2 table.

    Risk difference ``p1 - p2`` with Newcombe's hybrid score interval
    (method 10): with Wilson limits ``(l_i, u_i)`` for each risk,
    ``[RD - sqrt((p1 - l1)^2 + (u2 - p2)^2),
    RD + sqrt((u1 - p1)^2 + (p2 - l2)^2)]``.
    Risk ratio with the Katz log interval,
    ``SE(log RR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))``. Odds ratio with the
    Woolf log interval, ``SE(log OR) = sqrt(1/a + 1/b + 1/c + 1/d)``, after
    adding 0.5 to every cell when any cell is zero.

    Args:
        table: ``[[a, b], [c, d]]`` with exposure in rows and the outcome
            (event, no event) in columns; both row totals must be positive.
        confidence: Nominal two-sided level.

    Returns:
        :class:`TwoByTwoMeasures`.

    Raises:
        TypeError: If a count is not real.
        ValueError: If the table is not 2 x 2, a count is not a non-negative
            integer, a row total is zero, or ``confidence`` is invalid.

    Equation:
        [EQ-INF-CT-001].

    References:
        [REF-NEWCOMBE-1998], [REF-KATZ-ETAL-1978], [REF-WOOLF-1955],
        [REF-AGRESTI-2013, Chapter 3].

    """
    a, b, c, d = _two_by_two(table)
    if a + b == 0 or c + d == 0:
        raise ValueError("both row totals must be positive")
    z = _critical(confidence)
    n1, n2 = a + b, c + d
    p1, p2 = a / n1, c / n2
    w1 = wilson_score_interval(a, n1, confidence=confidence)
    w2 = wilson_score_interval(c, n2, confidence=confidence)
    difference = p1 - p2
    rd_interval = ConfidenceInterval(
        max(-1.0, difference - math.hypot(p1 - w1.lower, w2.upper - p2)),
        min(1.0, difference + math.hypot(w1.upper - p1, p2 - w2.lower)),
        confidence,
        "Newcombe hybrid score",
    )
    if a > 0 and c > 0:
        ratio = p1 / p2
        se = math.sqrt(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n2)
        rr_interval: ConfidenceInterval | None = ConfidenceInterval(
            ratio * math.exp(-z * se), ratio * math.exp(z * se), confidence, "Katz log"
        )
    else:
        ratio = math.inf if a > 0 else (0.0 if c > 0 else math.nan)
        rr_interval = None
    corrected = 0 in (a, b, c, d)
    shift = 0.5 if corrected else 0.0
    aa, bb, cc, dd = (value + shift for value in (a, b, c, d))
    odds = (aa * dd) / (bb * cc)
    se_or = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    or_interval = ConfidenceInterval(
        odds * math.exp(-z * se_or),
        odds * math.exp(z * se_or),
        confidence,
        "Woolf log" + (" (Haldane-Anscombe 0.5)" if corrected else ""),
    )
    return TwoByTwoMeasures(
        (a, b, c, d),
        p1,
        p2,
        difference,
        rd_interval,
        ratio,
        rr_interval,
        odds,
        or_interval,
        corrected,
        confidence,
    )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """McNemar's test for paired binary data.

    Attributes:
        discordant: ``(b, c)``, the pairs that changed in each direction.
        statistic: ``(|b - c| - k)^2 / (b + c)`` with ``k = 1`` under the
            continuity correction and ``0`` otherwise (``nan`` if ``b + c = 0``).
        p_value: Exact binomial or chi-square (1 df) two-sided p-value.
        method: ``"exact"`` or ``"chi-square"`` (with or without correction).
        odds_ratio: Conditional (matched-pairs) odds ratio ``b / c``.

    References:
        [REF-MCNEMAR-1947], [REF-AGRESTI-2013, Chapter 11].

    """

    discordant: tuple[int, int]
    statistic: float
    p_value: float
    method: str
    odds_ratio: float


def mcnemar_test(
    table: Iterable[Iterable[object]],
    *,
    exact: bool | None = None,
    continuity_correction: bool = True,
) -> McNemarResult:
    """Test marginal homogeneity of a paired 2 x 2 table.

    Under the null the discordant count ``b`` is ``Binomial(b + c, 1/2)``.
    The exact test uses that law (two-sided, doubling the smaller tail);
    the asymptotic test uses ``(|b - c| - 1)^2 / (b + c)`` (or without the
    ``-1`` when ``continuity_correction`` is false) against chi-square with
    one degree of freedom.

    Args:
        table: ``[[a, b], [c, d]]`` of paired outcomes (first measurement in
            rows, second in columns).
        exact: Use the exact binomial test; ``None`` chooses it when
            ``b + c < 25``.
        continuity_correction: Apply Edwards' correction in the asymptotic
            test.

    Returns:
        :class:`McNemarResult`.

    Raises:
        TypeError: If a count is not real.
        ValueError: If the table is not a 2 x 2 table of non-negative
            integers.

    Equation:
        [EQ-INF-CT-002].

    References:
        [REF-MCNEMAR-1947], [REF-AGRESTI-2013, Chapter 11].

    """
    _, b, c, _ = _two_by_two(table)
    discordant = b + c
    odds = b / c if c > 0 else (math.inf if b > 0 else math.nan)
    use_exact = discordant < 25 if exact is None else exact
    if discordant == 0:
        return McNemarResult((b, c), math.nan, 1.0, "no discordant pairs", odds)
    k = 1.0 if continuity_correction else 0.0
    statistic = max(abs(b - c) - k, 0.0) ** 2 / discordant
    if use_exact:
        p_value = binomial_test(b, discordant, null_probability=0.5).p_value
        method = "exact"
    else:
        p_value = chi_square_sf(statistic, 1.0)
        method = "chi-square" + (" (continuity-corrected)" if k else "")
    return McNemarResult((b, c), statistic, min(1.0, p_value), method, odds)


@dataclass(frozen=True, slots=True)
class ContingencyResiduals:
    """Expected counts and residuals of an ``r x c`` table under independence.

    Attributes:
        expected: ``E_ij = R_i C_j / n``.
        pearson: ``(O_ij - E_ij) / sqrt(E_ij)``; their squares sum to the
            Pearson chi-square statistic.
        adjusted: Haberman's adjusted residuals
            ``(O - E) / sqrt(E (1 - R_i/n)(1 - C_j/n))``, approximately
            standard normal under independence.
        row_proportions: ``O_ij / R_i`` (conditional distribution of the
            column variable given the row).
        column_proportions: ``O_ij / C_j``.

    References:
        [REF-HABERMAN-1973], [REF-AGRESTI-2013, Chapter 2].

    """

    expected: tuple[tuple[float, ...], ...]
    pearson: tuple[tuple[float, ...], ...]
    adjusted: tuple[tuple[float, ...], ...]
    row_proportions: tuple[tuple[float, ...], ...]
    column_proportions: tuple[tuple[float, ...], ...]

    def largest_adjusted(self) -> tuple[int, int, float]:
        """Return ``(row, column, residual)`` of the largest ``|adjusted|``.

        References:
            [REF-HABERMAN-1973]. No new equation is evaluated.

        """
        best = (0, 0, self.adjusted[0][0])
        for i, row in enumerate(self.adjusted):
            for j, value in enumerate(row):
                if abs(value) > abs(best[2]):
                    best = (i, j, value)
        return best


def contingency_residuals(table: Iterable[Iterable[object]]) -> ContingencyResiduals:
    """Return expected counts, Pearson and adjusted residuals, and proportions.

    Args:
        table: Row-wise counts of at least 2 x 2 with no empty row or column.

    Returns:
        :class:`ContingencyResiduals`.

    Raises:
        TypeError: If a count is not real.
        ValueError: If the table is not rectangular or has an empty row or
            column.

    Equation:
        [EQ-INF-CT-003].

    References:
        [REF-HABERMAN-1973], [REF-AGRESTI-2013, Chapter 2].

    """
    rows = _table(table)
    row_totals = [math.fsum(row) for row in rows]
    column_totals = [math.fsum(column) for column in zip(*rows, strict=True)]
    n = math.fsum(row_totals)
    expected = tuple(tuple(r * c / n for c in column_totals) for r in row_totals)
    pearson = tuple(
        tuple((o - e) / math.sqrt(e) for o, e in zip(row, row_e, strict=True))
        for row, row_e in zip(rows, expected, strict=True)
    )
    adjusted = tuple(
        tuple(
            (o - e)
            / math.sqrt(e * (1.0 - row_totals[i] / n) * (1.0 - column_totals[j] / n))
            for j, (o, e) in enumerate(zip(row, row_e, strict=True))
        )
        for i, (row, row_e) in enumerate(zip(rows, expected, strict=True))
    )
    row_proportions = tuple(
        tuple(o / row_totals[i] for o in row) for i, row in enumerate(rows)
    )
    column_proportions = tuple(
        tuple(o / column_totals[j] for j, o in enumerate(row)) for row in rows
    )
    return ContingencyResiduals(
        expected, pearson, adjusted, row_proportions, column_proportions
    )
