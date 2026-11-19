"""Fisher's exact test for r x c contingency tables (Freeman-Halton).

Given both margins, a table of counts ``x_ij`` has the multiple
hypergeometric probability ``P(x) = prod r_i! prod c_j! / (N! prod x_ij!)``
under independence. The two-sided p-value sums ``P`` over every table with
the observed margins that is no more probable than the observed one.

The exact sum is computed column by column on a network whose nodes are the
remaining row totals: for each node the total probability of its
completions, and their largest and smallest probabilities, are known
(the first in closed form, the other two by memoized recursion), so whole
subtrees are added or discarded without listing their tables. This is the
pruning idea of Mehta and Patel's network algorithm with exact path bounds.
The work grows quickly with the table size and the counts; past
``max_evaluations`` the test stops with an error, and ``simulate=True``
gives a seeded Monte Carlo p-value instead.

References:
    [EQ-INF-FISHER-RXC-001], [REF-FREEMAN-HALTON-1951],
    [REF-MEHTA-PATEL-1983], [REF-AGRESTI-2013, Chapter 3].

"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.validation import require_non_negative_integer

from ._hypothesis import clamp_probability

_RELATIVE_TIE = 1.0 + 1e-7


@dataclass(frozen=True, slots=True)
class FisherExactRxCResult:
    """Fisher-Freeman-Halton exact test of independence in an r x c table.

    Args:
        p_value: Conditional probability, given both margins and the null of
            independence, of a table no more probable than the observed one;
            a Monte Carlo estimate when ``exact`` is false. Not the
            probability that the null hypothesis is true.
        table_probability: Null probability ``P(x)`` of the observed table
            given its margins.
        shape: ``(rows, columns)`` after rows and columns with a zero total
            are removed.
        exact: Whether the p-value is the exact sum.
        evaluations: Work of the exact sum (network children enumerated
            plus partial tables examined) or random tables drawn (Monte
            Carlo).
        standard_error: Monte Carlo standard error of the p-value,
            ``sqrt(p (1 - p) / (B + 1))``; ``None`` for the exact test.
        method: Test label.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-FREEMAN-HALTON-1951].

    """

    p_value: float
    table_probability: float
    shape: tuple[int, int]
    exact: bool
    evaluations: int
    standard_error: float | None
    method: str

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-FREEMAN-HALTON-1951]. No equation is evaluated.

        """
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError("p_value must lie in [0, 1]")


def _counts(table: Iterable[Iterable[object]]) -> list[list[int]]:
    """Validate a table of counts and drop rows and columns with zero total.

    Args:
        table: Rows of non-negative integer counts.

    Returns:
        The table without empty rows and columns.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If the table is ragged or has fewer than two non-empty
            rows or columns.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-FREEMAN-HALTON-1951].

    """
    raw = [
        [require_non_negative_integer(value, name="table entry") for value in row]
        for row in table
    ]
    if not raw or len({len(row) for row in raw}) != 1:
        raise ValueError("the table must be a non-empty rectangular array of counts")
    rows = [row for row in raw if sum(row) > 0]
    keep = [j for j in range(len(raw[0])) if any(row[j] for row in rows)]
    counts = [[row[j] for j in keep] for row in rows]
    if len(counts) < 2 or len(keep) < 2:
        raise ValueError(
            "the table needs at least two rows and two columns with counts"
        )
    return counts


def _log_weight(cells: Iterable[int]) -> float:
    """Return ``-sum log(x!)`` over the given cells.

    References:
        [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

    """
    return -math.fsum(math.lgamma(x + 1) for x in cells)


def _compositions(total: int, capacities: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """Yield every split of ``total`` into cells bounded by ``capacities``.

    Args:
        total: Column total to distribute.
        capacities: Remaining row totals.

    Yields:
        Cell vectors ``x`` with ``0 <= x_i <= capacities[i]`` and
        ``sum x = total``.

    References:
        [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

    """
    if len(capacities) == 1:
        if total <= capacities[0]:
            yield (total,)
        return
    rest = sum(capacities[1:])
    for first in range(max(0, total - rest), min(total, capacities[0]) + 1):
        for tail in _compositions(total - first, capacities[1:]):
            yield (first, *tail)


class _Network:
    """Column-by-column network of remaining row totals.

    A node is ``(column index, sorted remaining row totals)``. Its
    completions are the sub-tables with those row totals and the remaining
    column totals; ``-sum log x!`` of a completion is its log weight. Every
    enumerated child counts toward the work limit.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

    """

    def __init__(self, columns: tuple[int, ...], limit: int) -> None:
        """Store the column totals and the work limit.

        Args:
            columns: Column totals in processing order.
            limit: Largest number of evaluations before ``ValueError``.

        References:
            [REF-MEHTA-PATEL-1983]. No equation is evaluated.

        """
        self.columns = columns
        self.limit = limit
        self.work = 0
        self._children: dict[
            tuple[int, tuple[int, ...]], list[tuple[int, tuple[int, ...]]]
        ] = {}
        self._bounds: dict[tuple[int, tuple[int, ...]], tuple[float, float]] = {}

    def charge(self, amount: int) -> None:
        """Add ``amount`` evaluations and stop past the limit.

        Raises:
            ValueError: If the work exceeds the limit.

        References:
            [REF-MEHTA-PATEL-1983]. No equation is evaluated.

        """
        self.work += amount
        if self.work > self.limit:
            raise ValueError(
                "the exact r x c test needs more than max_evaluations="
                f"{self.limit} evaluations; pass simulate=True for a "
                "Monte Carlo p-value or raise max_evaluations"
            )

    def total(self, column: int, rows: tuple[int, ...]) -> float:
        """Return the log of the summed completion weights of a node.

        ``sum over completions of prod 1/x! = n! / (prod R_i! prod c_k!)``
        for the remaining total ``n``.

        References:
            [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

        """
        remaining = sum(rows)
        return (
            math.lgamma(remaining + 1)
            + _log_weight(rows)
            + _log_weight(self.columns[column:])
        )

    def children(
        self, column: int, rows: tuple[int, ...]
    ) -> list[tuple[int, tuple[int, ...]]]:
        """Return ``(prod x!, child rows)`` for every split of the column.

        References:
            [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

        """
        key = (column, rows)
        cached = self._children.get(key)
        if cached is None:
            cached = []
            for cells in _compositions(self.columns[column], rows):
                product = math.prod(math.factorial(x) for x in cells)
                child = tuple(sorted(r - x for r, x in zip(rows, cells, strict=True)))
                cached.append((product, child))
            self.charge(len(cached))
            self._children[key] = cached
        return cached

    def bounds(self, column: int, rows: tuple[int, ...]) -> tuple[float, float]:
        """Return the largest and smallest completion log weights of a node.

        References:
            [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

        """
        key = (column, rows)
        cached = self._bounds.get(key)
        if cached is not None:
            return cached
        if column == len(self.columns) - 1:
            weight = -math.log(math.prod(math.factorial(r) for r in rows))
            result = (weight, weight)
        else:
            largest, smallest = -math.inf, math.inf
            for product, child in self.children(column, rows):
                high, low = self.bounds(column + 1, child)
                step = -math.log(product)
                largest = max(largest, step + high)
                smallest = min(smallest, step + low)
            result = (largest, smallest)
        self._bounds[key] = result
        return result


def _exact_p_value(counts: list[list[int]], max_evaluations: int) -> tuple[float, int]:
    """Sum the null probabilities of tables no more probable than observed.

    The network is traversed one column at a time. Partial tables that reach
    the same node with the same exact product ``prod x!`` are merged with a
    multiplicity, and a node whose completions all lie on one side of the
    threshold is settled from its bounds.

    Args:
        counts: Table without empty rows or columns.
        max_evaluations: Largest number of evaluations (enumerated children
            plus merged partial tables examined).

    Returns:
        ``(p-value, evaluations)``.

    Raises:
        ValueError: If more than ``max_evaluations`` evaluations are needed.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

    """
    row_totals = tuple(sorted(sum(row) for row in counts))
    column_totals = (sum(column) for column in zip(*counts, strict=True))
    network = _Network(tuple(sorted(column_totals, reverse=True)), max_evaluations)
    log_normalizer = network.total(0, row_totals)
    observed = -math.log(math.prod(math.factorial(x) for row in counts for x in row))
    threshold = observed + math.log(_RELATIVE_TIE)
    terms: list[float] = []
    level: dict[tuple[int, ...], dict[int, int]] = {row_totals: {1: 1}}
    for column in range(len(network.columns)):
        following: dict[tuple[int, ...], dict[int, int]] = {}
        for rows, partials in level.items():
            high, low = network.bounds(column, rows)
            log_total = network.total(column, rows)
            network.charge(len(partials))
            for product, multiplicity in partials.items():
                weight = -math.log(product)
                if weight + high <= threshold:
                    terms.append(
                        multiplicity * math.exp(weight + log_total - log_normalizer)
                    )
                elif weight + low <= threshold:
                    for step, child in network.children(column, rows):
                        bucket = following.setdefault(child, {})
                        key = product * step
                        bucket[key] = bucket.get(key, 0) + multiplicity
        level = following
    return math.fsum(terms), network.work


def _random_table(
    row_totals: list[int],
    column_labels: list[int],
    columns: int,
    generator: random.Random,
) -> list[int]:
    """Draw a table with the given margins uniformly over arrangements.

    Shuffling the column labels of the ``N`` units and cutting them into
    consecutive blocks of the row totals gives the multiple hypergeometric
    law of tables with fixed margins.

    Args:
        row_totals: Row totals.
        column_labels: Column label of each unit, ``c_j`` copies of ``j``.
        columns: Number of columns.
        generator: Seeded random generator.

    Returns:
        Cell counts in row-major order.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-AGRESTI-2013, Chapter 3].

    """
    generator.shuffle(column_labels)
    cells = [0] * (len(row_totals) * columns)
    start = 0
    for i, total in enumerate(row_totals):
        for label in column_labels[start : start + total]:
            cells[i * columns + label] += 1
        start += total
    return cells


def fisher_exact_rxc_test(
    table: Iterable[Iterable[object]],
    *,
    simulate: bool = False,
    replicates: int = 2000,
    seed: int | None = None,
    max_evaluations: int = 1_000_000,
) -> FisherExactRxCResult:
    """Run Fisher's exact test of independence on an r x c table.

    Conditioning on both margins, the observed table has the multiple
    hypergeometric probability ``P(x) = prod r_i! prod c_j! / (N! prod
    x_ij!)``. The two-sided p-value sums ``P`` over every table with the same
    margins whose probability is at most ``P(x)`` (relative tolerance
    ``1e-7``, as in R's ``fisher.test``). For a 2 x 2 table it equals the
    two-sided ``fisher_exact_test``. Rows and columns with a zero total are
    dropped first; they do not change the conditional law.

    With ``simulate=True`` the p-value is ``(1 + #{b : P(x*_b) <= P(x)}) /
    (B + 1)`` over ``B = replicates`` tables drawn with the observed margins
    (the estimate of R's ``simulate.p.value = TRUE``); it never reports zero.

    Args:
        table: Rows of non-negative integer counts (at least two non-empty
            rows and columns).
        simulate: Estimate the p-value by Monte Carlo instead of summing it.
        replicates: Number ``B`` of random tables when ``simulate`` is true.
        seed: Seed of the random tables; ``None`` draws fresh entropy.
        max_evaluations: Largest number of network nodes the exact sum may
            evaluate before it stops with ``ValueError``.

    Returns:
        P-value, observed table probability, shape, and the work done.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If the table is ragged, has fewer than two non-empty
            rows or columns, ``replicates`` or ``max_evaluations`` is not
            positive, or the exact sum needs more than ``max_evaluations``
            nodes.

    Equation:
        [EQ-INF-FISHER-RXC-001].

    References:
        [REF-FREEMAN-HALTON-1951], [REF-MEHTA-PATEL-1983],
        [REF-PHIPSON-SMYTH-2010], [REF-AGRESTI-2013, Chapter 3].

    """
    counts = _counts(table)
    limit = require_non_negative_integer(max_evaluations, name="max_evaluations")
    draws = require_non_negative_integer(replicates, name="replicates")
    if limit == 0:
        raise ValueError("max_evaluations must be positive")
    if draws == 0:
        raise ValueError("replicates must be positive")
    row_totals = [sum(row) for row in counts]
    column_totals = [sum(column) for column in zip(*counts, strict=True)]
    total = sum(row_totals)
    cells = [x for row in counts for x in row]
    observed = _log_weight(cells)
    log_constant = (
        math.fsum(math.lgamma(r + 1) for r in row_totals)
        + math.fsum(math.lgamma(c + 1) for c in column_totals)
        - math.lgamma(total + 1)
    )
    table_probability = math.exp(log_constant + observed)
    shape = (len(counts), len(column_totals))
    if not simulate:
        p_value, evaluations = _exact_p_value(counts, limit)
        return FisherExactRxCResult(
            clamp_probability(p_value),
            table_probability,
            shape,
            True,
            evaluations,
            None,
            f"Fisher exact test ({shape[0]} x {shape[1]}, Freeman-Halton)",
        )
    generator = random.Random(seed)
    labels = [j for j, c in enumerate(column_totals) for _ in range(c)]
    threshold = observed + math.log(_RELATIVE_TIE)
    extreme = 0
    for _ in range(draws):
        if (
            _log_weight(_random_table(row_totals, labels, shape[1], generator))
            <= threshold
        ):
            extreme += 1
    p_value = (1 + extreme) / (draws + 1)
    return FisherExactRxCResult(
        p_value,
        table_probability,
        shape,
        False,
        draws,
        math.sqrt(p_value * (1.0 - p_value) / (draws + 1)),
        f"Fisher exact test ({shape[0]} x {shape[1]}, Monte Carlo, B = {draws})",
    )
