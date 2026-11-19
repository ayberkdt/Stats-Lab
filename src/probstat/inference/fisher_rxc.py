"""Fisher's exact test for r x c contingency tables (Freeman-Halton).

Given both margins, a table of counts ``x_ij`` has the multiple
hypergeometric probability ``P(x) = prod r_i! prod c_j! / (N! prod x_ij!)``
under independence. The two-sided p-value sums ``P`` over every table with
the observed margins that is no more probable than the observed one.

The sum is computed exactly, without listing every table, on the network of
Mehta and Patel: the columns are filled one at a time, and a node is the
multiset of row totals still to be filled. For every node the total
probability of its completions is known in closed form, and the largest and
smallest completion probabilities (the longest and shortest paths to the
terminal node) are found by memoized recursion over the network. A partial
table whose completions are all at most as probable as the observed table
is added in one step, one whose completions are all more probable is
dropped, and only the remaining ones are extended. Partial tables that
reach the same node with the same multiset of cell counts have the same
probability and are merged with a multiplicity.

With two columns left a completion is fixed by one column split, so the
completion weights of such a node are sorted once and the summed weight
below any threshold is read off by binary search.

The work grows quickly with the number of cells and the counts (about
6.5e5 evaluations for a 4 x 4 table with ``N = 96``); past
``max_evaluations`` the exact test stops with an error, and
``simulate=True`` gives a seeded Monte Carlo p-value instead.

References:
    [EQ-INF-FISHER-RXC-001], [REF-FREEMAN-HALTON-1951],
    [REF-MEHTA-PATEL-1983], [REF-AGRESTI-2013, Chapter 3].

"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_non_negative_integer

from ._hypothesis import clamp_probability

_RELATIVE_TIE = 1.0 + 1e-7

# An edge of the network: (log weight -sum log x! of the column split, the
# split's counts of at least two in ascending order, the child node, and the
# number of splits of the node that share these three values).
_Edge = tuple[float, tuple[int, ...], tuple[int, ...], int]


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
        evaluations: Work of the exact sum (network edges enumerated plus
            partial tables examined) or random tables drawn (Monte Carlo).
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
        ValueError: If a count is negative, the table is ragged, or it has
            fewer than two non-empty rows or columns.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-FREEMAN-HALTON-1951].

    """
    raw = [
        [require_non_negative_integer(value, name="table entry") for value in row]
        for row in table
    ]
    if not raw or not raw[0] or len({len(row) for row in raw}) != 1:
        raise ValueError("the table must be a non-empty rectangular array of counts")
    rows = [row for row in raw if sum(row) > 0]
    keep = [j for j in range(len(raw[0])) if any(row[j] for row in rows)]
    counts = [[row[j] for j in keep] for row in rows]
    if len(counts) < 2 or len(keep) < 2:
        raise ValueError(
            "the table needs at least two rows and two columns with counts"
        )
    return counts


def _splits(total: int, capacities: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return every split of ``total`` into cells bounded by ``capacities``.

    Args:
        total: Column total to distribute.
        capacities: Remaining row totals.

    Returns:
        Cell vectors ``x`` with ``0 <= x_i <= capacities[i]`` and
        ``sum x = total``.

    References:
        [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

    """
    size = len(capacities)
    suffix = [0] * (size + 1)
    for index in range(size - 1, -1, -1):
        suffix[index] = suffix[index + 1] + capacities[index]
    found: list[tuple[int, ...]] = []
    cells = [0] * size

    def fill(index: int, remaining: int) -> None:
        """Choose the cell ``index`` and recurse on the rest.

        References:
            [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

        """
        if index == size - 1:
            cells[index] = remaining
            found.append(tuple(cells))
            return
        rest = suffix[index + 1]
        for value in range(
            max(0, remaining - rest), min(remaining, capacities[index]) + 1
        ):
            cells[index] = value
            fill(index + 1, remaining - value)

    if total <= suffix[0]:
        fill(0, total)
    return found


class _Network:
    """Column-by-column network of remaining row totals.

    A node is ``(column index, ascending remaining row totals)``. Its
    completions are the sub-tables with those row totals and the remaining
    column totals; ``-sum log x!`` of a completion is its log weight. Every
    enumerated column split counts toward the work limit.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

    """

    def __init__(
        self, columns: tuple[int, ...], log_factorial: Sequence[float], limit: int
    ) -> None:
        """Store the column totals, the log-factorial table, and the limit.

        Args:
            columns: Column totals in processing order.
            log_factorial: ``log k!`` for ``k = 0 .. N``.
            limit: Largest number of evaluations before ``ValueError``.

        References:
            [REF-MEHTA-PATEL-1983]. No equation is evaluated.

        """
        self.columns = columns
        self.log_factorial = log_factorial
        self.limit = limit
        self.work = 0
        self._column_terms = [
            math.fsum(log_factorial[c] for c in columns[k:])
            for k in range(len(columns))
        ]
        self._edges: dict[tuple[int, tuple[int, ...]], list[_Edge]] = {}
        self._bounds: dict[tuple[int, tuple[int, ...]], tuple[float, float]] = {}
        self._completions: dict[tuple[int, ...], tuple[list[float], list[float]]] = {}

    def charge(self, amount: int) -> None:
        """Add ``amount`` evaluations and stop past the limit.

        Args:
            amount: Number of evaluations to add.

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
        for the remaining row totals ``R_i``, column totals ``c_k``, and
        total ``n``.

        Args:
            column: Index of the next column to fill.
            rows: Remaining row totals.

        Returns:
            Log of the summed weights.

        References:
            [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

        """
        table = self.log_factorial
        return (
            table[sum(rows)]
            - math.fsum(table[r] for r in rows)
            - self._column_terms[column]
        )

    def edges(self, column: int, rows: tuple[int, ...]) -> list[_Edge]:
        """Return the merged splits of the next column total of a node.

        Args:
            column: Index of the column to fill.
            rows: Remaining row totals.

        Returns:
            ``(log weight, counts of at least two, child rows, multiplicity)``
            for every distinct split.

        Raises:
            ValueError: If the work exceeds the limit.

        References:
            [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

        """
        key = (column, rows)
        cached = self._edges.get(key)
        if cached is None:
            splits = _splits(self.columns[column], rows)
            self.charge(len(splits))
            table = self.log_factorial
            merged: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
            for cells in splits:
                child = tuple(sorted(r - x for r, x in zip(rows, cells, strict=True)))
                multiset = tuple(sorted(x for x in cells if x > 1))
                merged[(multiset, child)] = merged.get((multiset, child), 0) + 1
            cached = [
                (-math.fsum(table[x] for x in multiset), multiset, child, count)
                for (multiset, child), count in merged.items()
            ]
            self._edges[key] = cached
        return cached

    def completions(self, rows: tuple[int, ...]) -> tuple[list[float], list[float]]:
        """Return the sorted completion log weights of a node before the last column.

        With two columns left, a completion is fixed by the split ``x`` of
        the first of them and has log weight ``-sum (log x_i! + log (R_i -
        x_i)!)``. The weights are returned in ascending order with their
        running log-sum-exp, so the summed weight of every completion below
        a threshold is one binary search away.

        Args:
            rows: Remaining row totals.

        Returns:
            ``(ascending log weights, log of their running sums)``.

        Raises:
            ValueError: If the work exceeds the limit.

        References:
            [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

        """
        cached = self._completions.get(rows)
        if cached is None:
            splits = _splits(self.columns[-2], rows)
            self.charge(len(splits))
            table = self.log_factorial
            weights = sorted(
                -math.fsum(
                    table[x] + table[r - x] for r, x in zip(rows, cells, strict=True)
                )
                for cells in splits
            )
            running: list[float] = []
            accumulated = -math.inf
            for weight in weights:
                high, low = max(accumulated, weight), min(accumulated, weight)
                accumulated = high + math.log1p(math.exp(low - high))
                running.append(accumulated)
            cached = (weights, running)
            self._completions[rows] = cached
        return cached

    def bounds(self, column: int, rows: tuple[int, ...]) -> tuple[float, float]:
        """Return the largest and smallest completion log weights of a node.

        Args:
            column: Index of the next column to fill (before the last one).
            rows: Remaining row totals.

        Returns:
            ``(longest path, shortest path)`` to the terminal node.

        Raises:
            ValueError: If the work exceeds the limit.

        References:
            [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

        """
        key = (column, rows)
        cached = self._bounds.get(key)
        if cached is not None:
            return cached
        if column == len(self.columns) - 2:
            weights, _ = self.completions(rows)
            result = (weights[-1], weights[0])
        else:
            largest, smallest = -math.inf, math.inf
            for step, _, child, _ in self.edges(column, rows):
                high, low = self.bounds(column + 1, child)
                largest = max(largest, step + high)
                smallest = min(smallest, step + low)
            result = (largest, smallest)
        self._bounds[key] = result
        return result


def _exact_p_value(
    counts: list[list[int]], log_factorial: Sequence[float], max_evaluations: int
) -> tuple[float, int]:
    """Sum the null probabilities of tables no more probable than observed.

    Args:
        counts: Table without empty rows or columns.
        log_factorial: ``log k!`` for ``k = 0 .. N``.
        max_evaluations: Largest number of evaluations (network edges
            enumerated plus partial tables examined).

    Returns:
        ``(p-value, evaluations)``.

    Raises:
        ValueError: If more than ``max_evaluations`` evaluations are needed.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-MEHTA-PATEL-1983].

    """
    row_totals = tuple(sorted(sum(row) for row in counts))
    column_totals = sorted(
        (sum(column) for column in zip(*counts, strict=True)), reverse=True
    )
    network = _Network(tuple(column_totals), log_factorial, max_evaluations)
    log_normalizer = network.total(0, row_totals)
    observed = -math.fsum(log_factorial[x] for row in counts for x in row)
    threshold = observed + math.log(_RELATIVE_TIE)
    terms: list[float] = []
    # Partial tables at a node, keyed by their sorted counts of at least two:
    # [multiplicity, log weight].
    level: dict[tuple[int, ...], dict[tuple[int, ...], list[float]]] = {
        row_totals: {(): [1, 0.0]}
    }
    last = len(network.columns) - 2
    for column in range(last + 1):
        following: dict[tuple[int, ...], dict[tuple[int, ...], list[float]]] = {}
        for rows, partials in level.items():
            high, low = network.bounds(column, rows)
            log_total = network.total(column, rows)
            network.charge(len(partials))
            for key, (multiplicity, weight) in partials.items():
                if weight + high <= threshold:
                    terms.append(
                        multiplicity * math.exp(weight + log_total - log_normalizer)
                    )
                elif weight + low > threshold:
                    continue
                elif column == last:
                    weights, running = network.completions(rows)
                    index = bisect.bisect_right(weights, threshold - weight)
                    if index:
                        terms.append(
                            multiplicity
                            * math.exp(weight + running[index - 1] - log_normalizer)
                        )
                else:
                    for step, multiset, child, count in network.edges(column, rows):
                        bucket = following.setdefault(child, {})
                        merged = tuple(sorted(key + multiset)) if multiset else key
                        entry = bucket.get(merged)
                        if entry is None:
                            bucket[merged] = [multiplicity * count, weight + step]
                        else:
                            entry[0] += multiplicity * count
        level = following
    return math.fsum(terms), network.work


def _random_table(
    row_totals: Sequence[int],
    column_labels: list[int],
    columns: int,
    generator: random.Random,
) -> list[int]:
    """Draw a table with the given margins from the multiple hypergeometric law.

    Shuffling the column labels of the ``N`` units and cutting them into
    consecutive blocks of the row totals gives every arrangement of units
    the same probability, hence the null law of tables with fixed margins.

    Args:
        row_totals: Row totals.
        column_labels: Column label of each unit, ``c_j`` copies of ``j``;
            shuffled in place.
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
    max_evaluations: int = 2_000_000,
) -> FisherExactRxCResult:
    """Run Fisher's exact test of independence on an r x c table.

    Conditioning on both margins, the observed table has the multiple
    hypergeometric probability ``P(x) = prod r_i! prod c_j! / (N! prod
    x_ij!)``. The two-sided p-value sums ``P`` over every table with the same
    margins whose probability is at most ``P(x) (1 + 1e-7)``, so that tables
    tied with the observed one up to rounding are counted (the relative
    tolerance of the 2 x 2 ``fisher_exact_test``). For a 2 x 2 table it
    equals the two-sided ``fisher_exact_test``. Rows and columns with a zero
    total are dropped first; they do not change the conditional law.

    The exact sum uses the network algorithm described in the module
    docstring. Its cost depends on the table: a 2 x 3 table with
    ``N = 2757`` takes about 2.7e5 evaluations, a 4 x 4 table with
    ``N = 96`` about 6.5e5 (a few seconds each). When the sum would need
    more than ``max_evaluations``, a ``ValueError`` is raised before any
    p-value is reported.

    With ``simulate=True`` the p-value is ``(1 + #{b : P(x*_b) <= P(x)
    (1 + 1e-7)}) / (B + 1)`` over ``B = replicates`` tables drawn with the
    observed margins (the estimate of R's ``simulate.p.value = TRUE``); it
    is never zero.

    Args:
        table: Rows of non-negative integer counts (at least two non-empty
            rows and columns).
        simulate: Estimate the p-value by Monte Carlo instead of summing it.
        replicates: Number ``B`` of random tables when ``simulate`` is true.
        seed: Seed of the random tables; ``None`` draws fresh entropy.
        max_evaluations: Largest number of evaluations (network edges
            enumerated plus partial tables examined) the exact sum may use.

    Returns:
        P-value, observed table probability, shape, and the work done.

    Raises:
        TypeError: If a count, ``replicates``, or ``max_evaluations`` is not
            an integer.
        ValueError: If a count is negative, the table is ragged or has fewer
            than two non-empty rows or columns, ``replicates`` or
            ``max_evaluations`` is not positive, or the exact sum needs more
            than ``max_evaluations`` evaluations.

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
    log_factorial = [math.lgamma(k + 1.0) for k in range(total + 1)]
    observed = -math.fsum(log_factorial[x] for row in counts for x in row)
    log_constant = (
        math.fsum(log_factorial[r] for r in row_totals)
        + math.fsum(log_factorial[c] for c in column_totals)
        - log_factorial[total]
    )
    table_probability = math.exp(log_constant + observed)
    shape = (len(counts), len(column_totals))
    if not simulate:
        p_value, evaluations = _exact_p_value(counts, log_factorial, limit)
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
        cells = _random_table(row_totals, labels, shape[1], generator)
        if -math.fsum(log_factorial[x] for x in cells) <= threshold:
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
