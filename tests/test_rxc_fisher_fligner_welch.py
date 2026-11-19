"""Fisher's exact r x c test, the Fligner-Killeen test, and Welch's ANOVA.

Oracles: brute-force enumeration of every table with the observed margins in
exact rational arithmetic, the 2 x 2 ``fisher_exact_test``, the invariance
of the conditional law under row and column permutations and transposition,
the two-group identities ``F = t^2`` of Welch's ANOVA and the Welch t test,
the invariance of the Fligner-Killeen statistic under a shift of each group
and a common rescaling, and values from R 4.6.2 (``fisher.test``,
``fligner.test``, ``oneway.test``) pasted below with the R call.

References:
    [EQ-INF-FISHER-RXC-001], [EQ-INF-FISHER-001], [EQ-INF-FLIGNER-001],
    [EQ-INF-WELCH-ANOVA-001], [EQ-INF-T-003], [REF-FREEMAN-HALTON-1951],
    [REF-MEHTA-PATEL-1983], [REF-FLIGNER-KILLEEN-1976], [REF-WELCH-1951].

"""

from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction

from probstat.inference import (
    fisher_exact_rxc_test,
    fisher_exact_test,
    fligner_killeen_test,
    two_sample_t_test,
    welch_anova,
)


def tables_with_margins(
    rows: list[int], columns: list[int]
) -> list[tuple[tuple[int, ...], ...]]:
    """List every non-negative integer table with the given margins.

    Args:
        rows: Row totals.
        columns: Column totals.

    Returns:
        All tables, row by row.

    References:
        [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

    """
    if len(rows) == 1:
        return [(tuple(columns),)]
    found = []
    first = rows[0]
    ranges = [range(min(first, c) + 1) for c in columns]
    for head in itertools.product(*ranges):
        if sum(head) != first:
            continue
        rest = [c - h for c, h in zip(columns, head, strict=True)]
        for tail in tables_with_margins(rows[1:], rest):
            found.append((head, *tail))
    return found


def exact_p_value(table: list[list[int]]) -> Fraction:
    """Return the two-sided Freeman-Halton p-value in exact arithmetic.

    Exact ties count as no more probable, which is what the relative
    tolerance ``1e-7`` of the implementation achieves in floating point.

    Args:
        table: Table of counts.

    Returns:
        Exact p-value.

    References:
        [EQ-INF-FISHER-RXC-001], [REF-FREEMAN-HALTON-1951].

    """
    rows = [sum(row) for row in table]
    columns = [sum(column) for column in zip(*table, strict=True)]
    numerator = math.prod(math.factorial(r) for r in rows) * math.prod(
        math.factorial(c) for c in columns
    )
    total = math.factorial(sum(rows))

    def probability(cells: tuple[tuple[int, ...], ...] | list[list[int]]) -> Fraction:
        """Return the multiple hypergeometric probability of a table.

        References:
            [EQ-INF-FISHER-RXC-001]. No other equation is evaluated.

        """
        return Fraction(
            numerator,
            total * math.prod(math.factorial(x) for row in cells for x in row),
        )

    observed = probability(table)
    everything = [probability(t) for t in tables_with_margins(rows, columns)]
    assert sum(everything) == 1
    return sum((p for p in everything if p <= observed), Fraction(0))


def relative_error(value: float, reference: float) -> float:
    """Return ``|value - reference| / |reference|``.

    References:
        [REF-FREEMAN-HALTON-1951]. No equation is evaluated.

    """
    return abs(value - reference) / abs(reference)


class FisherRxCTests(unittest.TestCase):
    """Exact r x c Fisher test."""

    SMALL_TABLES = (
        [[2, 7, 1], [8, 2, 5]],
        [[5, 1, 0], [2, 4, 3], [0, 2, 6]],
        [[3, 0, 2], [1, 4, 0], [0, 1, 3], [2, 2, 1]],
        [[4, 1, 1, 0], [0, 3, 1, 2]],
        [[2, 2], [2, 2], [2, 2]],
    )

    def test_matches_brute_force_enumeration(self) -> None:
        """The network sum equals the exact rational sum over all tables.

        Oracle: every table with the observed margins listed and weighted
        in ``Fraction`` arithmetic.
        """
        for table in self.SMALL_TABLES:
            with self.subTest(table=table):
                expected = float(exact_p_value(table))
                got = fisher_exact_rxc_test(table)
                self.assertTrue(got.exact)
                self.assertLessEqual(relative_error(got.p_value, expected), 1e-12)

    def test_table_probability_is_exact(self) -> None:
        """``table_probability`` is ``prod r! prod c! / (N! prod x!)``.

        Oracle: the closed form in integer arithmetic.
        """
        table = [[5, 1, 0], [2, 4, 3], [0, 2, 6]]
        rows = [6, 9, 8]
        columns = [7, 7, 9]
        exact = Fraction(
            math.prod(map(math.factorial, rows))
            * math.prod(map(math.factorial, columns)),
            math.factorial(23) * math.prod(math.factorial(x) for r in table for x in r),
        )
        got = fisher_exact_rxc_test(table).table_probability
        self.assertLessEqual(relative_error(got, float(exact)), 1e-12)

    def test_two_by_two_equals_fisher_exact_test(self) -> None:
        """On 2 x 2 tables the r x c test is the two-sided 2 x 2 test.

        Oracle: ``fisher_exact_test``, a hypergeometric sum.
        """
        for table in (
            [[3, 1], [1, 3]],
            [[2, 10], [15, 3]],
            [[0, 5], [6, 1]],
            [[7, 7], [7, 7]],
        ):
            with self.subTest(table=table):
                self.assertLessEqual(
                    relative_error(
                        fisher_exact_rxc_test(table).p_value,
                        fisher_exact_test(table).p_value,
                    ),
                    1e-12,
                )

    def test_agrees_with_r(self) -> None:
        """Agreement with R's ``fisher.test`` on further tables.

        Oracle: R 4.6.2, ``fisher.test(t, workspace = 2e8)$p.value`` printed
        with ``%.17g`` for ``t <- rbind(c(8,3,2,0), c(1,6,4,2), c(2,1,5,9))``
        (0.00044796160719526101), ``rbind(c(12,8,5,3,1), c(2,5,9,10,14))``
        (6.4459483005124599e-05), ``rbind(c(60,45,30), c(40,55,70))``
        (0.0001104908204622428), ``rbind(c(3,5,2,1,0), c(2,3,4,3,1),
        c(1,2,3,5,4), c(0,1,2,4,6))`` (0.10491199331885238), and the 5 x 5
        ``rbind(c(4,2,1,0,0), c(2,4,2,1,0), c(1,2,4,2,1), c(0,1,2,4,2),
        c(0,0,1,2,4))`` (0.034900476483619665). FEXACT and the network sum
        add the terms in different orders, hence ``1e-10`` relative.
        """
        cases = (
            ([[8, 3, 2, 0], [1, 6, 4, 2], [2, 1, 5, 9]], 0.00044796160719526101),
            ([[12, 8, 5, 3, 1], [2, 5, 9, 10, 14]], 6.4459483005124599e-05),
            ([[60, 45, 30], [40, 55, 70]], 0.0001104908204622428),
            (
                [[3, 5, 2, 1, 0], [2, 3, 4, 3, 1], [1, 2, 3, 5, 4], [0, 1, 2, 4, 6]],
                0.10491199331885238,
            ),
            (
                [
                    [4, 2, 1, 0, 0],
                    [2, 4, 2, 1, 0],
                    [1, 2, 4, 2, 1],
                    [0, 1, 2, 4, 2],
                    [0, 0, 1, 2, 4],
                ],
                0.034900476483619665,
            ),
        )
        for table, reference in cases:
            with self.subTest(table=table):
                got = fisher_exact_rxc_test(table).p_value
                self.assertLessEqual(relative_error(got, reference), 1e-10)

    def test_invariant_under_permutation_and_transposition(self) -> None:
        """Reordering rows or columns, or transposing, leaves ``p`` unchanged.

        Oracle: the conditional law depends on the margins only as a
        multiset, and is symmetric in rows and columns.
        """
        table = [[3, 0, 2], [1, 4, 0], [0, 1, 3], [2, 2, 1]]
        reference = fisher_exact_rxc_test(table).p_value
        variants = (
            table[::-1],
            [row[::-1] for row in table],
            [list(column) for column in zip(*table, strict=True)],
            [table[2], table[0], table[3], table[1]],
        )
        for variant in variants:
            with self.subTest(variant=variant):
                got = fisher_exact_rxc_test(variant).p_value
                self.assertLessEqual(relative_error(got, reference), 1e-12)

    def test_empty_rows_and_columns_are_dropped(self) -> None:
        """Zero rows and columns do not change the result; the shape says so.

        Oracle: an empty margin has a single completion.
        """
        table = [[5, 0, 1, 0], [0, 0, 0, 0], [2, 0, 4, 3], [0, 0, 2, 6]]
        reduced = [[5, 1, 0], [2, 4, 3], [0, 2, 6]]
        got = fisher_exact_rxc_test(table)
        self.assertEqual(got.shape, (3, 3))
        self.assertEqual(got.p_value, fisher_exact_rxc_test(reduced).p_value)

    def test_monte_carlo_estimate(self) -> None:
        """The seeded estimate is reproducible, positive, and near the exact value.

        Oracle: the exact p-value, judged at four Monte Carlo standard
        errors; ``(1 + b)/(B + 1) >= 1/(B + 1)``.
        """
        table = [[3, 0, 2], [1, 4, 0], [0, 1, 3], [2, 2, 1]]
        exact = fisher_exact_rxc_test(table).p_value
        first = fisher_exact_rxc_test(table, simulate=True, replicates=4000, seed=7)
        again = fisher_exact_rxc_test(table, simulate=True, replicates=4000, seed=7)
        self.assertEqual(first, again)
        self.assertFalse(first.exact)
        self.assertEqual(first.evaluations, 4000)
        assert first.standard_error is not None
        self.assertLess(abs(first.p_value - exact), 4.0 * first.standard_error)
        extreme = fisher_exact_rxc_test(
            [[20, 0], [0, 20]], simulate=True, replicates=99, seed=1
        )
        self.assertGreaterEqual(extreme.p_value, 1.0 / 100.0)

    def test_work_limit_and_validation(self) -> None:
        """Too much work, ragged or degenerate tables, and bad counts raise.

        Oracle: the documented domain.
        """
        job = [[1, 3, 10, 6], [2, 3, 10, 7], [1, 6, 14, 12], [0, 1, 9, 11]]
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            fisher_exact_rxc_test(job, max_evaluations=1000)
        for bad in ([[1, 2], [3]], [[1, 2, 3]], [[0, 0], [1, 2]], [], [[]]):
            with self.subTest(table=bad), self.assertRaises(ValueError):
                fisher_exact_rxc_test(bad)
        with self.assertRaises(ValueError):
            fisher_exact_rxc_test([[1, -1], [2, 3]])
        for bad_count in (1.0, True, "2"):
            with self.subTest(count=bad_count), self.assertRaises(TypeError):
                fisher_exact_rxc_test([[bad_count, 1], [2, 3]])
        with self.assertRaises(ValueError):
            fisher_exact_rxc_test([[1, 2], [3, 4]], max_evaluations=0)
        with self.assertRaises(ValueError):
            fisher_exact_rxc_test([[1, 2], [3, 4]], simulate=True, replicates=0)


A = (2.9, 3.0, 2.5, 2.6, 3.2, 3.0)
B = (3.8, 2.7, 4.0, 2.4, 3.0)
C = (2.8, 3.4, 3.7, 2.2, 2.0, 4.1, 3.9)


class FlignerKilleenTests(unittest.TestCase):
    """Median-centred Fligner-Killeen test."""

    def test_agrees_with_r(self) -> None:
        """Agreement with R's ``fligner.test``.

        Oracle: R 4.6.2, ``fligner.test(y, g)`` for the three groups below
        (statistic 3.9884672248422199, p 0.13611793329597272) and
        ``fligner.test(list(c(1, 1, 2, 2, 2, 5, 9), c(3, 3, 3, 4, 10, 12)))``
        (statistic 0.3403973342704798, p 0.55959996280374213), printed with
        ``%.17g``.
        """
        result = fligner_killeen_test(A, B, C)
        self.assertEqual(result.df, (2.0,))
        self.assertEqual(result.sample_sizes, (6, 5, 7))
        self.assertLessEqual(
            relative_error(result.statistic, 3.9884672248422199), 1e-12
        )
        self.assertLessEqual(relative_error(result.p_value, 0.13611793329597272), 1e-10)
        tied = fligner_killeen_test((1, 1, 2, 2, 2, 5, 9), (3, 3, 3, 4, 10, 12))
        self.assertLessEqual(relative_error(tied.statistic, 0.3403973342704798), 1e-12)
        self.assertLessEqual(relative_error(tied.p_value, 0.55959996280374213), 1e-10)

    def test_invariant_under_group_shifts_and_rescaling(self) -> None:
        """Shifting each group or rescaling all data leaves ``X^2`` unchanged.

        Oracle: the statistic depends on the ranks of ``|x - med_i|`` only.
        Integer data, integer shifts, and a power-of-two scale keep every
        deviation exact, so ties are preserved.
        """
        first = (1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 9.0)
        second = (3.0, 3.0, 3.0, 4.0, 10.0, 12.0)
        third = (0.0, 7.0, 7.0, 8.0, 13.0)
        reference = fligner_killeen_test(first, second, third).statistic
        shifted = fligner_killeen_test(
            [x + 16.0 for x in first], [x - 8.0 for x in second], third
        ).statistic
        scaled = fligner_killeen_test(
            [0.25 * x for x in first],
            [0.25 * x for x in second],
            [0.25 * x for x in third],
        ).statistic
        self.assertEqual(shifted, reference)
        self.assertEqual(scaled, reference)

    def test_errors(self) -> None:
        """One group, an empty group, or equal deviations raise.

        Oracle: the documented domain (``V > 0``).
        """
        with self.assertRaises(ValueError):
            fligner_killeen_test(A)
        with self.assertRaises(ValueError):
            fligner_killeen_test(A, ())
        with self.assertRaises(ValueError):
            fligner_killeen_test((1.0, 3.0), (5.0, 7.0))


class WelchAnovaTests(unittest.TestCase):
    """Welch's one-way ANOVA."""

    def test_agrees_with_r(self) -> None:
        """Agreement with R's ``oneway.test(y ~ g)`` (``var.equal = FALSE``).

        Oracle: R 4.6.2 for the three groups below: F 0.69255499234997697,
        denominator df 7.6833039744413076, p 0.52901152154652153, printed
        with ``%.17g``.
        """
        result = welch_anova(A, B, C)
        self.assertEqual(result.df[0], 2.0)
        self.assertLessEqual(
            relative_error(result.statistic, 0.69255499234997697), 1e-12
        )
        self.assertLessEqual(relative_error(result.df[1], 7.6833039744413076), 1e-12)
        self.assertLessEqual(relative_error(result.p_value, 0.52901152154652153), 1e-10)

    def test_two_groups_reduce_to_the_welch_t_test(self) -> None:
        """For two groups ``F = t^2`` with the Welch-Satterthwaite df.

        Oracle: [EQ-INF-T-003] through ``two_sample_t_test``.
        """
        welch = welch_anova(A, C)
        t_test = two_sample_t_test(A, C)
        self.assertAlmostEqual(welch.statistic, t_test.statistic**2, places=12)
        self.assertAlmostEqual(welch.df[1], t_test.df, places=10)
        self.assertAlmostEqual(welch.p_value, t_test.p_value, places=10)

    def test_errors(self) -> None:
        """Too few groups, a singleton group, or a constant group raise.

        Oracle: the documented domain (``n_i >= 2``, ``s_i^2 > 0``).
        """
        with self.assertRaises(ValueError):
            welch_anova(A)
        with self.assertRaises(ValueError):
            welch_anova(A, (1.0,))
        with self.assertRaises(ValueError):
            welch_anova(A, (2.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
