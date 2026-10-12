"""Tests for two-by-two effect measures, McNemar's test, and table residuals.

Oracle values were computed independently with scipy 1.14
(``scipy.stats.contingency.relative_risk``, ``odds_ratio(kind="sample")``,
``binomtest``, ``expected_freq``) and are hard-coded so that the test suite
keeps zero dependencies.
"""

from __future__ import annotations

import math
import unittest

from probstat.inference import (
    chi_square_independence_test,
    contingency_residuals,
    mcnemar_test,
    two_by_two_measures,
)


class TwoByTwoMeasureTests(unittest.TestCase):
    """Risk difference, risk ratio, and odds ratio with their intervals."""

    def test_katz_and_woolf_intervals_match_scipy(self) -> None:
        """RR 2.571 (0.98975, 6.68075); OR 3.2 (1.01766, 10.0623)."""
        result = two_by_two_measures([[12, 30], [5, 40]])
        self.assertAlmostEqual(result.risk_ratio, 18.0 / 7.0, places=14)
        assert result.risk_ratio_interval is not None
        self.assertAlmostEqual(
            result.risk_ratio_interval.lower, 0.9897453238621571, places=12
        )
        self.assertAlmostEqual(
            result.risk_ratio_interval.upper, 6.680753865203489, places=11
        )
        self.assertAlmostEqual(result.odds_ratio, 3.2, places=14)
        self.assertAlmostEqual(
            result.odds_ratio_interval.lower, 1.0176617269350616, places=12
        )
        self.assertAlmostEqual(
            result.odds_ratio_interval.upper, 10.062282710425082, places=11
        )
        self.assertFalse(result.corrected)

    def test_newcombe_interval_from_wilson_limits(self) -> None:
        """The hybrid score interval follows from the two Wilson intervals."""
        result = two_by_two_measures([[56, 14], [48, 32]])
        z = 1.959963984540054

        def wilson(x: int, n: int) -> tuple[float, float]:
            """Return the Wilson score interval written out by hand."""
            p = x / n
            centre = (p + z * z / (2 * n)) / (1 + z * z / n)
            half = (
                z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            )
            return centre - half, centre + half

        l1, u1 = wilson(56, 70)
        l2, u2 = wilson(48, 80)
        p1, p2 = 0.8, 0.6
        self.assertAlmostEqual(result.risk_difference, 0.2, places=14)
        self.assertAlmostEqual(
            result.risk_difference_interval.lower,
            0.2 - math.hypot(p1 - l1, u2 - p2),
            places=12,
        )
        self.assertAlmostEqual(
            result.risk_difference_interval.upper,
            0.2 + math.hypot(u1 - p1, p2 - l2),
            places=12,
        )
        self.assertAlmostEqual(result.risk_difference_interval.lower, 0.0524, places=4)
        self.assertAlmostEqual(result.risk_difference_interval.upper, 0.3339, places=4)

    def test_zero_cells(self) -> None:
        """A zero cell triggers the 0.5 correction for the odds ratio only."""
        result = two_by_two_measures([[0, 10], [5, 5]])
        self.assertTrue(result.corrected)
        self.assertEqual(result.risk_ratio, 0.0)
        self.assertIsNone(result.risk_ratio_interval)
        self.assertAlmostEqual(result.odds_ratio, (0.5 * 5.5) / (10.5 * 5.5), places=14)
        self.assertLess(result.risk_difference_interval.upper, 0.0)

    def test_invalid_tables(self) -> None:
        """Non-2x2 shapes, fractional counts, and empty rows are rejected."""
        with self.assertRaises(ValueError):
            two_by_two_measures([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            two_by_two_measures([[1.5, 2], [3, 4]])
        with self.assertRaises(ValueError):
            two_by_two_measures([[0, 0], [3, 4]])
        with self.assertRaises(TypeError):
            two_by_two_measures([[True, 2], [3, 4]])


class McNemarTests(unittest.TestCase):
    """Exact and asymptotic McNemar tests."""

    def test_exact_matches_binomial(self) -> None:
        """B = 7, c = 2: two-sided exact p = 0.1796875 (= 46/256)."""
        result = mcnemar_test([[10, 7], [2, 30]])
        self.assertEqual(result.method, "exact")
        self.assertAlmostEqual(result.p_value, 46.0 / 256.0, places=14)
        self.assertAlmostEqual(result.odds_ratio, 3.5, places=14)

    def test_asymptotic_with_and_without_correction(self) -> None:
        """B = 30, c = 12: corrected X^2 = 289/42, p = 0.0087119129623796."""
        corrected = mcnemar_test([[10, 30], [12, 30]])
        self.assertAlmostEqual(corrected.statistic, 289.0 / 42.0, places=14)
        self.assertAlmostEqual(corrected.p_value, 0.008711912962379598, places=13)
        plain = mcnemar_test([[10, 30], [12, 30]], continuity_correction=False)
        self.assertAlmostEqual(plain.statistic, 324.0 / 42.0, places=14)
        self.assertLess(plain.p_value, corrected.p_value)

    def test_no_discordant_pairs(self) -> None:
        """Without discordant pairs there is no evidence: p = 1."""
        result = mcnemar_test([[5, 0], [0, 7]])
        self.assertEqual(result.p_value, 1.0)
        self.assertTrue(math.isnan(result.statistic))


class ResidualTests(unittest.TestCase):
    """Expected counts and residuals under independence."""

    def test_pearson_residuals_sum_to_chi_square(self) -> None:
        """Squared Pearson residuals add up to the Pearson statistic."""
        table = [[10, 20, 30], [25, 15, 5]]
        residuals = contingency_residuals(table)
        total = math.fsum(r * r for row in residuals.pearson for r in row)
        test = chi_square_independence_test(table)
        self.assertAlmostEqual(total, test.statistic, places=12)

    def test_adjusted_residuals_match_haberman_formula(self) -> None:
        """Largest |adjusted residual| 4.1833, in column 0."""
        residuals = contingency_residuals([[10, 20, 30], [25, 15, 5]])
        row, column, value = residuals.largest_adjusted()
        # In a two-row table the two residuals of a column are exact
        # opposites, so the first of the tied cells is reported.
        self.assertEqual(column, 0)
        self.assertAlmostEqual(abs(value), 4.183300132670378, places=12)
        self.assertEqual(residuals.adjusted[0][0], -residuals.adjusted[1][0])
        for row_values in residuals.row_proportions:
            self.assertAlmostEqual(math.fsum(row_values), 1.0, places=14)
        self.assertAlmostEqual(residuals.expected[0][0], 60 * 35 / 105, places=14)


if __name__ == "__main__":
    unittest.main()


class LargeCountTests(unittest.TestCase):
    """Huge counts keep exact integers and exact residual arithmetic."""

    def test_adjusted_residuals_dominant_cell(self) -> None:
        """[[1e15, 1], [1, 1]]: all four |residuals| equal 15811388.300841888."""
        residuals = contingency_residuals([[1e15, 1], [1, 1]])
        for row in residuals.adjusted:
            for value in row:
                self.assertAlmostEqual(abs(value) / 15811388.300841888, 1.0, places=12)

    def test_counts_above_two_to_the_53_stay_exact(self) -> None:
        """Integer counts are not rounded through float."""
        result = two_by_two_measures([[2**53 + 1, 1], [1, 1]])
        self.assertEqual(result.table[0], 2**53 + 1)
