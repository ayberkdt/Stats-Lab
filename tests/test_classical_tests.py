"""Tests for the k-sample, blocked, variance, normality, and post hoc tests.

Oracles:

* SciPy 1.14.1 computed offline (not a test dependency):
  ``scipy.stats.kruskal``, ``friedmanchisquare``, ``levene(center="mean")``
  and ``levene(center="median")``, ``bartlett``, ``shapiro`` (the Fortran
  ``swilk`` of AS R94), ``tukey_hsd`` with ``confidence_interval(0.95)``, and
  ``studentized_range.cdf`` / ``.ppf``;
* exact enumeration of every relabelling for the exact Kruskal-Wallis law;
* closed forms: the sign test's binomial tail, McNemar's statistic for
  Cochran's Q with two conditions, the ``k = 2`` studentized range
  (``P(|T_nu| <= q/sqrt 2)``), the ``n = 3`` Shapiro-Wilk law, and Dunn's
  statistic recomputed in exact rational arithmetic.
"""

from __future__ import annotations

import itertools
import math
import statistics
import unittest
from fractions import Fraction

from probstat.inference import (
    adjust_p_values,
    bartlett_test,
    brown_forsythe_test,
    cochran_q_test,
    dunn_test,
    friedman_test,
    kruskal_wallis_test,
    levene_test,
    mcnemar_test,
    one_way_anova,
    shapiro_wilk_coefficients,
    shapiro_wilk_test,
    sign_test,
    standard_normal_cdf,
    student_t_two_sided_p,
    studentized_range_cdf,
    studentized_range_quantile,
    studentized_range_sf,
    tukey_hsd_test,
)
from probstat.statistics import ranks

G1 = (2.9, 3.0, 2.5, 2.6, 3.2, 3.0)
G2 = (3.8, 2.7, 4.0, 2.4, 3.0)
G3 = (2.8, 3.4, 3.7, 2.2, 2.0, 4.1, 3.9)


def kruskal_h(groups: list[list[float]]) -> float:
    """Return the tie-corrected Kruskal-Wallis ``H`` computed from scratch."""
    pooled = [v for g in groups for v in g]
    n = len(pooled)
    r = ranks(pooled)
    total, start = 0.0, 0
    for g in groups:
        total += sum(r[start : start + len(g)]) ** 2 / len(g)
        start += len(g)
    ties = sum(t**3 - t for t in (pooled.count(v) for v in set(pooled)))
    return (12.0 / (n * (n + 1)) * total - 3 * (n + 1)) / (1 - ties / (n**3 - n))


class KruskalWallisTests(unittest.TestCase):
    """Kruskal-Wallis with the chi-square and exact permutation laws."""

    def test_chi_square_branch_matches_scipy(self) -> None:
        """``scipy.stats.kruskal(G1, G2, G3)``: H = 0.5640661238588668."""
        result = kruskal_wallis_test(G1, G2, G3, method="chi-square")
        self.assertAlmostEqual(result.statistic, 0.5640661238588668, places=12)
        self.assertAlmostEqual(result.p_value, 0.7542487472019979, places=12)
        self.assertEqual(result.df, (2.0,))
        self.assertFalse(result.exact)

    def test_exact_law_matches_brute_force_relabelling(self) -> None:
        """Every distinct labelling of the pooled data is enumerated directly."""
        cases = (
            [[1.0, 2.0, 3.5], [4.0, 5.0, 6.0], [3.5, 8.0, 9.0]],
            [[1.0, 1.0], [2.0, 3.0], [3.0, 4.0, 5.0]],
            [[0.3, 1.2, 2.2], [0.1, 0.5], [1.9, 2.8, 3.1, 4.0]],
        )
        for groups in cases:
            with self.subTest(groups=groups):
                pooled = [v for g in groups for v in g]
                labels = [i for i, g in enumerate(groups) for _ in g]
                observed = kruskal_h(groups)
                extreme = total = 0
                for labelling in set(itertools.permutations(labels)):
                    regrouped = [
                        [
                            v
                            for v, lab in zip(pooled, labelling, strict=True)
                            if lab == i
                        ]
                        for i in range(len(groups))
                    ]
                    total += 1
                    extreme += kruskal_h(regrouped) >= observed - 1e-9
                result = kruskal_wallis_test(*groups)
                self.assertTrue(result.exact)
                self.assertAlmostEqual(result.statistic, observed, places=12)
                self.assertAlmostEqual(result.p_value, extreme / total, places=14)

    def test_classic_three_by_three_critical_value(self) -> None:
        """Without ties, ``P(H >= 5.6) = 84/1680`` for sizes (3, 3, 3).

        The count comes from enumerating the 1680 rank assignments here.
        """
        # Rank sums 9, 12, 24: H = (12/90)(81 + 144 + 576)/3 - 30 = 5.6.
        groups = ([1.0, 2.0, 6.0], [3.0, 4.0, 5.0], [7.0, 8.0, 9.0])
        result = kruskal_wallis_test(*groups)
        self.assertAlmostEqual(result.statistic, 5.6, places=12)
        count = 0
        ranks_all = list(range(1, 10))
        for first in itertools.combinations(ranks_all, 3):
            rest = [r for r in ranks_all if r not in first]
            for second in itertools.combinations(rest, 3):
                third = [r for r in rest if r not in second]
                h = 12 / 90 * sum(sum(g) ** 2 / 3 for g in (first, second, third)) - 30
                count += h >= 5.6 - 1e-9
        self.assertEqual(count, 84)
        self.assertAlmostEqual(result.p_value, 84 / 1680, places=15)

    def test_auto_switches_to_chi_square_for_large_designs(self) -> None:
        """``N!/prod n_g!`` above ``max_exact`` uses the chi-square law."""
        result = kruskal_wallis_test(G1, G2, G3, max_exact=1000)
        self.assertFalse(result.exact)
        self.assertAlmostEqual(
            result.effect_size or 0.0, result.statistic / 17, places=14
        )

    def test_errors(self) -> None:
        """All-tied data and unknown methods are rejected."""
        with self.assertRaises(ValueError):
            kruskal_wallis_test([1.0, 1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            kruskal_wallis_test(G1, G2, method="normal")
        with self.assertRaises(ValueError):
            kruskal_wallis_test(G1)


class FriedmanAndSignTests(unittest.TestCase):
    """Blocked rank test and the sign test."""

    BLOCKS = (
        (31, 27, 24),
        (31, 28, 31),
        (45, 29, 46),
        (21, 18, 48),
        (42, 36, 46),
        (32, 17, 40),
        (27, 27, 30),
        (33, 20, 35),
    )

    def test_friedman_matches_scipy(self) -> None:
        """``scipy.stats.friedmanchisquare`` on the columns: Q = 9.8."""
        result = friedman_test(self.BLOCKS)
        self.assertAlmostEqual(result.statistic, 9.8, places=12)
        self.assertAlmostEqual(result.p_value, 0.007446583070924338, places=12)
        self.assertEqual(result.sample_sizes, (8, 3))
        self.assertAlmostEqual(result.effect_size or 0.0, 9.8 / 16, places=12)

    def test_friedman_errors(self) -> None:
        """Ragged and fully tied tables are rejected."""
        with self.assertRaises(ValueError):
            friedman_test([(1, 2, 3), (1, 2)])
        with self.assertRaises(ValueError):
            friedman_test([(1, 1), (2, 2)])

    def test_sign_test_exact_binomial(self) -> None:
        """Nine of ten non-zero differences positive: ``p = 22/1024``."""
        differences = (1, 2, 3, -1, 4, 5, 6, 7, 8, 9, 0)
        result = sign_test(differences)
        self.assertEqual((result.successes, result.trials), (9, 10))
        self.assertAlmostEqual(result.p_value, 22 / 1024, places=15)
        self.assertEqual(result.method, "exact sign test")
        upper = sign_test(differences, alternative="greater")
        self.assertAlmostEqual(upper.p_value, 11 / 1024, places=15)
        paired = sign_test([5, 6, 7], [4, 6, 9], null_value=0.0)
        self.assertEqual((paired.successes, paired.trials), (1, 2))
        with self.assertRaises(ValueError):
            sign_test([0.0, 0.0])


class VarianceTests(unittest.TestCase):
    """Levene, Brown-Forsythe, and Bartlett."""

    def test_levene_and_brown_forsythe_match_scipy(self) -> None:
        """``scipy.stats.levene`` with ``center="mean"`` and ``"median"``."""
        levene = levene_test(G1, G2, G3)
        self.assertAlmostEqual(levene.statistic, 5.94989778470858, places=11)
        self.assertAlmostEqual(levene.p_value, 0.012518926701516109, places=12)
        self.assertEqual(levene.df, (2.0, 15.0))
        bf = brown_forsythe_test(G1, G2, G3)
        self.assertAlmostEqual(bf.statistic, 2.5148455420498625, places=11)
        self.assertAlmostEqual(bf.p_value, 0.11432140636114904, places=12)
        self.assertEqual(levene_test(G1, G2, G3, center="median"), bf)

    def test_levene_is_anova_of_absolute_deviations(self) -> None:
        """The identity that defines the test."""
        deviations = [[abs(x - statistics.fmean(g)) for x in g] for g in (G1, G2, G3)]
        self.assertAlmostEqual(
            levene_test(G1, G2, G3).statistic,
            one_way_anova(*deviations).statistic,
            places=14,
        )

    def test_bartlett_matches_scipy(self) -> None:
        """``scipy.stats.bartlett``: T = 5.222373527294523."""
        result = bartlett_test(G1, G2, G3)
        self.assertAlmostEqual(result.statistic, 5.222373527294523, places=11)
        self.assertAlmostEqual(result.p_value, 0.07344732740254055, places=12)

    def test_errors(self) -> None:
        """Groups of size one, zero variance, and unknown centres are rejected."""
        with self.assertRaises(ValueError):
            bartlett_test([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            bartlett_test([1.0, 1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            levene_test(G1, G2, center="trimmed")


def sample(n: int) -> list[float]:
    """Return a deterministic, roughly symmetric sample of size ``n``."""
    return [((37 * i) % 101) / 10 + ((i * i) % 13) / 7 for i in range(n)]


def skewed(n: int) -> list[float]:
    """Return a deterministic right-skewed sample of size ``n``."""
    return [((i * i) % 29) ** 2 / 50 for i in range(n)]


class ShapiroWilkTests(unittest.TestCase):
    """Shapiro-Wilk W and p-value against ``scipy.stats.shapiro`` (AS R94)."""

    ORACLE = {
        ("sample", 4): (0.9729718562523806, 0.8597913352896351),
        ("skewed", 4): (0.765901584891761, 0.05380736854690237),
        ("sample", 5): (0.9978043368592836, 0.9985784690232393),
        ("skewed", 5): (0.7590006290465147, 0.03598307589359618),
        ("sample", 6): (0.9826981536527595, 0.9641567799851903),
        ("skewed", 6): (0.7571879442439031, 0.023401922768781894),
        ("sample", 7): (0.9693981269215104, 0.8940797045331816),
        ("skewed", 7): (0.7202885386696387, 0.006147231236241328),
        ("sample", 11): (0.963702097457224, 0.8168589188973855),
        ("skewed", 11): (0.8326101596938222, 0.025263237980915784),
        ("sample", 12): (0.9565866906638542, 0.7342250853463603),
        ("skewed", 12): (0.8055184795735092, 0.01083253397050119),
        ("sample", 20): (0.9500957313729549, 0.36854516947334837),
        ("skewed", 20): (0.8694878832623553, 0.011508219018489826),
        ("sample", 50): (0.9653906185834811, 0.14942695849488863),
        ("skewed", 50): (0.8613060157310101, 3.176562027816284e-05),
        ("sample", 200): (0.9664450255011674, 0.00010536152115475008),
        ("skewed", 200): (0.8567362378868185, 9.24026771661358e-13),
        ("sample", 1000): (0.9675947434477987, 3.940472097801261e-14),
        ("skewed", 1000): (0.8535123092467605, 1.8569320741516236e-29),
    }

    def test_matches_scipy_swilk(self) -> None:
        """W to ``1e-8`` and p to ``1e-6`` relative for ``n = 4 ... 1000``.

        SciPy's Fortran ``swilk`` keeps some AS R94 constants in single
        precision, which explains differences near ``1e-9`` in ``W``.
        """
        for (kind, n), (w, p) in self.ORACLE.items():
            with self.subTest(kind=kind, n=n):
                data = sample(n) if kind == "sample" else skewed(n)
                result = shapiro_wilk_test(data)
                self.assertAlmostEqual(result.statistic, w, delta=1e-8)
                self.assertAlmostEqual(result.p_value, p, delta=1e-6 * p)

    def test_three_observations_closed_form(self) -> None:
        """``n = 3``: ``W = 27/28`` for (1, 2, 4) and the exact arcsine law."""
        result = shapiro_wilk_test([4.0, 1.0, 2.0])
        self.assertAlmostEqual(result.statistic, 27 / 28, places=14)
        expected = 6 / math.pi * (math.asin(math.sqrt(27 / 28)) - math.pi / 3)
        self.assertAlmostEqual(result.p_value, expected, places=14)
        self.assertAlmostEqual(result.p_value, 0.6368868450289689, places=12)
        self.assertTrue(result.exact)

    def test_coefficients_are_normalized(self) -> None:
        """``2 sum a_i^2 = 1`` and the coefficients decrease."""
        for n in (3, 4, 5, 6, 10, 51, 500, 5000):
            with self.subTest(n=n):
                a = shapiro_wilk_coefficients(n)
                self.assertEqual(len(a), n // 2)
                self.assertAlmostEqual(2.0 * sum(v * v for v in a), 1.0, places=12)
                self.assertTrue(all(x >= y for x, y in itertools.pairwise(a)))

    def test_errors(self) -> None:
        """Sizes outside ``[3, 5000]`` and constant samples are rejected."""
        with self.assertRaises(ValueError):
            shapiro_wilk_test([1.0, 2.0])
        with self.assertRaises(ValueError):
            shapiro_wilk_test([2.0] * 10)
        with self.assertRaises(ValueError):
            shapiro_wilk_coefficients(5001)


class CochranQTests(unittest.TestCase):
    """Cochran's Q for matched binary data."""

    def test_two_conditions_equal_mcnemar_without_correction(self) -> None:
        """For ``k = 2``, ``Q = (b - c)^2/(b + c)``."""
        rows = [(1, 1)] * 5 + [(1, 0)] * 9 + [(0, 1)] * 3 + [(0, 0)] * 4
        q = cochran_q_test(rows)
        mcnemar = mcnemar_test(
            [[5, 9], [3, 4]], exact=False, continuity_correction=False
        )
        self.assertAlmostEqual(q.statistic, 36 / 12, places=14)
        self.assertAlmostEqual(q.statistic, mcnemar.statistic, places=14)
        self.assertAlmostEqual(q.p_value, mcnemar.p_value, places=14)

    def test_hand_computed_three_conditions(self) -> None:
        """Columns 6, 5, 1 and rows (2, 1, 3, 1, 2, 1, 0, 2): ``Q = 7``."""
        table = [
            (1, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, 0),
            (0, 0, 0),
            (1, 1, 0),
        ]
        # (k - 1)(k sum C^2 - T^2)/(k T - sum R^2) = 2 (3 62 - 144)/(36 - 24).
        result = cochran_q_test(table)
        self.assertAlmostEqual(
            result.statistic, 2 * (3 * 62 - 144) / (36 - 24), places=14
        )
        self.assertAlmostEqual(result.p_value, math.exp(-3.5), places=14)

    def test_errors(self) -> None:
        """Non-binary entries and uninformative tables are rejected."""
        with self.assertRaises(ValueError):
            cochran_q_test([(1, 2), (0, 1)])
        with self.assertRaises(ValueError):
            cochran_q_test([(1, 1), (0, 0)])


class StudentizedRangeTests(unittest.TestCase):
    """The studentized range distribution and Tukey's HSD."""

    def test_two_means_closed_forms(self) -> None:
        """``k = 2``: ``2 Phi(q/sqrt 2) - 1`` and ``1 - P(|T_nu| > q/sqrt 2)``."""
        for q in (0.2, 1.0, 2.5, 4.0, 7.0):
            with self.subTest(q=q):
                infinite = studentized_range_cdf(q, 2, math.inf)
                self.assertAlmostEqual(
                    infinite,
                    2.0 * standard_normal_cdf(q / math.sqrt(2.0)) - 1.0,
                    delta=1e-13,
                )
                for df in (1, 3, 12, 1000):
                    finite = studentized_range_cdf(q, 2, df)
                    closed = 1.0 - student_t_two_sided_p(q / math.sqrt(2.0), df)
                    self.assertAlmostEqual(finite, closed, delta=1e-12)

    def test_matches_scipy(self) -> None:
        """``scipy.stats.studentized_range.cdf`` and ``.ppf(0.95)``."""
        cdf = {
            (3, 10.0, 3.5): 0.9228966891615895,
            (4, 20.0, 2.0): 0.4945596545878862,
            (5, 12.0, 4.5): 0.9495348753910773,
            (10, 60.0, 4.6): 0.9455385040562583,
            (20, 1.0, 15.0): 0.8035957591498621,
            (3, 2.0, 1.2): 0.2837229716253325,
            (6, math.inf, 4.0): 0.9469238843020045,
        }
        for (k, df, q), value in cdf.items():
            with self.subTest(k=k, df=df, q=q):
                self.assertAlmostEqual(
                    studentized_range_cdf(q, k, df), value, delta=1e-12
                )
                self.assertAlmostEqual(
                    studentized_range_sf(q, k, df), 1.0 - value, delta=1e-12
                )
        for (k, df), value in {
            (3, 10.0): 3.876776750013159,
            (4, 20.0): 3.9582935609453833,
            (5, 12.0): 4.507709919837381,
        }.items():
            with self.subTest(k=k, df=df):
                self.assertAlmostEqual(
                    studentized_range_quantile(0.95, k, df), value, delta=1e-9
                )

    def test_errors(self) -> None:
        """Invalid groups and degrees of freedom are rejected."""
        with self.assertRaises(ValueError):
            studentized_range_cdf(1.0, 1, 10)
        with self.assertRaises(ValueError):
            studentized_range_cdf(1.0, 3, 0.5)
        with self.assertRaises(TypeError):
            studentized_range_cdf(1.0, 2.5, 10)  # type: ignore[arg-type]
        self.assertEqual(studentized_range_cdf(-1.0, 3, 10), 0.0)

    def test_tukey_hsd_matches_scipy(self) -> None:
        """``scipy.stats.tukey_hsd(G1, G2, G3)`` p-values and 95% intervals."""
        expected = {
            (0, 1): (
                -0.31333333333333346,
                0.7154889992222937,
                -1.345395123688815,
                0.7187284570221482,
            ),
            (0, 2): (
                -0.29047619047618944,
                0.7112714727681041,
                -1.2387134480508886,
                0.6577610670985096,
            ),
            (1, 2): (
                0.02285714285714402,
                0.9980509683267187,
                -0.9751337662967698,
                1.0208480520110579,
            ),
        }
        result = tukey_hsd_test(G1, G2, G3)
        self.assertEqual(result.df, 15.0)
        for (i, j), (difference, p, low, high) in expected.items():
            with self.subTest(pair=(i, j)):
                item = result.comparison(i, j)
                assert item.confidence_interval is not None
                self.assertAlmostEqual(item.difference, difference, places=12)
                self.assertAlmostEqual(item.p_value, p, delta=1e-10)
                self.assertAlmostEqual(item.confidence_interval.lower, low, delta=1e-9)
                self.assertAlmostEqual(item.confidence_interval.upper, high, delta=1e-9)
                self.assertFalse(item.rejected)


class DunnTests(unittest.TestCase):
    """Dunn's pairwise rank comparisons."""

    def test_statistics_in_exact_arithmetic(self) -> None:
        """Mean ranks and tie-corrected variance recomputed with fractions."""
        groups = (G1, G2, G3)
        pooled = [v for g in groups for v in g]
        n = len(pooled)
        order = sorted(pooled)
        rank = {
            v: Fraction(2 * order.index(v) + order.count(v) + 1, 2) for v in set(pooled)
        }
        ties = sum(order.count(v) ** 3 - order.count(v) for v in set(pooled))
        variance = Fraction(n * (n + 1), 12) - Fraction(ties, 12 * (n - 1))
        means = [sum(rank[v] for v in g) / len(g) for g in groups]
        result = dunn_test(*groups, adjustment="bonferroni")
        raw = []
        for item in result.comparisons:
            i, j = item.first, item.second
            z = float(means[i] - means[j]) / math.sqrt(
                float(
                    variance
                    * (Fraction(1, len(groups[i])) + Fraction(1, len(groups[j])))
                )
            )
            raw.append(2.0 * (1.0 - standard_normal_cdf(abs(z))))
            self.assertAlmostEqual(item.statistic, z, places=13)
            self.assertAlmostEqual(item.p_value, raw[-1], places=13)
        assert result.adjustment is not None
        self.assertEqual(
            result.adjustment.adjusted_p_values,
            adjust_p_values(raw, method="bonferroni").adjusted_p_values,
        )
        self.assertEqual(result.method, "Dunn test (bonferroni adjustment)")

    def test_errors(self) -> None:
        """All-tied data are rejected."""
        with self.assertRaises(ValueError):
            dunn_test([1.0, 1.0], [1.0])


if __name__ == "__main__":
    unittest.main()
