"""Tests for classical hypothesis tests, resampling, power, and multiplicity.

Reference numbers quoted in comments come from three kinds of source:
hand formulas recomputed inside the test, brute-force enumeration oracles
implemented inside the test, and published or cross-checked values. The
cross-checks were made once against SciPy 1.14.1 (not a runtime or test
dependency) and are noted where used.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
import unittest
import warnings

from probstat.foundations import RandomStream
from probstat.inference import (
    SmallExpectedCountWarning,
    adjust_p_values,
    binomial_test,
    bootstrap,
    bootstrap_confidence_interval,
    chi_square_goodness_of_fit,
    chi_square_independence_test,
    chi_square_sf,
    clopper_pearson_interval,
    cohens_d,
    f_sf,
    fisher_exact_test,
    hedges_correction,
    hedges_g,
    jackknife,
    mann_whitney_u_distribution,
    mann_whitney_u_test,
    noncentral_t_cdf,
    normal_mean_confidence_interval,
    one_sample_t_test,
    one_way_anova,
    paired_t_test,
    permutation_test,
    required_sample_size,
    signed_rank_distribution,
    student_t_cdf,
    student_t_mean_confidence_interval,
    student_t_quantile,
    student_t_sf,
    student_t_two_sided_p,
    t_test_power,
    tost_equivalence_test,
    two_sample_t_test,
    welch_satterthwaite_df,
    wilcoxon_signed_rank_test,
    z_test_power,
)

# Cushny-Peebles sleep data used by Student (1908); R's ``sleep`` data set.
SLEEP_GROUP_1 = (0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0)
SLEEP_GROUP_2 = (1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)


def _normal_cdf(z: float) -> float:
    """Return the standard normal CDF for test oracles."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


class SamplingDistributionTests(unittest.TestCase):
    """Check tail functions against closed forms and identities."""

    def test_chi_square_two_df_is_exponential(self) -> None:
        """For two degrees of freedom P(X >= x) = exp(-x/2)."""
        for x in (0.1, 1.0, 5.0, 30.0):
            self.assertAlmostEqual(chi_square_sf(x, 2), math.exp(-x / 2), places=13)
        self.assertEqual(chi_square_sf(-1.0, 3), 1.0)

    def test_f_with_one_numerator_df_matches_squared_t(self) -> None:
        """If T ~ t_nu then T^2 ~ F(1, nu), so tails agree."""
        for t, nu in ((1.3, 4.0), (2.7, 17.5), (0.2, 1.0)):
            self.assertAlmostEqual(
                f_sf(t * t, 1, nu), student_t_two_sided_p(t, nu), places=12
            )

    def test_student_t_tail_symmetry_and_cauchy_case(self) -> None:
        """One degree of freedom is Cauchy: P(T >= t) = 1/2 - atan(t)/pi."""
        for t in (-3.0, -0.5, 0.0, 0.7, 4.0):
            self.assertAlmostEqual(
                student_t_sf(t, 1.0), 0.5 - math.atan(t) / math.pi, places=12
            )
            self.assertAlmostEqual(
                student_t_sf(t, 6.0) + student_t_sf(-t, 6.0), 1.0, places=13
            )

    def test_noncentral_t_reduces_and_matches_simulation(self) -> None:
        """Zero noncentrality is central; a seeded simulation agrees."""
        self.assertAlmostEqual(
            noncentral_t_cdf(1.1, 7.0, 0.0), student_t_cdf(1.1, 7.0), places=13
        )
        # Cross-check (SciPy 1.14.1 nct.cdf(1.97, 126, 2.828)) = 0.19617542660433.
        self.assertAlmostEqual(
            noncentral_t_cdf(1.97, 126.0, 2.828), 0.19617542660433, places=9
        )
        generator = random.Random(4242)
        nu, delta, t, draws = 5, 1.5, 2.0, 20000
        hits = 0
        for _ in range(draws):
            chi = sum(generator.gauss(0.0, 1.0) ** 2 for _ in range(nu))
            if (generator.gauss(0.0, 1.0) + delta) / math.sqrt(chi / nu) <= t:
                hits += 1
        probability = noncentral_t_cdf(t, nu, delta)
        tolerance = 4.0 * math.sqrt(probability * (1 - probability) / draws)
        self.assertLess(abs(hits / draws - probability), tolerance)
        with self.assertRaises(ValueError):
            noncentral_t_cdf(1.0, 0.5, 1.0)


class TTestTests(unittest.TestCase):
    """Validate t statistics, degrees of freedom, and p-values."""

    def test_welch_matches_published_sleep_values(self) -> None:
        """R ``t.test(extra ~ group, sleep)``: t=-1.8608, df=17.776, p=0.07939."""
        result = two_sample_t_test(SLEEP_GROUP_1, SLEEP_GROUP_2)
        self.assertAlmostEqual(result.statistic, -1.8608, places=4)
        self.assertAlmostEqual(result.df, 17.776, places=3)
        self.assertAlmostEqual(result.p_value, 0.07939, places=5)
        self.assertEqual(result.method, "Welch two-sample t test")
        # Hand formula for the statistic.
        v1 = statistics.variance(SLEEP_GROUP_1)
        v2 = statistics.variance(SLEEP_GROUP_2)
        hand = (
            statistics.fmean(SLEEP_GROUP_1) - statistics.fmean(SLEEP_GROUP_2)
        ) / math.sqrt(v1 / 10 + v2 / 10)
        self.assertAlmostEqual(result.statistic, hand, places=12)

    def test_welch_satterthwaite_formula(self) -> None:
        """The df formula equals its hand evaluation and respects its bounds."""
        v1, n1, v2, n2 = 4.0, 8, 25.0, 13
        a, b = v1 / n1, v2 / n2
        hand = (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))
        df = welch_satterthwaite_df(v1, n1, v2, n2)
        self.assertAlmostEqual(df, hand, places=12)
        self.assertGreaterEqual(df, min(n1, n2) - 1)
        self.assertLessEqual(df, n1 + n2 - 2)
        # Equal variances and sizes give the pooled df exactly.
        self.assertAlmostEqual(welch_satterthwaite_df(3.0, 9, 3.0, 9), 16.0)
        with self.assertRaises(ValueError):
            welch_satterthwaite_df(0.0, 5, 0.0, 5)

    def test_pooled_and_paired_sleep_values(self) -> None:
        """Paired: t=-4.0621, df=9, p=0.002833 (R ``t.test(paired=TRUE)``)."""
        paired = paired_t_test(SLEEP_GROUP_1, SLEEP_GROUP_2)
        self.assertAlmostEqual(paired.statistic, -4.0621, places=4)
        self.assertEqual(paired.df, 9.0)
        self.assertAlmostEqual(paired.p_value, 0.002833, places=6)
        # R reports the 95% interval (-2.4598858, -0.7001142).
        self.assertAlmostEqual(paired.confidence_interval.lower, -2.4598858, places=6)
        self.assertAlmostEqual(paired.confidence_interval.upper, -0.7001142, places=6)
        pooled = two_sample_t_test(
            SLEEP_GROUP_1, SLEEP_GROUP_2, equal_variances=True, alternative="less"
        )
        self.assertEqual(pooled.df, 18.0)
        # Equal sizes make pooled and Welch statistics identical; one-sided
        # p-value cross-checked with SciPy 1.14.1: 0.03959335710796914.
        self.assertAlmostEqual(pooled.statistic, -1.8608134674868526, places=12)
        self.assertAlmostEqual(pooled.p_value, 0.03959335710796914, places=12)

    def test_one_sample_interval_and_alternatives(self) -> None:
        """One-sided p-values sum to one and the interval is the t interval."""
        values = (5.1, 4.9, 5.6, 5.8, 6.0, 5.3, 4.7, 5.5)
        greater = one_sample_t_test(values, null_mean=5.0, alternative="greater")
        less = one_sample_t_test(values, null_mean=5.0, alternative="less")
        both = one_sample_t_test(values, null_mean=5.0)
        self.assertAlmostEqual(greater.p_value + less.p_value, 1.0, places=12)
        self.assertAlmostEqual(both.p_value, 2 * min(greater.p_value, less.p_value))
        interval = student_t_mean_confidence_interval(values)
        self.assertAlmostEqual(both.confidence_interval.lower, interval.lower)
        self.assertAlmostEqual(both.confidence_interval.upper, interval.upper)
        mean, sd = statistics.fmean(values), statistics.stdev(values)
        self.assertAlmostEqual(both.effect_size, (mean - 5.0) / sd)

    def test_invalid_inputs_raise(self) -> None:
        """Degenerate samples and labels are rejected explicitly."""
        with self.assertRaises(ValueError):
            one_sample_t_test([1.0])
        with self.assertRaises(ValueError):
            one_sample_t_test([2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            one_sample_t_test([1.0, 2.0], alternative="bigger")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            two_sample_t_test([1.0, 1.0], [2.0, 2.0])
        with self.assertRaises(TypeError):
            two_sample_t_test([1.0, "a"], [2.0, 3.0])  # type: ignore[list-item]


class ConfidenceIntervalTests(unittest.TestCase):
    """Student-t and Clopper-Pearson intervals."""

    def test_student_t_coverage_simulation(self) -> None:
        """Seeded coverage of the t interval is near 95%; z undercovers at n=6."""
        generator = random.Random(97)
        replicates, n, true_mean = 2000, 6, 3.0
        t_hits = z_hits = 0
        for _ in range(replicates):
            sample = [generator.gauss(true_mean, 2.0) for _ in range(n)]
            t_interval = student_t_mean_confidence_interval(sample)
            z_interval = normal_mean_confidence_interval(sample)
            t_hits += t_interval.lower <= true_mean <= t_interval.upper
            z_hits += z_interval.lower <= true_mean <= z_interval.upper
        tolerance = 3.0 * math.sqrt(0.95 * 0.05 / replicates)
        self.assertLess(abs(t_hits / replicates - 0.95), tolerance)
        # The z interval with estimated sigma has true coverage near 0.89 here.
        self.assertLess(z_hits / replicates, 0.93)

    def test_clopper_pearson_tails_and_boundaries(self) -> None:
        """Endpoints solve the exact binomial tail equations."""
        x, n = 7, 20
        interval = clopper_pearson_interval(x, n)

        def tail_at_least(p: float) -> float:
            """Return P(X >= x) by direct summation."""
            return sum(
                math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x, n + 1)
            )

        def tail_at_most(p: float) -> float:
            """Return P(X <= x) by direct summation."""
            return sum(
                math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1)
            )

        self.assertAlmostEqual(tail_at_least(interval.lower), 0.025, places=10)
        self.assertAlmostEqual(tail_at_most(interval.upper), 0.025, places=10)
        zero = clopper_pearson_interval(0, 10)
        self.assertEqual(zero.lower, 0.0)
        self.assertAlmostEqual(zero.upper, 1.0 - 0.025 ** (1 / 10), places=10)
        with self.assertRaises(ValueError):
            clopper_pearson_interval(11, 10)
        with self.assertRaises(ValueError):
            student_t_mean_confidence_interval([1.0])


class ContingencyAndCountTests(unittest.TestCase):
    """Chi-square, Fisher exact, and binomial tests."""

    def test_mendel_goodness_of_fit(self) -> None:
        """Mendel's peas 315/108/101/32 vs 9:3:3:1: X^2 = 0.470, p = 0.925."""
        observed = (315, 108, 101, 32)
        probabilities = (9 / 16, 3 / 16, 3 / 16, 1 / 16)
        result = chi_square_goodness_of_fit(observed, probabilities)
        total = sum(observed)
        hand = sum(
            (o - total * p) ** 2 / (total * p)
            for o, p in zip(observed, probabilities, strict=True)
        )
        self.assertAlmostEqual(result.statistic, hand, places=12)
        self.assertAlmostEqual(result.statistic, 0.470, places=3)
        self.assertEqual(result.df, 3)
        self.assertAlmostEqual(result.p_value, 0.9254, places=4)
        self.assertFalse(result.small_expected_counts)

    def test_party_identification_independence(self) -> None:
        """Gender x party identification table: X^2 = 30.07 on 2 df (hand + SciPy)."""
        table = ((762, 327, 468), (484, 239, 477))
        result = chi_square_independence_test(table)
        rows = [sum(row) for row in table]
        columns = [sum(column) for column in zip(*table, strict=True)]
        n = sum(rows)
        hand = sum(
            (table[i][j] - rows[i] * columns[j] / n) ** 2 / (rows[i] * columns[j] / n)
            for i in range(2)
            for j in range(3)
        )
        self.assertAlmostEqual(result.statistic, hand, places=10)
        self.assertAlmostEqual(result.statistic, 30.07, places=2)
        self.assertEqual(result.df, 2)
        # Cross-check (SciPy 1.14.1): p = 2.9535891832117595e-07, V = 0.104436.
        self.assertAlmostEqual(result.p_value / 2.9535891832117595e-07, 1.0, places=9)
        self.assertAlmostEqual(result.effect_size, math.sqrt(hand / n), places=12)

    def test_yates_and_small_expected_warning(self) -> None:
        """Yates' statistic follows its formula; small counts warn."""
        table = ((12, 5), (7, 15))
        result = chi_square_independence_test(table, yates_correction=True)
        rows, columns, n = (17, 22), (19, 20), 39
        hand = sum(
            (abs(table[i][j] - rows[i] * columns[j] / n) - 0.5) ** 2
            / (rows[i] * columns[j] / n)
            for i in range(2)
            for j in range(2)
        )
        self.assertAlmostEqual(result.statistic, hand, places=12)
        with self.assertWarns(SmallExpectedCountWarning):
            small = chi_square_independence_test(((3, 1), (1, 3)))
        self.assertTrue(small.small_expected_counts)
        with self.assertWarns(SmallExpectedCountWarning):
            chi_square_goodness_of_fit([2, 3, 1])
        with self.assertRaises(ValueError):
            chi_square_independence_test(((1, 2, 3), (4, 5, 6)), yates_correction=True)
        with self.assertRaises(ValueError):
            chi_square_independence_test(((0, 0), (1, 2)))
        with self.assertRaises(ValueError):
            chi_square_goodness_of_fit([5, 5], [0.3, 0.3])

    def test_fisher_lady_tasting_tea(self) -> None:
        """Fisher's tea table: two-sided 34/70, greater 17/70 (hypergeometric)."""
        table = ((3, 1), (1, 3))
        self.assertAlmostEqual(fisher_exact_test(table).p_value, 34 / 70, places=14)
        greater = fisher_exact_test(table, alternative="greater")
        self.assertAlmostEqual(greater.p_value, 17 / 70, places=14)
        self.assertAlmostEqual(
            fisher_exact_test(table, alternative="less").p_value, 69 / 70, places=14
        )
        self.assertEqual(greater.odds_ratio, 9.0)

    def test_binomial_matches_direct_summation(self) -> None:
        """Exact binomial p-values equal brute-force sums of the null mass."""
        for x, n, p0 in ((7, 20, 0.3), (2, 15, 0.5), (19, 25, 0.6), (0, 8, 0.2)):
            mass = [math.comb(n, k) * p0**k * (1 - p0) ** (n - k) for k in range(n + 1)]
            upper = sum(mass[x:])
            lower = sum(mass[: x + 1])
            two = sum(m for m in mass if m <= mass[x] * (1 + 1e-7))
            self.assertAlmostEqual(
                binomial_test(x, n, null_probability=p0, alternative="greater").p_value,
                upper,
                places=12,
            )
            self.assertAlmostEqual(
                binomial_test(x, n, null_probability=p0, alternative="less").p_value,
                lower,
                places=12,
            )
            self.assertAlmostEqual(
                binomial_test(x, n, null_probability=p0).p_value,
                min(1.0, two),
                places=12,
            )
        with self.assertRaises(ValueError):
            binomial_test(3, 5, null_probability=1.0)


def _brute_mann_whitney(
    first: tuple[float, ...], second: tuple[float, ...]
) -> list[int]:
    """Enumerate every rank assignment and return null counts of U."""
    m, n = len(first), len(second)
    counts = [0] * (m * n + 1)
    for chosen in itertools.combinations(range(1, m + n + 1), m):
        counts[sum(chosen) - m * (m + 1) // 2] += 1
    return counts


class RankTestTests(unittest.TestCase):
    """Mann-Whitney and Wilcoxon against enumeration oracles."""

    def test_mann_whitney_distribution_matches_enumeration(self) -> None:
        """The generating-function counts equal brute-force enumeration."""
        for m, n in ((1, 1), (3, 4), (5, 5), (2, 7)):
            expected = _brute_mann_whitney(tuple(range(m)), tuple(range(n)))
            self.assertEqual(list(mann_whitney_u_distribution(m, n)), expected)

    def test_mann_whitney_exact_p_values(self) -> None:
        """Exact p-values equal tail sums of the enumerated distribution."""
        first = (1.83, 0.50, 1.62, 2.48, 1.68, 1.88)
        second = (0.878, 0.647, 0.598, 2.05, 1.06)
        u = sum((x > y) for x in first for y in second)
        counts = _brute_mann_whitney(first, second)
        total = sum(counts)
        upper = sum(counts[u:]) / total
        lower = sum(counts[: u + 1]) / total
        greater = mann_whitney_u_test(first, second, alternative="greater")
        self.assertTrue(greater.exact)
        self.assertEqual(greater.statistic, u)
        self.assertAlmostEqual(greater.p_value, upper, places=14)
        self.assertAlmostEqual(
            mann_whitney_u_test(first, second, alternative="less").p_value, lower
        )
        self.assertAlmostEqual(
            mann_whitney_u_test(first, second).p_value, min(1.0, 2 * min(upper, lower))
        )
        self.assertAlmostEqual(greater.effect_size, 2 * u / 30 - 1)

    def test_mann_whitney_normal_with_ties(self) -> None:
        """Tie-corrected normal approximation equals its hand formula."""
        first = (1, 2, 2, 3, 4, 5, 5, 5, 6)
        second = (3, 3, 4, 6, 7, 7, 8)
        result = mann_whitney_u_test(first, second, alternative="less")
        self.assertFalse(result.exact)
        u = sum((x > y) + 0.5 * (x == y) for x in first for y in second)
        self.assertEqual(result.statistic, u)
        m, n = 9, 7
        big_n = m + n
        pooled = first + second
        ties = sum(pooled.count(v) ** 3 - pooled.count(v) for v in set(pooled))
        variance = m * n / 12 * ((big_n + 1) - ties / (big_n * (big_n - 1)))
        z = (u - m * n / 2 + 0.5) / math.sqrt(variance)
        self.assertAlmostEqual(result.p_value, _normal_cdf(z), places=13)
        # Cross-check (SciPy 1.14.1 asymptotic, less): 0.054581532546190054.
        self.assertAlmostEqual(result.p_value, 0.054581532546190054, places=12)
        with self.assertRaises(ValueError):
            mann_whitney_u_test(first, second, method="exact")

    def test_wilcoxon_exact_matches_sign_enumeration(self) -> None:
        """Exact signed-rank p-values equal enumeration over 2^n sign vectors."""
        before = (1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
        after = (0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
        differences = [x - y for x, y in zip(before, after, strict=True)]
        order = sorted(range(9), key=lambda i: abs(differences[i]))
        rank = {index: position + 1 for position, index in enumerate(order)}
        w_plus = sum(rank[i] for i in range(9) if differences[i] > 0)
        sums = [
            sum(r for r, s in zip(range(1, 10), signs, strict=True) if s)
            for signs in itertools.product((False, True), repeat=9)
        ]
        upper = sum(s >= w_plus for s in sums) / 512
        result = wilcoxon_signed_rank_test(before, after, alternative="greater")
        self.assertTrue(result.exact)
        self.assertEqual(result.statistic, w_plus)
        self.assertAlmostEqual(result.p_value, upper, places=14)
        # R wilcox.test example data (Hamilton depression scale): V = 40,
        # p = 0.01953; SciPy 1.14.1 cross-check gives 0.01953125.
        self.assertAlmostEqual(result.p_value, 0.01953125, places=12)
        self.assertEqual(sum(signed_rank_distribution(9)), 512)

    def test_wilcoxon_normal_with_ties_and_zeros(self) -> None:
        """Zeros are dropped and the tie-corrected variance is used."""
        differences = (1.5, -2, 2, 3, -3, 0.5, 4, 4, -1, 6, 0.0)
        result = wilcoxon_signed_rank_test(differences, alternative="greater")
        kept = [d for d in differences if d != 0.0]
        n = len(kept)
        self.assertEqual(result.sample_sizes, (n,))
        absolute = [abs(d) for d in kept]
        ties = sum(absolute.count(v) ** 3 - absolute.count(v) for v in set(absolute))
        variance = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
        z = (result.statistic - n * (n + 1) / 4 - 0.5) / math.sqrt(variance)
        self.assertAlmostEqual(result.p_value, 1 - _normal_cdf(z), places=13)
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank_test([0.0, 0.0])


class PermutationAndAnovaTests(unittest.TestCase):
    """Permutation test enumeration and ANOVA F statistics."""

    def test_permutation_exact_enumeration(self) -> None:
        """Exact p equals an independent enumeration over index subsets."""
        first = (12.1, 14.3, 13.8, 15.2, 11.9)
        second = (10.2, 11.5, 12.0, 10.8, 13.1, 11.1)
        pooled = first + second
        observed = statistics.fmean(first) - statistics.fmean(second)
        extreme = total = 0
        for chosen in itertools.combinations(range(11), 5):
            a = [pooled[i] for i in chosen]
            b = [pooled[i] for i in range(11) if i not in chosen]
            total += 1
            extreme += statistics.fmean(a) - statistics.fmean(b) >= observed - 1e-12
        result = permutation_test(first, second, alternative="greater")
        self.assertTrue(result.exact)
        self.assertEqual(result.permutations, math.comb(11, 5))
        self.assertAlmostEqual(result.p_value, extreme / total, places=14)
        self.assertAlmostEqual(result.statistic, observed, places=12)

    def test_permutation_monte_carlo_correction(self) -> None:
        """Monte Carlo p is (b+1)/(m+1), seeded, and near the exact value."""
        first = (12.1, 14.3, 13.8, 15.2, 11.9)
        second = (10.2, 11.5, 12.0, 10.8, 13.1, 11.1)
        exact = permutation_test(first, second).p_value
        draws = 4999
        mc = permutation_test(first, second, permutations=draws, max_exact=0, seed=3)
        again = permutation_test(
            first, second, permutations=draws, max_exact=0, seed=RandomStream(3)
        )
        self.assertFalse(mc.exact)
        self.assertEqual(mc.p_value, again.p_value)
        hits = mc.p_value * (draws + 1) - 1
        self.assertAlmostEqual(hits, round(hits), places=6)
        self.assertGreater(mc.p_value, 0.0)
        self.assertLess(
            abs(mc.p_value - exact), 4 * math.sqrt(exact * (1 - exact) / draws)
        )

    def test_one_way_anova_hand_values(self) -> None:
        """F, df, and eta^2 match hand sums; two groups reproduce t^2."""
        groups = (
            (6.9, 5.4, 5.8, 4.6, 4.0),
            (8.3, 6.8, 7.8, 9.2, 6.5),
            (8.0, 10.5, 8.1, 6.9, 9.3),
        )
        result = one_way_anova(*groups)
        grand = statistics.fmean([v for g in groups for v in g])
        ssb = sum(len(g) * (statistics.fmean(g) - grand) ** 2 for g in groups)
        ssw = sum((v - statistics.fmean(g)) ** 2 for g in groups for v in g)
        self.assertAlmostEqual(result.statistic, (ssb / 2) / (ssw / 12), places=12)
        self.assertEqual((result.df_between, result.df_within), (2, 12))
        self.assertAlmostEqual(result.eta_squared, ssb / (ssb + ssw), places=12)
        # Cross-check (SciPy 1.14.1 f_oneway): F = 9.5911070364, p = 0.0032482226.
        self.assertAlmostEqual(result.p_value, 0.0032482226008593, places=12)
        two = one_way_anova(groups[0], groups[1])
        pooled = two_sample_t_test(groups[0], groups[1], equal_variances=True)
        self.assertAlmostEqual(two.statistic, pooled.statistic**2, places=10)
        self.assertAlmostEqual(two.p_value, pooled.p_value, places=12)
        with self.assertRaises(ValueError):
            one_way_anova(groups[0])
        with self.assertRaises(ValueError):
            one_way_anova((1.0, 1.0), (2.0, 2.0))


class BootstrapTests(unittest.TestCase):
    """Bootstrap replicates, jackknife, and bootstrap intervals."""

    @classmethod
    def setUpClass(cls) -> None:
        """Draw one seeded normal sample shared by interval tests."""
        generator = random.Random(20261011)
        cls.data = tuple(generator.gauss(10.0, 2.0) for _ in range(200))

    def test_bootstrap_is_reproducible_and_estimates_standard_error(self) -> None:
        """Same seed, same replicates; SE of the mean is near s/sqrt(n)."""
        first = bootstrap(statistics.fmean, self.data, replicates=500, seed=11)
        second = bootstrap(statistics.fmean, self.data, replicates=500, seed=11)
        self.assertEqual(first.replicates, second.replicates)
        analytic = statistics.pstdev(self.data) / math.sqrt(len(self.data))
        self.assertLess(abs(first.standard_error / analytic - 1.0), 0.1)
        self.assertAlmostEqual(
            first.bias_corrected_estimate, first.estimate - first.bias
        )
        with self.assertRaises(ValueError):
            bootstrap(statistics.fmean, self.data, replicates=1)

    def test_jackknife_known_identities(self) -> None:
        """Mean: SE = s/sqrt(n), bias 0; plug-in variance corrects to unbiased."""
        sample = (2.0, 4.5, 3.1, 7.7, 5.2, 6.0, 1.4)
        result = jackknife(statistics.fmean, sample)
        self.assertAlmostEqual(
            result.standard_error, statistics.stdev(sample) / math.sqrt(7), places=12
        )
        self.assertAlmostEqual(result.bias, 0.0, places=12)
        variance = jackknife(statistics.pvariance, sample)
        self.assertAlmostEqual(
            variance.bias_corrected_estimate, statistics.variance(sample), places=12
        )

    def test_bca_close_to_analytic_interval_for_normal_mean(self) -> None:
        """With n=200 normal data BCa and percentile track the t interval."""
        reference = student_t_mean_confidence_interval(self.data)
        half_width = 0.5 * (reference.upper - reference.lower)
        for method in ("bca", "percentile", "basic", "normal"):
            interval = bootstrap_confidence_interval(
                statistics.fmean, self.data, method=method, replicates=2000, seed=7
            )
            with self.subTest(method=method):
                self.assertLess(abs(interval.lower - reference.lower), 0.1 * half_width)
                self.assertLess(abs(interval.upper - reference.upper), 0.1 * half_width)
        bca = bootstrap_confidence_interval(
            statistics.fmean, self.data, method="bca", replicates=2000, seed=7
        )
        assert bca.acceleration is not None and bca.bias_correction is not None
        # For the mean, a = skewness / (6 sqrt(n)) is tiny for normal data.
        self.assertLess(abs(bca.acceleration), 0.01)
        self.assertLess(abs(bca.bias_correction), 0.1)

    def test_studentized_interval_and_errors(self) -> None:
        """Bootstrap-t is reproducible and invalid requests raise."""
        sample = self.data[:15]

        def mean_standard_error(values: object) -> float:
            """Return s/sqrt(n) for a resample."""
            seq = list(values)  # type: ignore[call-overload]
            return statistics.stdev(seq) / math.sqrt(len(seq))

        first = bootstrap_confidence_interval(
            statistics.fmean,
            sample,
            method="studentized",
            replicates=400,
            seed=5,
            standard_error=mean_standard_error,
        )
        second = bootstrap_confidence_interval(
            statistics.fmean,
            sample,
            method="studentized",
            replicates=400,
            seed=5,
            standard_error=mean_standard_error,
        )
        self.assertEqual((first.lower, first.upper), (second.lower, second.upper))
        self.assertLess(first.lower, statistics.fmean(sample))
        self.assertGreater(first.upper, statistics.fmean(sample))
        with self.assertRaises(ValueError):
            bootstrap_confidence_interval(
                statistics.fmean, (3.0, 3.0, 3.0, 3.0), method="bca", seed=1
            )
        with self.assertRaises(ValueError):
            bootstrap_confidence_interval(
                statistics.fmean,
                sample,
                method="magic",
                seed=1,  # type: ignore[arg-type]
            )


class PowerAndEffectSizeTests(unittest.TestCase):
    """Effect sizes, power functions, and sample-size planning."""

    def test_effect_sizes(self) -> None:
        """Cohen's d uses the pooled SD; Hedges' J matches its approximation."""
        first, second = (5.0, 6.0, 7.0, 8.0, 9.0), (4.0, 4.5, 6.0, 6.5, 5.0)
        pooled = math.sqrt(
            (4 * statistics.variance(first) + 4 * statistics.variance(second)) / 8
        )
        d = (statistics.fmean(first) - statistics.fmean(second)) / pooled
        self.assertAlmostEqual(cohens_d(first, second), d, places=12)
        self.assertAlmostEqual(hedges_g(first, second), hedges_correction(8) * d)
        self.assertLess(abs(hedges_correction(20) - (1 - 3 / 79)), 2e-4)
        with self.assertRaises(ValueError):
            hedges_correction(1.0)

    def test_t_power_matches_reference_values(self) -> None:
        """R ``power.t.test(n=64, delta=0.5)`` reports power 0.8014596."""
        power = t_test_power(0.5, 64, design="two-sample")
        self.assertAlmostEqual(power, 0.8014596, places=6)
        planned = required_sample_size(0.5, design="two-sample")
        self.assertEqual(planned.sample_size, 64)
        self.assertGreaterEqual(planned.achieved_power, 0.8)
        # Classic one-sample / paired requirement for d = 0.5: n = 34.
        self.assertEqual(required_sample_size(0.5).sample_size, 34)

    def test_z_power_closed_form_and_sample_size(self) -> None:
        """Z-test power equals its formula; two-sample d=0.5 needs 63 per group."""
        delta = 0.4 * math.sqrt(30)
        critical = 1.959963984540054
        expected = _normal_cdf(delta - critical) + _normal_cdf(-delta - critical)
        self.assertAlmostEqual(z_test_power(0.4, 30), expected, places=9)
        self.assertEqual(
            required_sample_size(0.5, test="z", design="two-sample").sample_size, 63
        )
        with self.assertRaises(ValueError):
            required_sample_size(0.0)
        with self.assertRaises(ValueError):
            required_sample_size(-0.3, alternative="greater")

    def test_t_power_matches_simulated_rejection_rate(self) -> None:
        """A seeded simulation of one-sided t tests agrees with exact power."""
        n, d, alpha, reps = 10, 0.8, 0.05, 4000
        critical = student_t_quantile(1 - alpha, n - 1)
        generator = random.Random(8)
        rejections = 0
        for _ in range(reps):
            sample = [generator.gauss(d, 1.0) for _ in range(n)]
            t = statistics.fmean(sample) / (statistics.stdev(sample) / math.sqrt(n))
            rejections += t > critical
        power = t_test_power(d, n, alternative="greater")
        tolerance = 4 * math.sqrt(power * (1 - power) / reps)
        self.assertLess(abs(rejections / reps - power), tolerance)


class MultipleTestingTests(unittest.TestCase):
    """Adjusted p-values on a hand-computed vector."""

    P_VALUES = (0.01, 0.04, 0.03, 0.005)

    def test_known_adjustments(self) -> None:
        """Hand computation for m = 4 (sorted 0.005, 0.01, 0.03, 0.04)."""
        expected = {
            "bonferroni": (0.04, 0.16, 0.12, 0.02),
            "holm": (0.03, 0.06, 0.06, 0.02),
            "bh": (0.02, 0.04, 0.04, 0.02),
            "by": tuple(
                v * (1 + 1 / 2 + 1 / 3 + 1 / 4) for v in (0.02, 0.04, 0.04, 0.02)
            ),
        }
        for method, values in expected.items():
            result = adjust_p_values(self.P_VALUES, method=method)  # type: ignore[arg-type]
            with self.subTest(method=method):
                for got, want in zip(result.adjusted_p_values, values, strict=True):
                    self.assertAlmostEqual(got, want, places=12)
        holm = adjust_p_values(self.P_VALUES, method="holm")
        self.assertEqual(holm.rejected, (True, False, False, True))
        self.assertEqual(holm.error_rate, "FWER")
        bh = adjust_p_values(self.P_VALUES, method="bh")
        self.assertEqual(bh.rejection_count, 4)
        self.assertEqual(bh.error_rate, "FDR")

    def test_invalid_inputs(self) -> None:
        """Empty families, invalid p-values, and unknown methods raise."""
        with self.assertRaises(ValueError):
            adjust_p_values([])
        with self.assertRaises(ValueError):
            adjust_p_values([0.2, 1.5])
        with self.assertRaises(ValueError):
            adjust_p_values([0.2], method="sidak")  # type: ignore[arg-type]


class EquivalenceTests(unittest.TestCase):
    """Two one-sided tests."""

    def test_tost_matches_one_sided_t_tests(self) -> None:
        """TOST p-values equal one-sided t tests at the margins."""
        values = (0.12, -0.05, 0.08, 0.02, -0.10, 0.04, 0.07, -0.02, 0.01, 0.03)
        result = tost_equivalence_test(values, lower_bound=-0.2, upper_bound=0.2)
        low = one_sample_t_test(values, null_mean=-0.2, alternative="greater")
        high = one_sample_t_test(values, null_mean=0.2, alternative="less")
        self.assertAlmostEqual(result.lower_p_value, low.p_value, places=13)
        self.assertAlmostEqual(result.upper_p_value, high.p_value, places=13)
        self.assertEqual(result.p_value, max(low.p_value, high.p_value))
        self.assertTrue(result.equivalent)
        inside = (
            -0.2 < result.confidence_interval.lower
            and result.confidence_interval.upper < 0.2
        )
        self.assertEqual(inside, result.equivalent)
        narrow = tost_equivalence_test(values, lower_bound=-0.01, upper_bound=0.01)
        self.assertFalse(narrow.equivalent)
        with self.assertRaises(ValueError):
            tost_equivalence_test(values, lower_bound=0.2, upper_bound=-0.2)

    def test_two_sample_tost_runs_with_warnings_clean(self) -> None:
        """Two-sample TOST uses Welch by default and emits no warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = tost_equivalence_test(
                SLEEP_GROUP_1, SLEEP_GROUP_2, lower_bound=-3.0, upper_bound=3.0
            )
        self.assertIn("Welch", result.method)


if __name__ == "__main__":
    unittest.main()
