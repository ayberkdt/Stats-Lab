"""Tests for robust means, shrinkage covariance, and dependence measures.

Median-of-means and Catoni's estimator are checked against the sample mean
in their limiting cases, for translation equivariance, against the derived
deviation bound by simulation, and for smaller 99th-percentile errors than
the sample mean on heavy-tailed data. Ledoit-Wolf shrinkage is compared
with a hand computation and with the literal definition of its quantities
and its Frobenius risk is simulated. Chatterjee's coefficient is compared
with its exact value for monotone data and its null variance; distance
correlation with its invariances, the linear case, and the bivariate normal
closed form, which is itself checked by an independent two-dimensional
quadrature.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.foundations import RandomStream
from probstat.linalg import cholesky, symmetric_eigen
from probstat.statistics import (
    catoni_influence,
    catoni_mean,
    catoni_scale,
    chatterjee_correlation,
    distance_correlation,
    distance_correlation_bivariate_normal,
    ledoit_wolf_covariance,
    mean,
    median,
    median_of_means,
    median_of_means_block_count,
    pearson_correlation,
)


def _student_t(rng: random.Random, dof: float, size: int) -> list[float]:
    """Return ``size`` Student-t draws as normal over root chi-square."""
    return [
        rng.gauss(0.0, 1.0) / math.sqrt(rng.gammavariate(0.5 * dof, 2.0) / dof)
        for _ in range(size)
    ]


def _upper_percentile(values: list[float], level: float) -> float:
    """Return the order statistic at ``floor(level * n)``."""
    ordered = sorted(values)
    return ordered[int(level * len(ordered))]


class MedianOfMeansTests(unittest.TestCase):
    """Median-of-means estimator."""

    def test_block_count_rule(self) -> None:
        """The helper returns ``ceil(8 log(1/delta))``."""
        self.assertEqual(median_of_means_block_count(0.05), 24)
        self.assertEqual(median_of_means_block_count(math.exp(-1.0)), 8)
        self.assertEqual(median_of_means_block_count(0.999), 1)
        for delta in (0.3, 0.05, 1e-3, 1e-6):
            k = median_of_means_block_count(delta)
            self.assertLessEqual(math.exp(-k / 8.0), delta)
            self.assertGreater(math.exp(-(k - 1) / 8.0), delta)

    def test_one_block_is_the_sample_mean(self) -> None:
        """With ``k = 1`` the estimate equals the arithmetic mean."""
        rng = random.Random(1)
        x = [rng.gauss(2.0, 3.0) for _ in range(57)]
        for assignment in ("random", "sequential"):
            result = median_of_means(x, blocks=1, assignment=assignment, seed=4)
            self.assertAlmostEqual(result.estimate, mean(x), places=13)
            self.assertEqual(result.block_sizes, (57,))

    def test_n_blocks_is_the_sample_median(self) -> None:
        """With ``k = n`` every block is one point, so the median results."""
        rng = random.Random(2)
        x = [rng.gauss(0.0, 1.0) for _ in range(31)]
        self.assertAlmostEqual(
            median_of_means(x, blocks=31, seed=0).estimate, median(x), places=13
        )

    def test_sequential_blocks_by_hand(self) -> None:
        """Seven values in three sequential blocks of sizes 3, 2, 2."""
        result = median_of_means(
            [1.0, 2.0, 6.0, 10.0, 20.0, -1.0, 1.0], blocks=3, assignment="sequential"
        )
        self.assertEqual(result.block_sizes, (3, 2, 2))
        self.assertEqual(result.block_means, (3.0, 15.0, 0.0))
        self.assertEqual(result.estimate, 3.0)
        even = median_of_means([1.0, 3.0, 5.0, 9.0], blocks=2, assignment="sequential")
        self.assertEqual(even.estimate, 4.5)

    def test_blocks_partition_the_sample(self) -> None:
        """Block sizes sum to ``n`` and the size-weighted means give the mean."""
        rng = random.Random(3)
        x = [rng.expovariate(1.0) for _ in range(103)]
        result = median_of_means(x, blocks=10, seed=11)
        self.assertEqual(sum(result.block_sizes), 103)
        self.assertEqual(max(result.block_sizes) - min(result.block_sizes), 1)
        weighted = sum(
            s * m for s, m in zip(result.block_sizes, result.block_means, strict=True)
        )
        self.assertAlmostEqual(weighted / 103, mean(x), places=12)

    def test_translation_equivariance_and_reproducibility(self) -> None:
        """Shifting the data shifts the estimate; a seed fixes the blocks."""
        rng = random.Random(4)
        x = [rng.paretovariate(2.5) for _ in range(200)]
        base = median_of_means(x, blocks=9, seed=7)
        again = median_of_means(x, blocks=9, seed=RandomStream(7))
        other = median_of_means(x, blocks=9, seed=8)
        shifted = median_of_means([v + 100.0 for v in x], blocks=9, seed=7)
        self.assertEqual(base, again)
        self.assertNotEqual(base.block_means, other.block_means)
        self.assertAlmostEqual(shifted.estimate, base.estimate + 100.0, places=10)

    def test_derived_deviation_bound_holds_in_simulation(self) -> None:
        """The failure frequency stays below ``exp(-k/8)`` on Pareto data."""
        alpha = 2.5
        mu = alpha / (alpha - 1.0)
        sigma = math.sqrt(alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0)))
        rng = random.Random(5)
        failures = 0
        replicates = 1500
        for replicate in range(replicates):
            x = [rng.paretovariate(alpha) for _ in range(120)]
            result = median_of_means(x, blocks=8, seed=replicate)
            self.assertAlmostEqual(result.failure_probability_bound, math.exp(-1.0))
            self.assertAlmostEqual(
                result.deviation_bound(sigma), 2.0 * sigma / math.sqrt(15.0)
            )
            if abs(result.estimate - mu) > result.deviation_bound(sigma):
                failures += 1
        self.assertLessEqual(failures / replicates, math.exp(-1.0))

    def test_heavy_tails_beat_the_sample_mean(self) -> None:
        """The 99th percentile of the error is smaller on t(2.5) and Pareto."""
        for name in ("student", "pareto"):
            rng = random.Random(5)
            mean_errors: list[float] = []
            mom_errors: list[float] = []
            for replicate in range(1500):
                if name == "student":
                    x = _student_t(rng, 2.5, 200)
                    mu = 0.0
                else:
                    x = [rng.paretovariate(2.5) for _ in range(200)]
                    mu = 2.5 / 1.5
                mean_errors.append(abs(mean(x) - mu))
                estimate = median_of_means(x, delta=0.01, seed=replicate).estimate
                mom_errors.append(abs(estimate - mu))
            self.assertLess(
                _upper_percentile(mom_errors, 0.99),
                0.9 * _upper_percentile(mean_errors, 0.99),
                name,
            )

    def test_validation(self) -> None:
        """Invalid block counts, levels, and assignments are rejected."""
        x = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            median_of_means(x)
        with self.assertRaises(ValueError):
            median_of_means(x, blocks=2, delta=0.1)
        with self.assertRaises(ValueError):
            median_of_means(x, blocks=4)
        with self.assertRaises(ValueError):
            median_of_means(x, blocks=0)
        with self.assertRaises(TypeError):
            median_of_means(x, blocks=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            median_of_means(x, delta=0.01)
        with self.assertRaises(ValueError):
            median_of_means(x, blocks=2, assignment="sorted")
        with self.assertRaises(ValueError):
            median_of_means([], blocks=1)
        with self.assertRaises(ValueError):
            median_of_means_block_count(1.0)
        with self.assertRaises(ValueError):
            median_of_means(x, blocks=1).deviation_bound(-1.0)


class CatoniTests(unittest.TestCase):
    """Catoni's M-estimator of the mean."""

    def test_influence_function(self) -> None:
        """``psi`` is odd, increasing, and ``x - x^3/6 + x^4/8`` near zero."""
        self.assertEqual(catoni_influence(0.0), 0.0)
        self.assertAlmostEqual(catoni_influence(1.0), math.log(2.5), places=15)
        self.assertAlmostEqual(catoni_influence(-2.0), -math.log(5.0), places=15)
        x = 1e-3
        series = x - x**3 / 6.0 + x**4 / 8.0
        self.assertAlmostEqual(catoni_influence(x), series, delta=1e-15)
        grid = [-5.0 + 0.1 * i for i in range(101)]
        values = [catoni_influence(v) for v in grid]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:], strict=False)))

    def test_default_scale(self) -> None:
        """The default scale is ``sqrt(2 log(1/delta)/(n v))``."""
        self.assertAlmostEqual(
            catoni_scale(50, 4.0, 0.05),
            math.sqrt(2.0 * math.log(20.0) / 200.0),
            places=15,
        )
        x = [0.0, 1.0, 5.0, 2.0]
        explicit = catoni_mean(x, variance_bound=3.0, delta=0.1)
        self.assertAlmostEqual(explicit.scale, catoni_scale(4, 3.0, 0.1), places=15)
        self.assertFalse(explicit.variance_estimated)
        plug_in = catoni_mean(x, delta=0.1)
        self.assertTrue(plug_in.variance_estimated)
        self.assertAlmostEqual(plug_in.variance_bound or 0.0, 14.0 / 3.0, places=13)

    def test_estimating_equation_is_solved(self) -> None:
        """The returned root satisfies the estimating equation."""
        rng = random.Random(6)
        x = _student_t(rng, 2.5, 300)
        result = catoni_mean(x, scale=0.3)
        self.assertTrue(result.converged)
        total = math.fsum(catoni_influence(0.3 * (v - result.estimate)) for v in x)
        self.assertAlmostEqual(total, 0.0, delta=1e-9)
        self.assertAlmostEqual(result.residual, total, delta=1e-9)
        self.assertGreater(result.evaluations, 2)
        self.assertTrue(min(x) < result.estimate < max(x))

    def test_small_scale_tends_to_the_sample_mean(self) -> None:
        """The gap to the mean shrinks like ``a^2`` as ``a -> 0``."""
        rng = random.Random(7)
        x = [rng.expovariate(0.5) for _ in range(80)]
        gaps = [
            abs(catoni_mean(x, scale=a).estimate - mean(x)) for a in (1e-1, 1e-2, 1e-3)
        ]
        self.assertLess(gaps[1], gaps[0] / 50.0)
        self.assertLess(gaps[2], gaps[1] / 50.0)
        self.assertLess(gaps[2], 1e-5)

    def test_translation_equivariance(self) -> None:
        """Shifting the data shifts the estimate for a fixed scale."""
        rng = random.Random(8)
        x = [rng.paretovariate(2.5) for _ in range(150)]
        base = catoni_mean(x, scale=0.2).estimate
        shifted = catoni_mean([v - 37.5 for v in x], scale=0.2).estimate
        self.assertAlmostEqual(shifted, base - 37.5, places=9)
        plug = catoni_mean(x).estimate
        self.assertAlmostEqual(
            catoni_mean([v + 5.0 for v in x]).estimate, plug + 5.0, places=9
        )

    def test_outlier_is_downweighted(self) -> None:
        """One gross outlier moves the estimate far less than the mean."""
        x = [float(i % 7) for i in range(70)]
        dirty = [*x, 1e6]
        clean = catoni_mean(x, scale=0.5).estimate
        self.assertLess(abs(catoni_mean(dirty, scale=0.5).estimate - clean), 1.0)
        self.assertGreater(abs(mean(dirty) - mean(x)), 1e4)

    def test_heavy_tails_beat_the_sample_mean(self) -> None:
        """The 99th percentile of the error is smaller on t(2.5) and Pareto."""
        cases = {
            "student": (0.0, 5.0),
            "pareto": (2.5 / 1.5, 2.5 / (1.5**2 * 0.5)),
        }
        for name, (mu, variance) in cases.items():
            rng = random.Random(5)
            mean_errors: list[float] = []
            catoni_errors: list[float] = []
            for _ in range(1500):
                if name == "student":
                    x = _student_t(rng, 2.5, 200)
                else:
                    x = [rng.paretovariate(2.5) for _ in range(200)]
                mean_errors.append(abs(mean(x) - mu))
                result = catoni_mean(x, variance_bound=variance, delta=0.01)
                self.assertTrue(result.converged)
                catoni_errors.append(abs(result.estimate - mu))
            self.assertLess(
                _upper_percentile(catoni_errors, 0.99),
                0.9 * _upper_percentile(mean_errors, 0.99),
                name,
            )

    def test_constant_sample_and_validation(self) -> None:
        """A constant sample returns its value; bad arguments raise."""
        constant = catoni_mean([4.0, 4.0, 4.0])
        self.assertEqual(constant.estimate, 4.0)
        self.assertTrue(constant.converged)
        self.assertEqual(catoni_mean([3.0], scale=1.0).estimate, 3.0)
        with self.assertRaises(ValueError):
            catoni_mean([1.0, 2.0], scale=0.0)
        with self.assertRaises(ValueError):
            catoni_mean([1.0, 2.0], variance_bound=-1.0)
        with self.assertRaises(ValueError):
            catoni_mean([1.0, 2.0], delta=0.0)
        with self.assertRaises(ValueError):
            catoni_mean([1.0, 2.0], tolerance=0.0)
        with self.assertRaises(ValueError):
            catoni_mean([1.0, 2.0], max_iterations=0)
        with self.assertRaises(ValueError):
            catoni_mean([])
        with self.assertRaises(ValueError):
            catoni_scale(0, 1.0, 0.1)
        with self.assertRaises(ValueError):
            catoni_influence(math.inf)


def _frobenius_loss(matrix: tuple[tuple[float, ...], ...], truth: list[float]) -> float:
    """Return the squared Frobenius distance to ``diag(truth)``."""
    p = len(truth)
    return sum(
        (matrix[i][j] - (truth[i] if i == j else 0.0)) ** 2
        for i in range(p)
        for j in range(p)
    )


class LedoitWolfTests(unittest.TestCase):
    """Ledoit-Wolf linear shrinkage."""

    def test_hand_computed_example(self) -> None:
        """Four mean-zero points in the plane, worked by hand.

        Rows (1,0), (-1,0), (0,2), (0,-2): ``S = diag(1/2, 2)``,
        ``m = 5/4``, ``d^2 = ((3/4)^2 + (3/4)^2)/2 = 9/16``. The outer
        products are ``diag(1,0)`` twice and ``diag(0,4)`` twice, so
        ``||x x' - S||^2`` is ``(1/4 + 4)/2 = 17/8`` twice and
        ``(1/4 + 4)/2 = 17/8`` twice, ``bbar^2 = 4 (17/8)/16 = 17/32``.
        Intensity ``(17/32)/(9/16) = 17/18`` and the estimate is
        ``diag(5/4 - (3/4)/18, 5/4 + (3/4)/18)``.
        """
        data = [(1.0, 0.0), (-1.0, 0.0), (0.0, 2.0), (0.0, -2.0)]
        for center in (True, False):
            result = ledoit_wolf_covariance(data, center=center)
            self.assertEqual(result.sample_covariance, ((0.5, 0.0), (0.0, 2.0)))
            self.assertAlmostEqual(result.target_scale, 1.25, places=15)
            self.assertAlmostEqual(result.dispersion, 9.0 / 16.0, places=15)
            self.assertAlmostEqual(result.raw_sample_error, 17.0 / 32.0, places=15)
            self.assertAlmostEqual(result.shrinkage, 17.0 / 18.0, places=14)
            self.assertAlmostEqual(result.covariance[0][0], 1.25 - 1.0 / 24.0)
            self.assertAlmostEqual(result.covariance[1][1], 1.25 + 1.0 / 24.0)
            self.assertEqual(result.covariance[0][1], 0.0)
            self.assertEqual((result.observations, result.dimension), (4, 2))

    def test_matches_the_literal_definition(self) -> None:
        """The shortcut for ``bbar^2`` equals the sum of squared norms."""
        rng = random.Random(9)
        n, p = 12, 5
        raw = [[rng.gauss(0.0, 1.0 + j) + 0.5 * j for j in range(p)] for _ in range(n)]
        result = ledoit_wolf_covariance(raw)
        centre = [sum(row[j] for row in raw) / n for j in range(p)]
        rows = [[row[j] - centre[j] for j in range(p)] for row in raw]
        s = [[sum(r[i] * r[j] for r in rows) / n for j in range(p)] for i in range(p)]
        m = sum(s[i][i] for i in range(p)) / p
        d2 = (
            sum(
                (s[i][j] - (m if i == j else 0.0)) ** 2
                for i in range(p)
                for j in range(p)
            )
            / p
        )
        bbar2 = (
            sum(
                sum((r[i] * r[j] - s[i][j]) ** 2 for i in range(p) for j in range(p))
                / p
                for r in rows
            )
            / n**2
        )
        rho = min(bbar2, d2) / d2
        for j in range(p):
            self.assertAlmostEqual(result.location[j], centre[j], places=13)
        self.assertAlmostEqual(result.dispersion, d2, places=11)
        self.assertAlmostEqual(result.raw_sample_error, bbar2, places=11)
        self.assertAlmostEqual(result.shrinkage, rho, places=11)
        for i in range(p):
            for j in range(p):
                expected = (1.0 - rho) * s[i][j] + (rho * m if i == j else 0.0)
                self.assertAlmostEqual(result.covariance[i][j], expected, places=11)

    def test_positive_definite_when_p_exceeds_n(self) -> None:
        """With ``p > n`` the sample matrix is singular, the estimate is not."""
        rng = random.Random(10)
        n, p = 6, 15
        data = [[rng.gauss(0.0, 1.0 + 0.2 * j) for j in range(p)] for _ in range(n)]
        result = ledoit_wolf_covariance(data)
        self.assertTrue(0.0 < result.shrinkage <= 1.0)
        sample_eigen = symmetric_eigen(result.sample_covariance)[0]
        shrunk_eigen = symmetric_eigen(result.covariance)[0]
        self.assertLess(min(sample_eigen), 1e-10)
        floor = result.shrinkage * result.target_scale
        self.assertGreater(min(shrunk_eigen), 0.99 * floor)
        cholesky(result.covariance)
        with self.assertRaises(ValueError):
            cholesky(result.sample_covariance)
        self.assertAlmostEqual(
            sum(result.covariance[i][i] for i in range(p)),
            sum(result.sample_covariance[i][i] for i in range(p)),
            places=10,
        )

    def test_intensity_limits(self) -> None:
        """Near 1 for an identity truth with ``p ~ n``, near 0 for ``n >> p``."""
        rng = random.Random(11)
        identity = [[rng.gauss(0.0, 1.0) for _ in range(20)] for _ in range(25)]
        self.assertGreater(ledoit_wolf_covariance(identity).shrinkage, 0.9)
        scales = (1.0, 2.0, 3.0, 4.0)
        many = [[rng.gauss(0.0, s) for s in scales] for _ in range(4000)]
        result = ledoit_wolf_covariance(many)
        self.assertTrue(0.0 <= result.shrinkage < 0.01)

    def test_frobenius_risk_is_smaller_than_for_the_sample_covariance(self) -> None:
        """Simulated risk for ``p = 20``, ``n = 25`` and a diagonal truth."""
        rng = random.Random(3)
        p, n = 20, 25
        truth = [1.0 + 3.0 * i / (p - 1) for i in range(p)]
        sample_loss = shrunk_loss = 0.0
        wins = 0
        replicates = 30
        for _ in range(replicates):
            data = [
                [rng.gauss(0.0, math.sqrt(truth[j])) for j in range(p)]
                for _ in range(n)
            ]
            result = ledoit_wolf_covariance(data)
            self.assertTrue(0.0 <= result.shrinkage <= 1.0)
            loss_s = _frobenius_loss(result.sample_covariance, truth)
            loss_l = _frobenius_loss(result.covariance, truth)
            sample_loss += loss_s
            shrunk_loss += loss_l
            wins += loss_l < loss_s
        self.assertLess(shrunk_loss, 0.3 * sample_loss)
        self.assertEqual(wins, replicates)

    def test_degenerate_and_invalid_inputs(self) -> None:
        """A sample equal to its target gives intensity 0; bad data raise."""
        result = ledoit_wolf_covariance(
            [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]
        )
        self.assertEqual(result.dispersion, 0.0)
        self.assertEqual(result.shrinkage, 0.0)
        self.assertEqual(result.covariance, ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            ledoit_wolf_covariance([(1.0, 2.0)])
        with self.assertRaises(ValueError):
            ledoit_wolf_covariance([(1.0, 2.0), (1.0,)])
        with self.assertRaises(ValueError):
            ledoit_wolf_covariance([])
        single = ledoit_wolf_covariance([(1.0, 2.0)], center=False)
        self.assertEqual(single.sample_covariance, ((1.0, 2.0), (2.0, 4.0)))


class ChatterjeeTests(unittest.TestCase):
    """Chatterjee's rank correlation."""

    def test_exact_value_for_monotone_relations(self) -> None:
        """Strictly monotone data give ``(n - 2)/(n + 1)`` exactly."""
        for n in (2, 3, 10, 101):
            x = [0.37 * i - 4.0 for i in range(n)]
            for transform in (math.exp, lambda v: -(v**3)):
                result = chatterjee_correlation(x, [transform(v) for v in x])
                self.assertAlmostEqual(result.statistic, (n - 2) / (n + 1), places=14)

    def test_hand_computed_example(self) -> None:
        """Ranks 2, 4, 1, 3, 5 give ``1 - 3 * 9 / 24``."""
        result = chatterjee_correlation(
            [1.0, 2.0, 3.0, 4.0, 5.0], [20.0, 40.0, 10.0, 30.0, 50.0]
        )
        self.assertAlmostEqual(result.statistic, 1.0 - 27.0 / 24.0, places=15)
        self.assertAlmostEqual(result.z_score, math.sqrt(5.0 / 0.4) * (-0.125))
        self.assertAlmostEqual(
            result.p_value, 0.5 * math.erfc(result.z_score / math.sqrt(2.0))
        )
        self.assertEqual(result.x_ties, 0)

    def test_null_mean_and_variance(self) -> None:
        """Under independence ``sqrt(n) xi`` has mean 0 and variance 2/5."""
        rng = random.Random(12)
        n, replicates = 300, 1500
        values = []
        rejections = 0
        for _ in range(replicates):
            x = [rng.random() for _ in range(n)]
            y = [rng.gauss(0.0, 1.0) for _ in range(n)]
            result = chatterjee_correlation(x, y)
            values.append(math.sqrt(n) * result.statistic)
            rejections += result.p_value < 0.05
        centre = sum(values) / replicates
        spread = sum((v - centre) ** 2 for v in values) / (replicates - 1)
        self.assertLess(abs(centre), 4.0 * math.sqrt(0.4 / replicates))
        # The variance of a sample variance of normal data is 2 s^4/(R - 1).
        self.assertLess(abs(spread - 0.4), 4.0 * 0.4 * math.sqrt(2.0 / replicates))
        level = rejections / replicates
        self.assertLess(abs(level - 0.05), 4.0 * math.sqrt(0.05 * 0.95 / replicates))

    def test_detects_a_symmetric_non_monotone_relation(self) -> None:
        """``Y = X^2`` on symmetric ``X``: Pearson near 0, ``xi`` near 1."""
        rng = random.Random(13)
        x = [rng.gauss(0.0, 1.0) for _ in range(500)]
        y = [v * v for v in x]
        result = chatterjee_correlation(x, y)
        self.assertLess(abs(pearson_correlation(x, y)), 0.15)
        self.assertGreater(result.statistic, 0.95)
        self.assertLess(result.p_value, 1e-50)
        reverse = chatterjee_correlation(y, x)
        self.assertLess(reverse.statistic, 0.6)

    def test_ties(self) -> None:
        """Ties in ``x`` are broken by the seed; ties in ``y`` raise."""
        x = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        y = [0.3, 0.1, 0.9, 0.5, 0.2, 0.7]
        first = chatterjee_correlation(x, y, seed=1)
        self.assertEqual(first.x_ties, 5)
        self.assertEqual(first, chatterjee_correlation(x, y, seed=1))
        seen = {chatterjee_correlation(x, y, seed=s).statistic for s in range(30)}
        self.assertGreater(len(seen), 1)
        with self.assertRaises(ValueError):
            chatterjee_correlation(y, x)

    def test_validation(self) -> None:
        """Unequal lengths and single pairs are rejected."""
        with self.assertRaises(ValueError):
            chatterjee_correlation([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            chatterjee_correlation([1.0], [1.0])
        with self.assertRaises(TypeError):
            chatterjee_correlation([1.0, "a"], [1.0, 2.0])  # type: ignore[list-item]


def _expected_abs_product(r: float) -> float:
    """Return ``E|UV|`` for a standard bivariate normal by quadrature.

    With ``U = R cos(phi)`` and ``V = r U + sqrt(1 - r^2) R sin(phi)`` for
    independent standard normal coordinates in polar form, ``E[R^2] = 2``
    and ``phi`` is uniform, so ``E|UV|`` is twice the mean over ``phi`` of
    ``|cos(phi) (r cos(phi) + sqrt(1 - r^2) sin(phi))|`` (midpoint rule).
    """
    steps = 20_000
    s = math.sqrt(1.0 - r * r)
    total = 0.0
    for i in range(steps):
        phi = 2.0 * math.pi * (i + 0.5) / steps
        total += abs(math.cos(phi) * (r * math.cos(phi) + s * math.sin(phi)))
    return 2.0 * total / steps


class DistanceCorrelationTests(unittest.TestCase):
    """Distance covariance and correlation."""

    def test_two_point_example_by_hand(self) -> None:
        """For ``n = 2`` the centred matrices are ``+-d/2`` and ``dCor = 1``."""
        result = distance_correlation([0.0, 3.0], [(1.0, 1.0), (1.0, 5.0)])
        self.assertAlmostEqual(result.distance_variance_x, 1.5, places=14)
        self.assertAlmostEqual(result.distance_variance_y, 2.0, places=14)
        self.assertAlmostEqual(result.distance_covariance, math.sqrt(3.0), places=14)
        self.assertAlmostEqual(result.distance_correlation, 1.0, places=14)
        self.assertIsNone(result.p_value)

    def test_linear_relation_gives_one(self) -> None:
        """``Y = a + b X`` with ``b != 0`` in one dimension gives 1."""
        rng = random.Random(14)
        x = [rng.gauss(0.0, 1.0) for _ in range(60)]
        for slope in (2.5, -0.3):
            y = [1.0 + slope * v for v in x]
            self.assertAlmostEqual(
                distance_correlation(x, y).distance_correlation, 1.0, places=12
            )

    def test_range_symmetry_and_invariance(self) -> None:
        """Values in [0, 1]; invariant to shifts, rotations, and scaling."""
        rng = random.Random(15)
        n = 70
        x = [(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(n)]
        y = [
            (u * v + 0.3 * rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), abs(u))
            for u, v in x
        ]
        base = distance_correlation(x, y)
        self.assertTrue(0.0 < base.distance_correlation < 1.0)
        swapped = distance_correlation(y, x)
        self.assertAlmostEqual(
            swapped.distance_correlation, base.distance_correlation, places=13
        )
        angle = 0.7
        c, s = math.cos(angle), math.sin(angle)
        moved_x = [(c * u - s * v + 5.0, s * u + c * v - 2.0) for u, v in x]
        # A reflection composed with a permutation of axes, then a shift.
        moved_y = [(w + 1.0, -u, v - 3.0) for u, v, w in y]
        moved = distance_correlation(moved_x, moved_y)
        self.assertAlmostEqual(
            moved.distance_correlation, base.distance_correlation, places=12
        )
        self.assertAlmostEqual(
            moved.distance_covariance, base.distance_covariance, places=12
        )
        scaled = distance_correlation([(3.0 * u, 3.0 * v) for u, v in x], y)
        self.assertAlmostEqual(
            scaled.distance_correlation, base.distance_correlation, places=12
        )
        self.assertAlmostEqual(
            scaled.distance_covariance,
            math.sqrt(3.0) * base.distance_covariance,
            places=12,
        )

    def test_matches_the_direct_definition(self) -> None:
        """A literal double-centring gives the same squared covariance."""
        rng = random.Random(16)
        n = 9
        x = [rng.random() for _ in range(n)]
        y = [rng.random() for _ in range(n)]

        def centred(values: list[float]) -> list[list[float]]:
            """Return the double-centred distance matrix, entry by entry."""
            d = [[abs(u - v) for v in values] for u in values]
            row = [sum(r) / n for r in d]
            grand = sum(row) / n
            return [
                [d[k][j] - row[k] - row[j] + grand for j in range(n)] for k in range(n)
            ]

        a, b = centred(x), centred(y)
        dcov2 = sum(a[k][j] * b[k][j] for k in range(n) for j in range(n)) / n**2
        dvx = sum(v * v for r in a for v in r) / n**2
        dvy = sum(v * v for r in b for v in r) / n**2
        result = distance_correlation(x, y)
        self.assertAlmostEqual(result.distance_covariance**2, dcov2, places=14)
        self.assertAlmostEqual(
            result.distance_correlation,
            math.sqrt(dcov2 / math.sqrt(dvx * dvy)),
            places=13,
        )

    def test_bivariate_normal_closed_form_by_quadrature(self) -> None:
        """The closed form agrees with the three-expectation definition."""
        for rho in (0.0, 0.3, -0.6, 0.9):
            # Differences X - X', Y - Y' have variance 2 and correlation rho;
            # X - X', Y - Y'' have correlation rho / 2.
            joint = 2.0 * _expected_abs_product(rho)
            mixed = 2.0 * _expected_abs_product(0.5 * rho)
            product = (2.0 / math.sqrt(math.pi)) ** 2
            dcov2 = joint + product - 2.0 * mixed
            dvar2 = 2.0 * _expected_abs_product(1.0) + product
            dvar2 -= 4.0 * _expected_abs_product(0.5)
            expected = math.sqrt(max(dcov2, 0.0) / dvar2)
            self.assertAlmostEqual(
                distance_correlation_bivariate_normal(rho), expected, delta=1e-6
            )
        self.assertEqual(distance_correlation_bivariate_normal(0.0), 0.0)
        self.assertAlmostEqual(distance_correlation_bivariate_normal(1.0), 1.0)
        self.assertAlmostEqual(distance_correlation_bivariate_normal(-1.0), 1.0)
        grid = [distance_correlation_bivariate_normal(0.05 * i) for i in range(21)]
        self.assertTrue(all(b > a for a, b in zip(grid, grid[1:], strict=False)))
        self.assertTrue(all(v <= 0.05 * i + 1e-12 for i, v in enumerate(grid)))
        with self.assertRaises(ValueError):
            distance_correlation_bivariate_normal(1.5)

    def test_sample_value_approaches_the_normal_closed_form(self) -> None:
        """Simulated normal pairs: increasing in ``rho`` and near the limit."""
        rng = random.Random(17)
        n = 400
        previous = 0.0
        for rho in (0.3, 0.6, 0.9):
            x = [rng.gauss(0.0, 1.0) for _ in range(n)]
            y = [
                3.0 + 2.0 * (rho * u + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, 1.0))
                for u in x
            ]
            value = distance_correlation(x, y).distance_correlation
            self.assertAlmostEqual(
                value, distance_correlation_bivariate_normal(rho), delta=0.06
            )
            self.assertGreater(value, previous)
            previous = value

    def test_permutation_test(self) -> None:
        """Dependence is detected, independence is not; seeds reproduce."""
        rng = random.Random(18)
        x = [rng.gauss(0.0, 1.0) for _ in range(60)]
        noise = [rng.gauss(0.0, 1.0) for _ in x]
        dependent = distance_correlation(
            x, [v * v for v in x], permutations=199, seed=1
        )
        independent = distance_correlation(x, noise, permutations=199, seed=1)
        self.assertEqual(dependent.p_value, 1.0 / 200.0)
        self.assertGreater(independent.p_value or 0.0, 0.05)
        self.assertEqual(
            independent, distance_correlation(x, noise, permutations=199, seed=1)
        )
        other = distance_correlation(x, noise, permutations=199, seed=2)
        self.assertEqual(other.distance_correlation, independent.distance_correlation)
        self.assertNotEqual(other.p_value, independent.p_value)

    def test_permutation_p_values_are_uniform_under_independence(self) -> None:
        """Rejection frequency at level 0.1 matches the level."""
        rng = random.Random(19)
        replicates, rejections = 200, 0
        for replicate in range(replicates):
            x = [rng.random() for _ in range(15)]
            y = [rng.gauss(0.0, 1.0) for _ in range(15)]
            result = distance_correlation(x, y, permutations=39, seed=replicate)
            rejections += (result.p_value or 1.0) <= 0.1
        self.assertLess(abs(rejections / replicates - 0.1), 4.0 * math.sqrt(0.09 / 200))

    def test_degenerate_and_invalid_inputs(self) -> None:
        """A constant sample gives 0; malformed inputs raise."""
        constant = distance_correlation([1.0, 1.0, 1.0], [0.0, 2.0, 5.0])
        self.assertEqual(constant.distance_correlation, 0.0)
        self.assertEqual(constant.distance_variance_x, 0.0)
        with self.assertRaises(ValueError):
            distance_correlation([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            distance_correlation([1.0], [1.0])
        with self.assertRaises(ValueError):
            distance_correlation([(1.0, 2.0), (1.0,)], [1.0, 2.0])
        with self.assertRaises(ValueError):
            distance_correlation([1.0, 2.0], [1.0, 2.0], permutations=-1)
        with self.assertRaises(ValueError):
            distance_correlation([], [])


if __name__ == "__main__":
    unittest.main()
