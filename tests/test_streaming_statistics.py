"""Tests for one-pass streaming accumulators in ``probstat.statistics``."""

from __future__ import annotations

import bisect
import math
import random
import unittest
from fractions import Fraction
from statistics import NormalDist

from probstat.foundations import RandomStream
from probstat.inference import wilson_score_interval
from probstat.monte_carlo import estimate_mean_streaming, estimate_probability
from probstat.statistics import (
    ExceedanceCounter,
    P2Quantile,
    ReservoirSampler,
    RunningCovariance,
    RunningMoments,
    WeightedRunningMoments,
    covariance,
    excess_kurtosis,
    pearson_correlation,
    quantile,
    run_until,
    skewness,
    variance,
)

TIGHT = 1e-12


def _relative(actual: float, expected: float) -> float:
    """Return ``|actual - expected| / |expected|``."""
    return abs(actual - expected) / abs(expected)


def _exact_central(
    values: list[float],
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return the exact rational mean and central moments ``m2, m3, m4``."""
    exact = [Fraction(value) for value in values]
    n = len(exact)
    center = sum(exact) / n
    m2 = sum((v - center) ** 2 for v in exact) / n
    m3 = sum((v - center) ** 3 for v in exact) / n
    m4 = sum((v - center) ** 4 for v in exact) / n
    return center, m2, m3, m4


def _data(offset: float, size: int = 2000, seed: int = 7) -> list[float]:
    """Return skewed (exponential) data shifted by ``offset``."""
    generator = random.Random(seed)
    return [offset + generator.expovariate(1.0) for _ in range(size)]


class RunningMomentsTests(unittest.TestCase):
    """Welford/Pébay moments equal exact and two-pass computations."""

    def test_matches_exact_rational_moments_with_and_without_offset(self) -> None:
        """Mean, variance, skewness, kurtosis are exact to 1e-12 at offset 1e9."""
        for offset in (0.0, 1e9):
            values = _data(offset)
            center, m2, m3, m4 = _exact_central(values)
            moments = RunningMoments(values)
            g1 = float(m3) / float(m2) ** 1.5
            g2 = float(m4 / (m2 * m2)) - 3.0
            with self.subTest(offset=offset):
                self.assertEqual(moments.count, len(values))
                self.assertLess(_relative(moments.mean, float(center)), TIGHT)
                self.assertLess(_relative(moments.variance(), float(m2)), TIGHT)
                self.assertLess(_relative(moments.skewness(), g1), TIGHT)
                self.assertLess(_relative(moments.excess_kurtosis(), g2), TIGHT)

    def test_matches_two_pass_library_functions(self) -> None:
        """Variance equals two-pass ``variance``; shape matches at zero offset."""
        for offset in (0.0, 1e9):
            values = _data(offset, seed=11)
            moments = RunningMoments(values)
            with self.subTest(offset=offset):
                self.assertLess(
                    _relative(moments.variance(ddof=1), variance(values, ddof=1)),
                    TIGHT,
                )
        values = _data(0.0, seed=12)
        moments = RunningMoments(values)
        self.assertLess(_relative(moments.skewness(), skewness(values)), TIGHT)
        self.assertLess(
            _relative(moments.excess_kurtosis(), excess_kurtosis(values)), TIGHT
        )

    def test_merge_of_chunks_equals_single_pass(self) -> None:
        """Uneven chunk merges reproduce the single pass to 1e-12."""
        values = _data(1e9, size=3001, seed=5)
        single = RunningMoments(values)
        bounds = [0, 1, 17, 400, 1999, 2000, 3001]
        chunks = [
            RunningMoments(values[a:b])
            for a, b in zip(bounds, bounds[1:], strict=False)
        ]
        left_fold = chunks[0]
        for chunk in chunks[1:]:
            left_fold = left_fold.merge(chunk)
        tree = (
            chunks[0]
            .merge(chunks[1])
            .merge(chunks[2].merge(chunks[3]).merge(chunks[4].merge(chunks[5])))
        )
        for merged in (left_fold, tree, RunningMoments().merge(single)):
            self.assertEqual(merged.count, single.count)
            self.assertLess(_relative(merged.mean, single.mean), TIGHT)
            self.assertLess(_relative(merged.variance(), single.variance()), TIGHT)
            self.assertLess(_relative(merged.skewness(), single.skewness()), TIGHT)
            self.assertLess(
                _relative(merged.excess_kurtosis(), single.excess_kurtosis()), TIGHT
            )

    def test_standard_error_and_validation(self) -> None:
        """Standard error is ``s / sqrt(n)``; empty or degenerate input raises."""
        moments = RunningMoments([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(
            moments.standard_error(), math.sqrt(variance([1, 2, 3, 4], ddof=1) / 4)
        )
        with self.assertRaises(ValueError):
            _ = RunningMoments().mean
        with self.assertRaises(ValueError):
            RunningMoments([2.0, 2.0]).skewness()
        with self.assertRaises(ValueError):
            RunningMoments([1.0]).push(math.inf)
        with self.assertRaises(ValueError):
            moments.variance(ddof=4)


class WeightedAndCovarianceTests(unittest.TestCase):
    """Weighted moments and co-moments equal direct formulas."""

    def test_weighted_moments_match_direct_formulas(self) -> None:
        """West's update matches sum-based formulas, with merge and offset."""
        generator = random.Random(3)
        values = [1e9 + generator.gauss(0.0, 2.0) for _ in range(1500)]
        weights = [generator.uniform(0.1, 3.0) for _ in values]
        exact_w = [Fraction(w) for w in weights]
        exact_x = [Fraction(x) for x in values]
        total = sum(exact_w)
        center = sum(w * x for w, x in zip(exact_w, exact_x, strict=True)) / total
        s = sum(w * (x - center) ** 2 for w, x in zip(exact_w, exact_x, strict=True))
        squares = sum(w * w for w in exact_w)
        pairs = list(zip(values, weights, strict=True))
        single = WeightedRunningMoments(pairs)
        merged = WeightedRunningMoments(pairs[:700]).merge(
            WeightedRunningMoments(pairs[700:])
        )
        for moments in (single, merged):
            self.assertLess(_relative(moments.mean, float(center)), TIGHT)
            self.assertLess(_relative(moments.variance(), float(s / total)), TIGHT)
            self.assertLess(
                _relative(moments.variance(kind="frequency"), float(s / (total - 1))),
                TIGHT,
            )
            self.assertLess(
                _relative(
                    moments.variance(kind="reliability"),
                    float(s / (total - squares / total)),
                ),
                TIGHT,
            )
            self.assertLess(
                _relative(moments.effective_sample_size, float(total**2 / squares)),
                TIGHT,
            )

    def test_unit_weights_reduce_to_unweighted(self) -> None:
        """Unit weights give the ordinary mean and ``ddof=1`` variance."""
        values = _data(0.0, size=500, seed=9)
        weighted = WeightedRunningMoments((x, 1.0) for x in values)
        weighted.push(123.0, 0.0)
        plain = RunningMoments(values)
        self.assertEqual(weighted.count, 501)
        self.assertLess(_relative(weighted.mean, plain.mean), TIGHT)
        self.assertLess(
            _relative(weighted.variance(kind="frequency"), plain.variance(ddof=1)),
            TIGHT,
        )
        with self.assertRaises(ValueError):
            weighted.variance(kind="unknown")
        with self.assertRaises(ValueError):
            weighted.push(1.0, -1.0)

    def test_covariance_matches_two_pass_and_merges(self) -> None:
        """Pair covariance and correlation match two-pass values at offset 1e9."""
        generator = random.Random(21)
        xs = [1e9 + generator.gauss(0.0, 1.0) for _ in range(2000)]
        ys = [-5e8 + 0.6 * (x - 1e9) + generator.gauss(0.0, 0.8) for x in xs]
        zs = [generator.expovariate(2.0) for _ in xs]
        pairs = list(zip(xs, ys, strict=True))
        single = RunningCovariance(2)
        single.extend(pairs)
        left = RunningCovariance(2)
        left.extend(pairs[:1234])
        right = RunningCovariance(2)
        right.extend(pairs[1234:])
        for accumulator in (single, left.merge(right)):
            self.assertLess(
                _relative(accumulator.covariance(), covariance(xs, ys)), TIGHT
            )
            self.assertLess(
                _relative(accumulator.covariance(ddof=1), covariance(xs, ys, ddof=1)),
                TIGHT,
            )
            self.assertLess(
                _relative(accumulator.correlation(), pearson_correlation(xs, ys)),
                TIGHT,
            )
        triple = RunningCovariance(3)
        triple.extend(zip(xs, ys, zs, strict=True))
        matrix = triple.covariance_matrix(ddof=1)
        self.assertLess(_relative(matrix[2][2], variance(zs, ddof=1)), TIGHT)
        self.assertLess(_relative(matrix[0][2], covariance(xs, zs, ddof=1)), TIGHT)
        self.assertEqual(matrix[0][2], matrix[2][0])
        self.assertEqual(triple.correlation_matrix()[1][1], 1.0)
        with self.assertRaises(ValueError):
            triple.push((1.0, 2.0))


class StreamingQuantileTests(unittest.TestCase):
    """P² estimates and reservoir samples behave as documented."""

    def test_p2_tracks_known_quantiles(self) -> None:
        """|F(q) - p| <= 0.01 and empirical rank error <= 0.002 at n = 2e4."""
        normal = NormalDist()
        cases = (
            ("normal", normal.cdf, lambda g: g.gauss(0.0, 1.0)),
            ("exponential", lambda x: 1.0 - math.exp(-x), lambda g: g.expovariate(1.0)),
        )
        for name, cdf, draw in cases:
            for probability in (0.5, 0.9, 0.99):
                generator = random.Random(100)
                data = [draw(generator) for _ in range(20_000)]
                estimator = P2Quantile(probability)
                estimator.extend(data)
                ordered = sorted(data)
                rank = bisect.bisect_left(ordered, estimator.estimate) / len(data)
                with self.subTest(distribution=name, p=probability):
                    self.assertLessEqual(
                        abs(cdf(estimator.estimate) - probability), 0.01
                    )
                    self.assertLessEqual(abs(rank - probability), 0.002)

    def test_p2_median_at_one_hundred_thousand(self) -> None:
        """The normal median estimate at n = 1e5 is within 0.01 of zero."""
        generator = random.Random(4)
        estimator = P2Quantile(0.5)
        estimator.extend(generator.gauss(0.0, 1.0) for _ in range(100_000))
        self.assertEqual(estimator.count, 100_000)
        self.assertLess(abs(estimator.estimate), 0.01)

    def test_p2_small_samples_are_exact(self) -> None:
        """Up to five observations, the estimate is the type-7 quantile."""
        estimator = P2Quantile(0.3)
        values = [5.0, 1.0, 4.0, 2.0]
        estimator.extend(values)
        self.assertEqual(estimator.estimate, quantile(values, 0.3))
        with self.assertRaises(ValueError):
            P2Quantile(1.0)

    def test_reservoir_is_deterministic_by_seed(self) -> None:
        """Equal seeds give equal samples; different seeds differ."""

        def draw(seed: int | RandomStream) -> tuple[float, ...]:
            """Return a reservoir sample of 0..999 for one seed."""
            sampler = ReservoirSampler(10, seed=seed)
            sampler.extend(float(i) for i in range(1000))
            return sampler.sample

        self.assertEqual(draw(42), draw(42))
        self.assertEqual(draw(RandomStream(42)), draw(42))
        self.assertNotEqual(draw(42), draw(43))
        short = ReservoirSampler(10, seed=1)
        short.extend([1.0, 2.0, 3.0])
        self.assertEqual(short.sample, (1.0, 2.0, 3.0))

    def test_reservoir_inclusion_is_uniform(self) -> None:
        """Inclusion counts of 20 items (k = 5, 4000 runs) pass chi-square."""
        root = RandomStream(2026)
        trials, items, capacity = 4000, 20, 5
        counts = [0] * items
        for stream in root.spawn(trials, label="reservoir"):
            sampler = ReservoirSampler(capacity, seed=stream)
            sampler.extend(float(i) for i in range(items))
            for value in sampler.sample:
                counts[int(value)] += 1
        expected = trials * capacity / items
        statistic = sum((c - expected) ** 2 / expected for c in counts)
        self.assertLess(statistic, 43.82)  # chi-square(19) upper 0.001 point


class ExceedanceAndStoppingTests(unittest.TestCase):
    """Exceedance counting, the Wilson interval, and relative-error stopping."""

    def test_exceedance_counter_matches_library_wilson(self) -> None:
        """Counts, merge, and Wilson interval agree with the inference helper."""
        generator = random.Random(8)
        values = [generator.gauss(0.0, 1.0) for _ in range(5000)]
        counter = ExceedanceCounter(threshold=1.5)
        counter.extend(values)
        events = sum(v > 1.5 for v in values)
        self.assertEqual((counter.events, counter.trials), (events, 5000))
        reference = wilson_score_interval(events, 5000, confidence=0.9)
        lower, upper = counter.wilson_interval(confidence=0.9)
        self.assertAlmostEqual(lower, reference.lower, places=15)
        self.assertAlmostEqual(upper, reference.upper, places=15)
        left = ExceedanceCounter(1.5)
        left.extend(values[:100])
        right = ExceedanceCounter(1.5)
        right.extend(values[100:])
        self.assertEqual(left.merge(right).events, events)
        indicator = ExceedanceCounter()
        indicator.extend([True, False, False])
        self.assertEqual(indicator.wilson_interval()[0] > 0.0, True)
        with self.assertRaises(TypeError):
            indicator.push(1.0)
        with self.assertRaises(ValueError):
            left.merge(ExceedanceCounter(2.0))

    def test_streaming_probability_estimate_accepts_generators(self) -> None:
        """``estimate_probability`` consumes a generator without storing it."""
        generator = random.Random(1)
        estimate = estimate_probability(generator.random() < 0.25 for _ in range(4000))
        self.assertEqual(estimate.trials, 4000)
        self.assertLess(abs(estimate.probability - 0.25), 0.03)

    def test_run_until_reaches_target(self) -> None:
        """The running mean of Exp(1) stops once SE/|mean| <= 1%."""
        generator = random.Random(17)
        result = run_until(
            lambda: generator.expovariate(1.0),
            target_relative_error=0.01,
            max_samples=100_000,
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(result.relative_error, 0.01)
        self.assertLess(result.sample_size, 100_000)
        # SE/mean = 1/sqrt(n) for Exp(1): about 1e4 draws are required.
        self.assertGreater(result.sample_size, 8_000)
        self.assertLess(result.sample_size, 12_500)
        self.assertLess(abs(result.mean - 1.0), 4 * result.standard_error)
        self.assertEqual(result.trace[-1][0], result.sample_size)
        self.assertEqual(result.moments.count, result.sample_size)

    def test_run_until_reports_budget_exhaustion(self) -> None:
        """An unreachable target stops at the budget; a short iterable stops early."""
        generator = random.Random(2)
        result = run_until(
            lambda: generator.gauss(0.0, 1.0),
            target_relative_error=0.01,
            max_samples=1000,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.sample_size, 1000)
        short = run_until(
            iter([1.0, 2.0, 3.0]), target_relative_error=1e-6, max_samples=500
        )
        self.assertEqual(short.sample_size, 3)
        self.assertFalse(short.converged)
        with self.assertRaises(ValueError):
            run_until([1.0], target_relative_error=0.1, max_samples=200)

    def test_estimate_mean_streaming_matches_stored_estimate(self) -> None:
        """Fixed-budget streaming equals the stored-sample mean and variance."""
        generator = random.Random(6)
        values = [generator.uniform(2.0, 4.0) for _ in range(3000)]
        estimate = estimate_mean_streaming(iter(values), max_samples=3000)
        self.assertEqual(estimate.sample_size, 3000)
        self.assertEqual(estimate.evaluations, 3000)
        self.assertLess(_relative(estimate.estimate, math.fsum(values) / 3000), TIGHT)
        self.assertLess(
            _relative(estimate.sample_variance, variance(values, ddof=1)), TIGHT
        )
        constant = estimate_mean_streaming(lambda: 2.5, max_samples=50)
        self.assertEqual(constant.sample_size, 50)
        self.assertEqual(constant.standard_error, 0.0)
        targeted = estimate_mean_streaming(
            lambda: generator.uniform(2.0, 4.0),
            max_samples=100_000,
            target_relative_error=0.002,
        )
        self.assertLessEqual(targeted.standard_error / targeted.estimate, 0.002)
        self.assertLess(targeted.sample_size, 100_000)


if __name__ == "__main__":
    unittest.main()
