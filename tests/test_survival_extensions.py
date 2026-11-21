"""Tests for competing-risks cumulative incidence and restricted-mean contrasts."""

from __future__ import annotations

import math
import random
import unittest
from statistics import NormalDist

from probstat.inference.survival import (
    compare_restricted_means,
    cumulative_incidence,
    cumulative_incidence_curves,
    kaplan_meier,
    restricted_mean_survival,
    survival_data,
)


def _competing_sample(
    generator: random.Random, size: int, rates: tuple[float, float], censoring: float
) -> tuple[list[float], list[int]]:
    """Draw exponential latent failure times with exponential censoring."""
    times: list[float] = []
    causes: list[int] = []
    for _ in range(size):
        first = generator.expovariate(rates[0])
        second = generator.expovariate(rates[1])
        limit = generator.expovariate(censoring) if censoring > 0.0 else math.inf
        observed = min(first, second, limit)
        times.append(observed)
        causes.append(0 if observed == limit else (1 if observed == first else 2))
    return times, causes


class CumulativeIncidenceTests(unittest.TestCase):
    """Aalen-Johansen estimator against exact identities and simulation."""

    def test_hand_example(self) -> None:
        """A six-subject example reproduces the hand-computed curve."""
        times = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
        causes = [1, 2, 0, 1, 0, 2]
        one = cumulative_incidence(times, causes, cause=1)
        # t=1: n=6, F1 = 1/6, S = 5/6. t=2: n=5, cause 2, S = 5/6 * 4/5 = 2/3.
        # t=3: n=3, F1 = 1/6 + (2/3)(1/3) = 7/18, S = 4/9. t=5: n=1, cause 2.
        self.assertEqual(one.times, (1.0, 2.0, 3.0, 5.0))
        self.assertEqual(one.at_risk, (6, 5, 3, 1))
        self.assertEqual(one.cause_events, (1, 0, 1, 0))
        self.assertEqual(one.all_events, (1, 1, 1, 1))
        for value, expected in zip(
            one.incidence, (1 / 6, 1 / 6, 7 / 18, 7 / 18), strict=True
        ):
            self.assertAlmostEqual(value, expected, places=14)
        for value, expected in zip(
            one.overall_survival, (5 / 6, 2 / 3, 4 / 9, 0.0), strict=True
        ):
            self.assertAlmostEqual(value, expected, places=14)
        two = cumulative_incidence(times, causes, cause=2)
        self.assertAlmostEqual(two.incidence[-1], 1 / 6 + 4 / 9, places=14)
        self.assertEqual(one.incidence_at(0.5), 0.0)
        self.assertAlmostEqual(one.incidence_at(3.5), 7 / 18, places=14)
        self.assertEqual(one.interval_at(0.5), (0.0, 0.0))
        self.assertEqual(one.size, 6)
        self.assertEqual(one.last_time, 5.0)
        # First event time: binomial variance p (1 - p) / n with p = 1/6.
        self.assertAlmostEqual(one.variance[0], (1 / 6) * (5 / 6) / 6, places=14)

    def test_incidences_sum_to_one_minus_overall_survival(self) -> None:
        """``sum_k F_k = 1 - S`` at every event time, with ties and censoring."""
        generator = random.Random(5)
        times = [float(generator.randint(1, 15)) for _ in range(120)]
        causes = [generator.choice((0, 1, 1, 2, 3)) for _ in range(120)]
        curves = cumulative_incidence_curves(times, causes)
        self.assertEqual(tuple(curves), (1, 2, 3))
        overall = kaplan_meier(survival_data(times, [int(c > 0) for c in causes]))
        self.assertEqual(curves[1].times, overall.times)
        for index, survival in enumerate(overall.survival):
            total = sum(curve.incidence[index] for curve in curves.values())
            self.assertAlmostEqual(total, 1.0 - survival, places=13)
            self.assertAlmostEqual(
                curves[2].overall_survival[index], survival, places=13
            )
        self.assertEqual(
            curves[3].incidence, cumulative_incidence(times, causes, cause=3).incidence
        )

    def test_without_censoring_equals_the_empirical_subdistribution(self) -> None:
        """Uncensored data give the empirical fraction failed from the cause."""
        generator = random.Random(8)
        times, causes = _competing_sample(generator, 60, (0.7, 1.1), 0.0)
        estimate = cumulative_incidence(times, causes, cause=1)
        for point in (0.1, 0.4, 1.0, 3.0):
            empirical = sum(
                1 for t, c in zip(times, causes, strict=True) if t <= point and c == 1
            ) / len(times)
            self.assertAlmostEqual(estimate.incidence_at(point), empirical, places=13)
        # ... and the variance is the binomial F (1 - F) / n at every time.
        for value, variance in zip(estimate.incidence, estimate.variance, strict=True):
            self.assertAlmostEqual(variance, value * (1.0 - value) / 60, places=13)

    def test_single_cause_equals_one_minus_kaplan_meier_with_greenwood(self) -> None:
        """With one cause the estimator and its variance reduce to Kaplan-Meier."""
        generator = random.Random(21)
        times = [generator.expovariate(1.0) for _ in range(50)]
        events = [int(generator.random() < 0.7) for _ in range(50)]
        estimate = cumulative_incidence(times, events, cause=1)
        curve = kaplan_meier(survival_data(times, events))
        for index, survival in enumerate(curve.survival):
            self.assertAlmostEqual(estimate.incidence[index], 1.0 - survival, places=13)
            self.assertAlmostEqual(
                estimate.variance[index], curve.variance[index], places=13
            )

    def test_naive_kaplan_meier_complement_overestimates(self) -> None:
        """``1 - KM`` with competing events censored is never below ``F_k``."""
        generator = random.Random(3)
        times, causes = _competing_sample(generator, 200, (0.6, 0.9), 0.5)
        estimate = cumulative_incidence(times, causes, cause=1)
        naive = kaplan_meier(survival_data(times, [int(c == 1) for c in causes]))
        gaps = [
            1.0 - naive.survival_at(time) - value
            for time, value in zip(estimate.times, estimate.incidence, strict=True)
        ]
        self.assertTrue(all(gap >= -1e-13 for gap in gaps))
        self.assertGreater(gaps[-1], 0.05)

    def test_exponential_truth_and_variance_by_replication(self) -> None:
        """Mean, variance estimate, and coverage agree with the exponential truth."""
        rates = (0.6, 0.9)
        point = 1.0
        truth = rates[0] / sum(rates) * (1.0 - math.exp(-sum(rates) * point))
        naive_truth = 1.0 - math.exp(-rates[0] * point)
        generator = random.Random(1978)
        replicates = 400
        estimates: list[float] = []
        variances: list[float] = []
        naive: list[float] = []
        covered = 0
        for _ in range(replicates):
            times, causes = _competing_sample(generator, 120, rates, 0.5)
            estimate = cumulative_incidence(times, causes, cause=1)
            estimates.append(estimate.incidence_at(point))
            index = sum(1 for time in estimate.times if time <= point)
            variances.append(estimate.variance[index - 1])
            lower, upper = estimate.interval_at(point)
            covered += lower <= truth <= upper
            curve = kaplan_meier(survival_data(times, [int(c == 1) for c in causes]))
            naive.append(1.0 - curve.survival_at(point))
        mean = sum(estimates) / replicates
        spread = sum((value - mean) ** 2 for value in estimates) / (replicates - 1)
        self.assertLess(abs(mean - truth), 4.0 * math.sqrt(spread / replicates))
        # The replicate variance has relative standard error sqrt(2 / 399) = 7%.
        self.assertLess(abs(sum(variances) / replicates / spread - 1.0), 0.25)
        self.assertGreater(covered / replicates, 0.90)
        naive_mean = sum(naive) / replicates
        self.assertLess(abs(naive_mean - naive_truth), 0.02)
        self.assertGreater(naive_mean - mean, 0.1)

    def test_degenerate_inputs(self) -> None:
        """No events, absent causes, and invalid input are handled."""
        empty = cumulative_incidence([1.0, 2.0], [0, 0], cause=1)
        self.assertEqual(empty.times, ())
        self.assertEqual(empty.incidence_at(5.0), 0.0)
        self.assertEqual(dict(cumulative_incidence_curves([1.0, 2.0], [0, 0])), {})
        absent = cumulative_incidence([1.0, 2.0], [2, 2], cause=1)
        self.assertEqual(absent.incidence, (0.0, 0.0))
        self.assertEqual(absent.variance, (0.0, 0.0))
        self.assertEqual(absent.standard_errors, (0.0, 0.0))
        with self.assertRaises(ValueError):
            cumulative_incidence([1.0], [1], cause=0)
        with self.assertRaises(ValueError):
            cumulative_incidence([], [], cause=1)
        with self.assertRaises(ValueError):
            cumulative_incidence([1.0, 2.0], [1], cause=1)
        with self.assertRaises(ValueError):
            cumulative_incidence([-1.0], [1], cause=1)
        with self.assertRaises(ValueError):
            cumulative_incidence([1.0], [-1], cause=1)
        with self.assertRaises(TypeError):
            cumulative_incidence([1.0], [1.0], cause=1)
        with self.assertRaises(ValueError):
            cumulative_incidence([1.0], [1], cause=1, confidence=1.0)
        with self.assertRaises(ValueError):
            cumulative_incidence_curves([1.0], [1], confidence=0.0)


class RestrictedMeanTests(unittest.TestCase):
    """Restricted mean survival time and its two-sample contrasts."""

    def test_without_censoring_equals_the_truncated_sample_mean(self) -> None:
        """Uncensored data give ``mean(min(T, tau))`` and its sample variance."""
        generator = random.Random(4)
        times = [generator.expovariate(0.5) for _ in range(80)]
        tau = 2.5
        result = restricted_mean_survival(survival_data(times, [1] * 80), tau)
        truncated = [min(time, tau) for time in times]
        mean = sum(truncated) / 80
        self.assertAlmostEqual(result.estimate, mean, places=12)
        # Greenwood then reduces to the variance of the mean with divisor n.
        self.assertAlmostEqual(
            result.variance,
            sum((value - mean) ** 2 for value in truncated) / 80**2,
            places=12,
        )
        self.assertEqual(result.tau, tau)

    def test_function_matches_the_kaplan_meier_method(self) -> None:
        """The function is the method of the Kaplan-Meier estimate."""
        data = survival_data((1, 2, 3, 4, 4, 5, 6, 8), (1, 1, 1, 0, 1, 1, 0, 1))
        direct = kaplan_meier(data, confidence=0.9).restricted_mean(6.0)
        self.assertEqual(restricted_mean_survival(data, 6.0, confidence=0.9), direct)
        # Hand value: S = 1, 7/8, 3/4, 5/8, 1/2, 1/3 on unit intervals up to 6.
        self.assertAlmostEqual(
            restricted_mean_survival(data, 6.0).estimate, 49.0 / 12.0, places=12
        )

    def test_exponential_truth_and_variance_by_replication(self) -> None:
        """Mean, variance estimate, and coverage agree with the exponential truth."""
        rate, tau = 0.8, 2.0
        truth = (1.0 - math.exp(-rate * tau)) / rate
        generator = random.Random(1958)
        replicates = 300
        estimates: list[float] = []
        variances: list[float] = []
        covered = 0
        for _ in range(replicates):
            times: list[float] = []
            events: list[bool] = []
            for _ in range(200):
                failure = generator.expovariate(rate)
                limit = generator.uniform(1.0, 4.0)
                times.append(min(failure, limit))
                events.append(failure <= limit)
            times.append(4.0)
            events.append(False)
            result = restricted_mean_survival(survival_data(times, events), tau)
            estimates.append(result.estimate)
            variances.append(result.variance)
            covered += result.lower <= truth <= result.upper
        mean = sum(estimates) / replicates
        spread = sum((value - mean) ** 2 for value in estimates) / (replicates - 1)
        self.assertLess(abs(mean - truth), 4.0 * math.sqrt(spread / replicates))
        self.assertLess(abs(sum(variances) / replicates / spread - 1.0), 0.25)
        self.assertGreater(covered / replicates, 0.90)

    def test_comparison_arithmetic(self) -> None:
        """Difference, ratio, intervals, and test follow from the two arms."""
        first = survival_data((2, 3, 5, 6, 7, 8, 9, 9), (1, 0, 1, 1, 0, 1, 0, 0))
        second = survival_data((1, 1, 2, 3, 4, 5, 6, 9), (1, 1, 1, 1, 0, 1, 1, 0))
        tau = 8.0
        result = compare_restricted_means(first, second, tau, confidence=0.9)
        one = restricted_mean_survival(first, tau, confidence=0.9)
        two = restricted_mean_survival(second, tau, confidence=0.9)
        self.assertEqual(result.first, one)
        self.assertEqual(result.second, two)
        z = NormalDist().inv_cdf(0.95)
        error = math.sqrt(one.variance + two.variance)
        self.assertAlmostEqual(result.difference, one.estimate - two.estimate)
        self.assertAlmostEqual(result.difference_standard_error, error)
        self.assertAlmostEqual(result.difference_lower, result.difference - z * error)
        self.assertAlmostEqual(result.difference_upper, result.difference + z * error)
        self.assertAlmostEqual(result.ratio, one.estimate / two.estimate)
        log_error = math.sqrt(
            one.variance / one.estimate**2 + two.variance / two.estimate**2
        )
        self.assertAlmostEqual(result.log_ratio_standard_error, log_error)
        self.assertAlmostEqual(
            math.log(result.ratio_upper / result.ratio_lower), 2.0 * z * log_error
        )
        self.assertAlmostEqual(result.ratio_lower * result.ratio_upper, result.ratio**2)
        self.assertAlmostEqual(result.statistic, result.difference / error)
        self.assertAlmostEqual(
            result.p_value, 2.0 * NormalDist().cdf(-abs(result.statistic))
        )
        self.assertGreater(result.difference, 0.0)
        swapped = compare_restricted_means(second, first, tau, confidence=0.9)
        self.assertAlmostEqual(swapped.difference, -result.difference)
        self.assertAlmostEqual(swapped.ratio, 1.0 / result.ratio)
        self.assertAlmostEqual(swapped.p_value, result.p_value)

    def test_comparison_calibration_under_equal_and_unequal_rates(self) -> None:
        """The test holds its level, and both intervals cover the truth."""
        tau = 1.5
        generator = random.Random(77)

        def arm(rate: float) -> object:
            """Draw one censored exponential arm of 80 subjects."""
            times: list[float] = [3.0]
            events: list[bool] = [False]
            for _ in range(80):
                failure = generator.expovariate(rate)
                limit = generator.uniform(1.0, 3.0)
                times.append(min(failure, limit))
                events.append(failure <= limit)
            return survival_data(times, events)

        replicates = 300
        rejections = 0
        for _ in range(replicates):
            rejections += (
                compare_restricted_means(arm(1.0), arm(1.0), tau).p_value < 0.05
            )  # type: ignore[arg-type]
        # Binomial(300, 0.05): mean 15, standard deviation 3.8.
        self.assertLess(abs(rejections - 15), 12)
        truths = [(1.0 - math.exp(-rate * tau)) / rate for rate in (0.5, 1.2)]
        difference_cover = ratio_cover = 0
        for _ in range(replicates):
            result = compare_restricted_means(arm(0.5), arm(1.2), tau)  # type: ignore[arg-type]
            difference_cover += (
                result.difference_lower
                <= truths[0] - truths[1]
                <= result.difference_upper
            )
            ratio_cover += (
                result.ratio_lower <= truths[0] / truths[1] <= result.ratio_upper
            )
        self.assertGreater(difference_cover / replicates, 0.90)
        self.assertGreater(ratio_cover / replicates, 0.90)

    def test_validation_and_degenerate_cases(self) -> None:
        """Unidentified horizons, zero variance, and bad levels are handled."""
        data = survival_data((1, 2, 3, 4), (1, 1, 0, 0))
        with self.assertRaises(ValueError):
            restricted_mean_survival(data, 5.0)
        with self.assertRaises(ValueError):
            restricted_mean_survival(data, 0.0)
        with self.assertRaises(ValueError):
            compare_restricted_means(data, data, 5.0)
        with self.assertRaises(ValueError):
            compare_restricted_means(data, data, 2.0, confidence=1.2)
        censored = survival_data((3.0, 4.0), (0, 0))
        result = compare_restricted_means(censored, censored, 2.0)
        self.assertEqual(result.difference, 0.0)
        self.assertEqual(result.ratio, 1.0)
        self.assertTrue(math.isnan(result.p_value))
        self.assertEqual((result.ratio_lower, result.ratio_upper), (1.0, 1.0))
        immediate = survival_data((0.0, 0.0), (1, 1))
        with self.assertRaises(ValueError):
            compare_restricted_means(immediate, censored, 2.0)


if __name__ == "__main__":
    unittest.main()
