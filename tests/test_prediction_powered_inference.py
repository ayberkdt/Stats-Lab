"""Tests for prediction-powered inference for a mean.

Oracles: the variance formula against the empirical variance of the
estimator over seeded replicates of jointly Gaussian ``(Y, f)`` with known
covariance; the optimal weight against a direct search and against the
closed-form minimum variance; the classical z interval of the library for
weight zero; hand-computed moments for weight one; and coverage of the
interval within binomial error.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.foundations import RandomStream
from probstat.inference import (
    PredictionPoweredMean,
    normal_mean_confidence_interval,
    prediction_powered_mean,
    prediction_powered_variance,
    prediction_powered_weight,
)

MEAN_Y = 2.0
VAR_Y = 4.0
VAR_F = 2.25
RHO = 0.8
COV = RHO * math.sqrt(VAR_Y * VAR_F)
BIAS = 1.5


def _draw(
    generator: random.Random, n: int, big_n: int
) -> tuple[list[float], list[float], list[float]]:
    """Draw labels, their predictions, and unlabelled predictions.

    ``(Y, f)`` is bivariate normal with variances ``VAR_Y``, ``VAR_F`` and
    correlation ``RHO``; the predictions are biased by ``BIAS``.
    """
    sd_y, sd_f = math.sqrt(VAR_Y), math.sqrt(VAR_F)
    labels: list[float] = []
    predictions: list[float] = []
    for _ in range(n):
        a, b = generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)
        labels.append(MEAN_Y + sd_y * a)
        predictions.append(
            MEAN_Y + BIAS + sd_f * (RHO * a + math.sqrt(1.0 - RHO * RHO) * b)
        )
    unlabelled = [
        MEAN_Y + BIAS + sd_f * generator.gauss(0.0, 1.0) for _ in range(big_n)
    ]
    return labels, predictions, unlabelled


class VarianceFormulaTests(unittest.TestCase):
    """The population variance and the optimal weight."""

    def test_variance_matches_simulation(self) -> None:
        """Empirical variance over replicates agrees for several weights."""
        n, big_n, replicates = 20, 60, 4000
        stream = RandomStream(101)
        samples = [
            _draw(stream.child("variance", r).generator(), n, big_n)
            for r in range(replicates)
        ]
        for weight in (0.0, 0.5, 1.0, 1.7, -0.6):
            estimates = [
                prediction_powered_mean(y, f, big, weight=weight).estimate
                for y, f, big in samples
            ]
            predicted = prediction_powered_variance(
                label_variance=VAR_Y,
                prediction_variance=VAR_F,
                covariance=COV,
                labelled_size=n,
                unlabelled_size=big_n,
                weight=weight,
            )
            empirical = statistics.variance(estimates)
            # Normal estimator: sd of a sample variance is v sqrt(2/(R-1)).
            tolerance = 4.0 * predicted * math.sqrt(2.0 / (replicates - 1))
            self.assertAlmostEqual(empirical, predicted, delta=tolerance)
            # Unbiased for every weight although the predictions are biased.
            error = statistics.fmean(estimates) - MEAN_Y
            self.assertLess(abs(error), 4.0 * math.sqrt(predicted / replicates))

    def test_variance_hand_values(self) -> None:
        """Weight zero and weight one reduce to textbook variances."""
        common = {
            "label_variance": VAR_Y,
            "prediction_variance": VAR_F,
            "covariance": COV,
            "labelled_size": 25,
            "unlabelled_size": 100,
        }
        self.assertAlmostEqual(
            prediction_powered_variance(weight=0.0, **common), VAR_Y / 25
        )
        # Var(Y - f)/n + Var(f)/N.
        self.assertAlmostEqual(
            prediction_powered_variance(weight=1.0, **common),
            (VAR_Y + VAR_F - 2 * COV) / 25 + VAR_F / 100,
        )

    def test_optimal_weight_minimizes_the_variance(self) -> None:
        """The closed-form weight beats a fine grid and weights 0 and 1."""
        generator = RandomStream(102).generator()
        for _ in range(200):
            var_y = generator.uniform(0.1, 5.0)
            var_f = generator.uniform(0.1, 5.0)
            rho = generator.uniform(-0.99, 0.99)
            cov = rho * math.sqrt(var_y * var_f)
            n = generator.randint(1, 200)
            big_n = generator.randint(1, 5000)

            def variance(weight: float) -> float:
                """Return the predicted variance at one weight."""
                return prediction_powered_variance(
                    label_variance=var_y,  # noqa: B023
                    prediction_variance=var_f,  # noqa: B023
                    covariance=cov,  # noqa: B023
                    labelled_size=n,  # noqa: B023
                    unlabelled_size=big_n,  # noqa: B023
                    weight=weight,
                )

            best = prediction_powered_weight(
                prediction_variance=var_f,
                covariance=cov,
                labelled_size=n,
                unlabelled_size=big_n,
            )
            self.assertAlmostEqual(best, cov / ((1 + n / big_n) * var_f))
            minimum = variance(best)
            self.assertLessEqual(minimum, variance(0.0) + 1e-15)
            self.assertLessEqual(minimum, variance(1.0) + 1e-15)
            for step in range(-40, 41):
                self.assertLessEqual(minimum, variance(best + 0.05 * step) + 1e-15)
            closed = var_y * (1 - rho * rho / (1 + n / big_n)) / n
            self.assertAlmostEqual(minimum, closed, places=12)

    def test_degenerate_and_invalid_inputs(self) -> None:
        """Constant predictions give weight zero; bad moments raise."""
        self.assertEqual(
            prediction_powered_weight(
                prediction_variance=0.0,
                covariance=0.0,
                labelled_size=5,
                unlabelled_size=5,
            ),
            0.0,
        )
        with self.assertRaises(ValueError):
            prediction_powered_variance(
                label_variance=1.0,
                prediction_variance=1.0,
                covariance=1.5,
                labelled_size=5,
                unlabelled_size=5,
                weight=1.0,
            )
        with self.assertRaises(ValueError):
            prediction_powered_variance(
                label_variance=-1.0,
                prediction_variance=1.0,
                covariance=0.0,
                labelled_size=5,
                unlabelled_size=5,
                weight=1.0,
            )
        with self.assertRaises(ValueError):
            prediction_powered_weight(
                prediction_variance=1.0,
                covariance=0.5,
                labelled_size=0,
                unlabelled_size=5,
            )


class PredictionPoweredMeanTests(unittest.TestCase):
    """The estimator and its interval."""

    def setUp(self) -> None:
        """Draw one seeded data set."""
        self.labels, self.predictions, self.unlabelled = _draw(
            RandomStream(103).generator(), 60, 900
        )

    def test_weight_zero_is_the_classical_interval(self) -> None:
        """``weight=0`` reproduces the library's estimated-sigma z interval."""
        result = prediction_powered_mean(
            self.labels, self.predictions, self.unlabelled, weight=0.0
        )
        classical = normal_mean_confidence_interval(self.labels)
        self.assertIsInstance(result, PredictionPoweredMean)
        self.assertAlmostEqual(result.estimate, statistics.fmean(self.labels))
        self.assertAlmostEqual(result.interval.lower, classical.lower, places=12)
        self.assertAlmostEqual(result.interval.upper, classical.upper, places=12)
        self.assertAlmostEqual(result.variance_ratio, 1.0)
        self.assertFalse(result.tuned)

    def test_weight_one_is_the_original_estimator(self) -> None:
        """``mean_N(F) + mean_n(Y - f)`` with its two variance terms."""
        result = prediction_powered_mean(
            self.labels, self.predictions, self.unlabelled, weight=1.0
        )
        residuals = [y - f for y, f in zip(self.labels, self.predictions, strict=True)]
        self.assertAlmostEqual(
            result.estimate,
            statistics.fmean(self.unlabelled) + statistics.fmean(residuals),
        )
        variance = (
            statistics.variance(residuals) / 60
            + statistics.variance(self.unlabelled) / 900
        )
        self.assertAlmostEqual(result.standard_error, math.sqrt(variance), places=12)
        self.assertAlmostEqual(result.rectifier, statistics.fmean(residuals))
        self.assertAlmostEqual(
            result.interval.upper - result.estimate,
            1.959963984540054 * result.standard_error,
            places=9,
        )

    def test_tuned_weight_minimizes_the_estimated_variance(self) -> None:
        """The tuned standard error is never above any fixed-weight one."""
        tuned = prediction_powered_mean(self.labels, self.predictions, self.unlabelled)
        self.assertTrue(tuned.tuned)
        self.assertEqual(tuned.labelled_size, 60)
        self.assertEqual(tuned.unlabelled_size, 900)
        for step in range(-30, 31):
            weight = tuned.weight + 0.05 * step
            other = prediction_powered_mean(
                self.labels, self.predictions, self.unlabelled, weight=weight
            )
            self.assertLessEqual(tuned.standard_error, other.standard_error + 1e-14)
        for weight in (0.0, 1.0):
            other = prediction_powered_mean(
                self.labels, self.predictions, self.unlabelled, weight=weight
            )
            self.assertLess(tuned.standard_error, other.standard_error)
        self.assertLess(tuned.variance_ratio, 1.0)
        self.assertAlmostEqual(
            tuned.classical_standard_error,
            statistics.stdev(self.labels) / math.sqrt(60),
        )

    def test_tuned_weight_converges_to_the_population_optimum(self) -> None:
        """With many observations the tuned weight approaches ``w*``."""
        labels, predictions, unlabelled = _draw(
            RandomStream(104).generator(), 20_000, 40_000
        )
        tuned = prediction_powered_mean(labels, predictions, unlabelled)
        optimum = prediction_powered_weight(
            prediction_variance=VAR_F,
            covariance=COV,
            labelled_size=20_000,
            unlabelled_size=40_000,
        )
        self.assertAlmostEqual(tuned.weight, optimum, delta=0.02)
        predicted = prediction_powered_variance(
            label_variance=VAR_Y,
            prediction_variance=VAR_F,
            covariance=COV,
            labelled_size=20_000,
            unlabelled_size=40_000,
            weight=optimum,
        )
        self.assertAlmostEqual(tuned.standard_error**2 / predicted, 1.0, delta=0.05)

    def test_coverage_within_binomial_error(self) -> None:
        """Intervals cover the true mean at the nominal rate."""
        replicates, confidence = 2000, 0.9
        stream = RandomStream(105)
        covered = {"tuned": 0, "one": 0, "zero": 0}
        ratio = 0.0
        for replicate in range(replicates):
            labels, predictions, unlabelled = _draw(
                stream.child("coverage", replicate).generator(), 100, 400
            )
            for name, weight in (("tuned", None), ("one", 1.0), ("zero", 0.0)):
                result = prediction_powered_mean(
                    labels,
                    predictions,
                    unlabelled,
                    weight=weight,
                    confidence=confidence,
                )
                interval = result.interval
                covered[name] += interval.lower <= MEAN_Y <= interval.upper
                if name == "tuned":
                    ratio += result.variance_ratio / replicates
        band = 3.5 * math.sqrt(confidence * (1 - confidence) / replicates)
        for count in covered.values():
            self.assertAlmostEqual(count / replicates, confidence, delta=band)
        # Predicted gain: 1 - rho^2 / (1 + n/N) = 1 - 0.64 / 1.25 = 0.488.
        self.assertAlmostEqual(ratio, 0.488, delta=0.03)

    def test_useless_predictions_do_not_hurt(self) -> None:
        """Constant or anti-correlated predictions are handled."""
        constant = prediction_powered_mean(self.labels, [3.0] * 60, [3.0] * 10)
        self.assertEqual(constant.weight, 0.0)
        self.assertAlmostEqual(constant.estimate, constant.classical_estimate)
        flipped = prediction_powered_mean(
            self.labels,
            [-value for value in self.predictions],
            [-value for value in self.unlabelled],
        )
        tuned = prediction_powered_mean(self.labels, self.predictions, self.unlabelled)
        self.assertLess(flipped.weight, 0.0)
        self.assertAlmostEqual(flipped.estimate, tuned.estimate, places=12)
        self.assertAlmostEqual(flipped.standard_error, tuned.standard_error, places=12)
        degenerate = prediction_powered_mean(
            [1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.0, 3.0]
        )
        self.assertEqual(degenerate.standard_error, 0.0)
        self.assertTrue(math.isnan(degenerate.variance_ratio))

    def test_validation(self) -> None:
        """Short or mismatched samples and bad levels are rejected."""
        with self.assertRaises(ValueError):
            prediction_powered_mean([1.0, 2.0], [1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            prediction_powered_mean([1.0], [1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            prediction_powered_mean([1.0, 2.0], [1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            prediction_powered_mean([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], confidence=1.0)
        with self.assertRaises(ValueError):
            prediction_powered_mean([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], weight=math.inf)
        with self.assertRaises(TypeError):
            prediction_powered_mean([1.0, "a"], [1.0, 2.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
