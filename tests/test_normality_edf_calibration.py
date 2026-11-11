"""Tests for fitted-null EDF normality tests (Lilliefors, CvM, AD).

Oracles: the null law of each fitted statistic is location-scale invariant,
so a seeded Monte Carlo simulation of standard-normal samples is an exact
(up to simulation error) reference for every p-value; the hand formula of
``W^2`` on a two-point sample; the existing Kolmogorov-Smirnov and
Anderson-Darling implementations on the same standardized sample; and the
published fact that the fully specified KS p-value is far too large when the
parameters are estimated (Lilliefors 1967).
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal, Uniform
from probstat.inference import (
    anderson_darling_test,
    cramer_von_mises_statistic,
    kolmogorov_smirnov_test,
    lilliefors_p_value,
    normality_tests,
)

SIMULATIONS = 4000


def _null_statistics(size: int, seed: int) -> list[tuple[float, ...]]:
    """Simulate fitted statistics and p-values for standard-normal samples.

    References:
        [EQ-INF-GOF-NORM-001], [REF-STEPHENS-1986].

    """
    rng = random.Random(seed)
    rows: list[tuple[float, ...]] = []
    for _ in range(SIMULATIONS):
        result = normality_tests([rng.gauss(0.0, 1.0) for _ in range(size)])
        rows.append(
            tuple(test.statistic for test in result.tests)
            + tuple(test.p_value for test in result.tests)
        )
    return rows


class NormalityCalibrationTests(unittest.TestCase):
    """The tabulated p-values reproduce the simulated fitted-null law."""

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate the null law once for n = 20.

        References:
            [REF-STEPHENS-1986].

        """
        cls.null = _null_statistics(20, seed=2026)

    def test_null_rejection_rates_match_nominal_levels(self) -> None:
        """Under normality each test rejects at about its nominal rate."""
        for alpha in (0.05, 0.10):
            tolerance = 4.0 * math.sqrt(alpha * (1 - alpha) / SIMULATIONS) + 0.005
            for column in range(3):
                rate = sum(row[3 + column] < alpha for row in self.null) / SIMULATIONS
                self.assertAlmostEqual(rate, alpha, delta=tolerance)

    def test_p_values_match_simulated_null_tail(self) -> None:
        """Formula p-values agree with the Monte Carlo tail of the same statistic."""
        rng = random.Random(7)
        samples = (
            [rng.gauss(5.0, 3.0) for _ in range(20)],
            [rng.expovariate(1.0) ** 0.5 for _ in range(20)],
            [rng.uniform(-1.0, 1.0) for _ in range(20)],
            [rng.lognormvariate(0.0, 0.6) for _ in range(20)],
        )
        for sample in samples:
            result = normality_tests(sample)
            for column, test in enumerate(result.tests):
                tail = sum(row[column] >= test.statistic for row in self.null)
                simulated = tail / SIMULATIONS
                se = math.sqrt(max(simulated * (1 - simulated), 1e-4) / SIMULATIONS)
                self.assertAlmostEqual(
                    test.p_value, simulated, delta=4.0 * se + 0.012, msg=test.name
                )

    def test_naive_ks_p_value_is_badly_conservative(self) -> None:
        """Ignoring estimation inflates the KS p-value (Lilliefors's point)."""
        rng = random.Random(3)
        sample = [rng.expovariate(1.0) for _ in range(40)]
        result = normality_tests(sample)
        fitted = Normal(result.mean, result.standard_deviation)
        naive = kolmogorov_smirnov_test(sample, fitted)
        self.assertAlmostEqual(naive.statistic, result.lilliefors.statistic, places=12)
        self.assertLess(result.lilliefors.p_value, 0.05)
        self.assertGreater(naive.p_value, 2.0 * result.lilliefors.p_value)

    def test_power_against_skewed_alternative(self) -> None:
        """Exponential samples of size 50 are rejected by every test."""
        rng = random.Random(11)
        result = normality_tests([rng.expovariate(2.0) for _ in range(50)])
        for test in result.tests:
            self.assertTrue(test.rejected(0.01), test.name)


class StatisticDefinitionTests(unittest.TestCase):
    """The statistics equal their defining formulas."""

    def test_cramer_von_mises_hand_formula(self) -> None:
        """Two points against U(0, 1): W^2 = 1/24 + (0.2-0.25)^2 + (0.9-0.75)^2."""
        value = cramer_von_mises_statistic([0.9, 0.2], Uniform(0.0, 1.0))
        expected = 1.0 / 24.0 + 0.05**2 + 0.15**2
        self.assertAlmostEqual(value, expected, places=15)

    def test_statistics_agree_with_existing_edf_code(self) -> None:
        """KS, AD, and CvM equal the generic routines on the fitted normal law."""
        rng = random.Random(5)
        sample = [rng.gauss(1.0, 2.0) for _ in range(30)]
        result = normality_tests(sample)
        mean = sum(sample) / len(sample)
        sd = math.sqrt(sum((x - mean) ** 2 for x in sample) / (len(sample) - 1))
        self.assertAlmostEqual(result.mean, mean, places=12)
        self.assertAlmostEqual(result.standard_deviation, sd, places=12)
        law = Normal(mean, sd)
        self.assertAlmostEqual(
            result.anderson_darling.statistic,
            anderson_darling_test(sample, law).statistic,
            places=10,
        )
        self.assertAlmostEqual(
            result.cramer_von_mises.statistic,
            cramer_von_mises_statistic(sample, law),
            places=12,
        )
        n = len(sample)
        self.assertAlmostEqual(
            result.anderson_darling.modified_statistic,
            result.anderson_darling.statistic * (1 + 0.75 / n + 2.25 / n**2),
            places=14,
        )

    def test_location_scale_invariance(self) -> None:
        """Affine maps with positive slope leave every statistic unchanged."""
        rng = random.Random(9)
        sample = [rng.gauss(0.0, 1.0) for _ in range(25)]
        base = normality_tests(sample)
        moved = normality_tests([10.0 + 4.0 * x for x in sample])
        for left, right in zip(base.tests, moved.tests, strict=True):
            self.assertAlmostEqual(left.statistic, right.statistic, places=10)
            self.assertAlmostEqual(left.p_value, right.p_value, places=9)

    def test_lilliefors_p_value_branches(self) -> None:
        """Small D gives p = 1, huge D gives p near 0, large n is rescaled."""
        self.assertEqual(lilliefors_p_value(0.01, 50), 1.0)
        self.assertLess(lilliefors_p_value(0.5, 50), 1e-10)
        self.assertGreater(lilliefors_p_value(0.12, 200), 0.0)
        previous = 1.0
        for step in range(1, 40):
            value = lilliefors_p_value(step / 100.0, 30)
            self.assertLessEqual(value, previous + 0.02)
            previous = value


class InvalidInputTests(unittest.TestCase):
    """Bad inputs raise informative errors."""

    def test_invalid_inputs(self) -> None:
        """Short, constant, or malformed inputs are rejected."""
        with self.assertRaises(ValueError):
            normality_tests([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            normality_tests([2.0] * 10)
        with self.assertRaises(ValueError):
            normality_tests([1.0, 2.0, math.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        with self.assertRaises(ValueError):
            lilliefors_p_value(1.5, 20)
        with self.assertRaises(ValueError):
            lilliefors_p_value(0.1, 4)
        with self.assertRaises(TypeError):
            lilliefors_p_value(0.1, 20.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cramer_von_mises_statistic([1.0], object())  # type: ignore[arg-type]
        result = normality_tests([float(i) for i in range(10)])
        with self.assertRaises(ValueError):
            result.lilliefors.rejected(1.0)


if __name__ == "__main__":
    unittest.main()
