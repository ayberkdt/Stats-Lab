"""Behavioral tests for finite-sample descriptive statistics."""

from __future__ import annotations

import math
import unittest

from probstat.statistics import (
    covariance,
    empirical_cdf,
    mean,
    pearson_correlation,
    standard_deviation,
    variance,
    z_scores,
)


class DescriptiveStatisticsTests(unittest.TestCase):
    """Verify definitions, conventions, invariants, and invalid domains."""

    def test_location_and_spread(self) -> None:
        """Mean and both variance conventions match a hand-computed sample."""
        data = (1.0, 2.0, 3.0, 4.0)
        self.assertEqual(mean(data), 2.5)
        self.assertAlmostEqual(variance(data), 1.25)
        self.assertAlmostEqual(variance(data, ddof=1), 5.0 / 3.0)
        self.assertAlmostEqual(standard_deviation(data), math.sqrt(1.25))

    def test_z_scores_center_at_zero(self) -> None:
        """Standardized observations have numerical mean zero."""
        scores = z_scores((1.0, 2.0, 4.0, 8.0))
        self.assertAlmostEqual(mean(scores), 0.0)

    def test_empirical_cdf_is_right_continuous_at_observations(self) -> None:
        """Tied observations contribute their complete mass at the tie."""
        data = (1.0, 2.0, 2.0, 4.0)
        self.assertEqual(empirical_cdf(data, 0.0), 0.0)
        self.assertEqual(empirical_cdf(data, 2.0), 0.75)
        self.assertEqual(empirical_cdf(data, 5.0), 1.0)

    def test_covariance_and_correlation(self) -> None:
        """Perfect affine association has correlation one."""
        x = (1.0, 2.0, 3.0)
        y = (2.0, 4.0, 6.0)
        self.assertAlmostEqual(covariance(x, y), 4.0 / 3.0)
        self.assertAlmostEqual(pearson_correlation(x, y), 1.0)

    def test_invalid_samples_are_rejected(self) -> None:
        """Empty, non-finite, mismatched, and degenerate inputs fail."""
        with self.assertRaises(ValueError):
            mean(())
        with self.assertRaises(ValueError):
            variance((1.0,), ddof=1)
        with self.assertRaises(ValueError):
            mean((math.inf,))
        with self.assertRaises(ValueError):
            covariance((1.0,), (1.0, 2.0))
        with self.assertRaises(ValueError):
            pearson_correlation((1.0, 1.0), (2.0, 3.0))


if __name__ == "__main__":
    unittest.main()

