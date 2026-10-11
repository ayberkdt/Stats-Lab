"""Behavioral tests for finite-sample descriptive statistics."""

from __future__ import annotations

import math
import unittest

from probstat.statistics import (
    covariance,
    empirical_cdf,
    gaussian_kde,
    kde_bandwidth,
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


class KernelDensityTests(unittest.TestCase):
    """Verify the Gaussian KDE against hand formulas and invariants."""

    def test_two_point_density_and_cdf_are_closed_form(self) -> None:
        """``f(x) = (phi((x+1)/h) + phi((x-1)/h)) / 2h`` for data (-1, 1)."""
        density = gaussian_kde([-1.0, 1.0], bandwidth=0.5)

        def phi(z: float) -> float:
            """Return the standard normal density."""
            return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

        for x in (-2.0, 0.0, 0.3, 1.0):
            want = (phi((x + 1.0) / 0.5) + phi((x - 1.0) / 0.5)) / (2.0 * 0.5)
            self.assertAlmostEqual(density.pdf(x), want, places=14)
        self.assertAlmostEqual(density.cdf(0.0), 0.5, places=14)

    def test_density_integrates_to_one(self) -> None:
        """The trapezoid integral over the default grid is one."""
        sample = [0.1 * i * i for i in range(30)]
        density = gaussian_kde(sample)
        grid = density.grid(2000, extend=8.0)
        step = grid[1] - grid[0]
        area = step * (
            math.fsum(density.pdf(x) for x in grid)
            - 0.5 * (density.pdf(grid[0]) + density.pdf(grid[-1]))
        )
        self.assertAlmostEqual(area, 1.0, places=6)

    def test_bandwidth_rules(self) -> None:
        """Silverman uses min(s, IQR/1.34); the normal rule uses s."""
        sample = [float(i) for i in range(1, 11)]
        s = math.sqrt(sum((v - 5.5) ** 2 for v in sample) / 9.0)
        n_factor = 10 ** (-0.2)
        self.assertAlmostEqual(
            kde_bandwidth(sample, rule="normal"), 1.06 * s * n_factor, places=14
        )
        iqr = 7.75 - 3.25
        self.assertAlmostEqual(
            kde_bandwidth(sample), 0.9 * min(s, iqr / 1.34) * n_factor, places=14
        )
        with self.assertRaises(ValueError):
            gaussian_kde([2.0, 2.0])
        with self.assertRaises(ValueError):
            gaussian_kde([1.0, 2.0], bandwidth=0.0)
        with self.assertRaises(ValueError):
            kde_bandwidth(sample, rule="scott")
