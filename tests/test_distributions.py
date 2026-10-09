"""Behavioral tests for initial probability distributions."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Bernoulli, Normal


class DistributionTests(unittest.TestCase):
    """Verify probability identities, moments, bounds, and reproducibility."""

    def test_bernoulli_mass_and_moments(self) -> None:
        """Bernoulli masses normalize and moments match their equations."""
        distribution = Bernoulli(0.3)
        self.assertAlmostEqual(distribution.pmf(0) + distribution.pmf(1), 1.0)
        self.assertEqual(distribution.pmf(2), 0.0)
        self.assertEqual(distribution.mean, 0.3)
        self.assertAlmostEqual(distribution.variance, 0.21)
        self.assertEqual(distribution.cdf(-1.0), 0.0)
        self.assertEqual(distribution.cdf(0.0), 0.7)
        self.assertEqual(distribution.cdf(1.0), 1.0)
        self.assertEqual(distribution.support.finite_values(), (0, 1))
        self.assertAlmostEqual(math.exp(distribution.log_pmf(1)), 0.3)

    def test_normal_reference_values(self) -> None:
        """Standard normal density and CDF match analytic center values."""
        distribution = Normal()
        self.assertAlmostEqual(distribution.pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(distribution.cdf(0.0), 0.5)
        self.assertEqual(distribution.mean, 0.0)
        self.assertEqual(distribution.variance, 1.0)
        self.assertTrue(distribution.support.contains(-1e100))
        self.assertAlmostEqual(
            math.exp(distribution.log_pdf(0.0)), distribution.pdf(0.0)
        )

    def test_sampling_is_call_reproducible(self) -> None:
        """Equal seeds produce equal samples without shared mutable RNG state."""
        self.assertEqual(
            Bernoulli(0.4).sample(20, seed=11),
            Bernoulli(0.4).sample(20, seed=11),
        )
        self.assertEqual(
            Normal(2.0, 3.0).sample(10, seed=9),
            Normal(2.0, 3.0).sample(10, seed=9),
        )

    def test_invalid_parameters_are_rejected(self) -> None:
        """Out-of-domain probabilities and scales fail at construction."""
        with self.assertRaises(ValueError):
            Bernoulli(1.1)
        with self.assertRaises(ValueError):
            Normal(sigma=0.0)
        with self.assertRaises(ValueError):
            Normal(mu=math.nan)


if __name__ == "__main__":
    unittest.main()

