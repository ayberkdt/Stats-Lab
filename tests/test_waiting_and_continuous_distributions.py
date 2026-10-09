"""Tests for waiting-count and foundational continuous distributions."""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.distributions import (
    Bernoulli,
    Binomial,
    Exponential,
    Geometric,
    NegativeBinomial,
    Normal,
    Poisson,
    Uniform,
)
from probstat.distributions._special import regularized_beta
from probstat.foundations import (
    ContinuousProbabilityLaw,
    DiscreteProbabilityLaw,
    HazardFunction,
    LogProbabilityDensity,
    QuantileFunction,
    SurvivalFunction,
)


class RegularizedBetaTests(unittest.TestCase):
    """Verify incomplete-beta identities and numerical complement behavior."""

    def test_elementary_shapes_and_symmetry(self) -> None:
        """Simple polynomial cases and the beta symmetry identity agree."""
        for point in (0.0, 0.1, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(regularized_beta(point, 1.0, 1.0), point)
            self.assertAlmostEqual(regularized_beta(point, 2.0, 1.0), point**2)
            self.assertAlmostEqual(
                regularized_beta(point, 1.0, 2.0),
                1.0 - (1.0 - point) ** 2,
            )
            self.assertAlmostEqual(
                regularized_beta(point, 2.5, 4.0)
                + regularized_beta(1.0 - point, 4.0, 2.5),
                1.0,
                places=13,
            )

    def test_invalid_domains_are_rejected(self) -> None:
        """Beta evaluation rejects invalid points and shape parameters."""
        with self.assertRaises(ValueError):
            regularized_beta(-0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            regularized_beta(0.5, 0.0, 1.0)

    def test_positive_integer_shapes_match_binomial_tail_sums(self) -> None:
        """DLMF's binomial-expansion identity holds across shape regimes."""
        for first, second, point in (
            (2, 3, 0.2),
            (3, 7, 0.65),
            (10, 4, 0.8),
            (20, 25, 0.45),
        ):
            trials = first + second - 1
            expected = math.fsum(
                math.comb(trials, index)
                * point**index
                * (1.0 - point) ** (trials - index)
                for index in range(first, trials + 1)
            )
            self.assertAlmostEqual(
                regularized_beta(point, float(first), float(second)),
                expected,
                places=13,
            )


class GeometricTests(unittest.TestCase):
    """Verify the explicit failures-before-first-success convention."""

    def test_mass_tails_moments_and_quantiles(self) -> None:
        """Geometric identities agree under the zero-based parameterization."""
        law = Geometric(0.25)
        self.assertAlmostEqual(law.pmf(2), 0.25 * 0.75**2)
        self.assertAlmostEqual(law.cdf(2), 1.0 - 0.75**3)
        self.assertAlmostEqual(law.sf(2), 0.75**3)
        self.assertEqual(law.mean, 3.0)
        self.assertEqual(law.variance, 12.0)
        for target in (0.0, 0.1, 0.25, 0.5, 0.9):
            quantile = int(law.quantile(target))
            self.assertGreaterEqual(law.cdf(quantile), target)
            if quantile > 0:
                self.assertLess(law.cdf(quantile - 1), target)
        self.assertEqual(law.quantile(1.0), math.inf)

    def test_degenerate_probability_and_sampling(self) -> None:
        """Certain success is concentrated at zero and seeds are local."""
        certain = Geometric(1.0)
        self.assertEqual(certain.pmf(0), 1.0)
        self.assertEqual(certain.sample(5, seed=1), (0, 0, 0, 0, 0))
        law = Geometric(0.25)
        sample = law.sample(10_000, seed=91)
        self.assertEqual(sample, law.sample(10_000, seed=91))
        self.assertAlmostEqual(statistics.fmean(sample), law.mean, delta=0.15)


class NegativeBinomialTests(unittest.TestCase):
    """Verify failure counts before a positive target number of successes."""

    def test_mass_beta_tails_moments_and_geometric_reduction(self) -> None:
        """PMF sums, beta tails, and the r=1 reduction agree."""
        law = NegativeBinomial(3, 0.4)
        expected_mass = math.comb(4, 2) * 0.4**3 * 0.6**2
        expected_cdf = math.fsum(law.pmf(k) for k in range(3))
        self.assertAlmostEqual(law.pmf(2), expected_mass)
        self.assertAlmostEqual(law.cdf(2), expected_cdf)
        self.assertAlmostEqual(law.cdf(2) + law.sf(2), 1.0)
        self.assertAlmostEqual(law.mean, 4.5)
        self.assertAlmostEqual(law.variance, 11.25)
        geometric = Geometric(0.4)
        reduced = NegativeBinomial(1, 0.4)
        for k in range(8):
            self.assertAlmostEqual(reduced.pmf(k), geometric.pmf(k))

    def test_quantiles_boundaries_and_sampling(self) -> None:
        """Quantiles bracket targets and independent sums sample correctly."""
        law = NegativeBinomial(4, 0.55)
        for target in (0.0, 0.1, 0.5, 0.9, 0.999):
            quantile = int(law.quantile(target))
            self.assertGreaterEqual(law.cdf(quantile), target)
            if quantile > 0:
                self.assertLess(law.cdf(quantile - 1), target)
        sample = law.sample(10_000, seed=13)
        self.assertAlmostEqual(statistics.fmean(sample), law.mean, delta=0.15)
        self.assertEqual(NegativeBinomial(2, 1.0).sample(3), (0, 0, 0))
        with self.assertRaises(ValueError):
            NegativeBinomial(0, 0.5)


class UniformTests(unittest.TestCase):
    """Verify finite continuous uniform density, tails, and inversion."""

    def test_density_distribution_quantile_and_moments(self) -> None:
        """Uniform formulas agree at interior, exterior, and endpoint values."""
        law = Uniform(-2.0, 4.0)
        self.assertEqual(law.pdf(-3.0), 0.0)
        self.assertAlmostEqual(law.pdf(1.0), 1.0 / 6.0)
        self.assertAlmostEqual(math.exp(law.log_pdf(1.0)), law.pdf(1.0))
        self.assertEqual(law.cdf(-2.0), 0.0)
        self.assertEqual(law.cdf(4.0), 1.0)
        self.assertAlmostEqual(law.cdf(1.0), 0.5)
        self.assertAlmostEqual(law.sf(1.0), 0.5)
        self.assertEqual(law.quantile(0.5), 1.0)
        self.assertEqual(law.mean, 1.0)
        self.assertEqual(law.variance, 3.0)
        self.assertAlmostEqual(law.hazard(1.0), 1.0 / 3.0)
        with self.assertRaises(ValueError):
            law.hazard(4.0)

    def test_parameters_sampling_and_protocols(self) -> None:
        """Invalid intervals fail and continuous interfaces are structural."""
        with self.assertRaises(ValueError):
            Uniform(1.0, 1.0)
        law = Uniform(2.0, 5.0)
        sample = law.sample(5_000, seed=27)
        self.assertTrue(all(2.0 <= value < 5.0 for value in sample))
        self.assertAlmostEqual(statistics.fmean(sample), law.mean, delta=0.05)
        self.assertIsInstance(law, ContinuousProbabilityLaw)
        self.assertIsInstance(law, LogProbabilityDensity)
        self.assertIsInstance(law, SurvivalFunction)
        self.assertIsInstance(law, QuantileFunction)
        self.assertIsInstance(law, HazardFunction)


class ExponentialTests(unittest.TestCase):
    """Verify rate-parameterized exponential waiting times."""

    def test_density_stable_tails_quantiles_and_hazard(self) -> None:
        """Closed forms, inversion, moments, and memoryless hazard agree."""
        law = Exponential(2.0)
        self.assertEqual(law.pdf(-1.0), 0.0)
        self.assertEqual(law.pdf(0.0), 2.0)
        self.assertAlmostEqual(law.cdf(0.5), 1.0 - math.exp(-1.0))
        self.assertAlmostEqual(law.sf(0.5), math.exp(-1.0))
        self.assertAlmostEqual(law.cdf(0.5) + law.sf(0.5), 1.0)
        self.assertAlmostEqual(law.quantile(law.cdf(0.5)), 0.5)
        self.assertEqual(law.quantile(1.0), math.inf)
        self.assertEqual(law.hazard(100.0), 2.0)
        self.assertEqual(law.mean, 0.5)
        self.assertEqual(law.variance, 0.25)

    def test_sampling_and_invalid_rate(self) -> None:
        """Inverse samples reproduce the configured mean and local seed."""
        with self.assertRaises(ValueError):
            Exponential(0.0)
        law = Exponential(1.5)
        sample = law.sample(10_000, seed=46)
        self.assertEqual(sample, law.sample(10_000, seed=46))
        self.assertAlmostEqual(statistics.fmean(sample), law.mean, delta=0.03)


class ExtendedInterfaceTests(unittest.TestCase):
    """Verify quantile and survival additions on earlier distributions."""

    def test_existing_ordered_laws_expose_tail_and_quantile_interfaces(self) -> None:
        """Ordered analytic laws conform without changing categorical semantics."""
        laws = (
            Bernoulli(0.3),
            Binomial(8, 0.4),
            Poisson(3.0),
            Normal(),
            Geometric(0.4),
            NegativeBinomial(2, 0.4),
        )
        for law in laws:
            self.assertIsInstance(law, SurvivalFunction)
            self.assertIsInstance(law, QuantileFunction)
        self.assertAlmostEqual(Normal().sf(8.0), 6.220960574271819e-16)
        self.assertAlmostEqual(Normal().quantile(0.5), 0.0)
        self.assertEqual(Bernoulli(0.3).quantile(0.7), 0.0)
        self.assertEqual(Bernoulli(0.3).quantile(0.7000001), 1.0)
        self.assertEqual(Binomial(8, 0.4).quantile(0.0), 0.0)
        self.assertEqual(Poisson(3.0).quantile(1.0), math.inf)
        self.assertIsInstance(Geometric(0.4), DiscreteProbabilityLaw)


if __name__ == "__main__":
    unittest.main()
