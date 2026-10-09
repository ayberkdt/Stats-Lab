"""Tests for ATHENS UQ-1 distributions, multivariate laws, and inference."""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.distributions import LogNormal, Normal, Uniform, Weibull
from probstat.foundations import (
    ContinuousProbabilityLaw,
    HazardFunction,
    LogProbabilityDensity,
    QuantileFunction,
    SurvivalFunction,
)
from probstat.inference import (
    anderson_darling_test,
    fit_lognormal_mle,
    fit_normal_mle,
    kolmogorov_smirnov_test,
    normal_mean_confidence_interval,
    wilson_score_interval,
)
from probstat.multivariate import MultivariateNormal


class LogNormalTests(unittest.TestCase):
    """Verify log-space parameterization, stable tails, and sampling."""

    def test_density_tails_quantile_and_moments(self) -> None:
        """Closed forms agree with the transformed standard normal law."""
        law = LogNormal(0.0, 1.0)
        self.assertEqual(law.pdf(0.0), 0.0)
        self.assertAlmostEqual(law.pdf(1.0), 1.0 / math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(law.cdf(1.0), 0.5)
        self.assertAlmostEqual(law.sf(1.0), 0.5)
        self.assertAlmostEqual(law.cdf(2.0) + law.sf(2.0), 1.0)
        self.assertAlmostEqual(law.quantile(law.cdf(2.0)), 2.0)
        self.assertEqual(law.quantile(0.0), 0.0)
        self.assertEqual(law.quantile(1.0), math.inf)
        self.assertAlmostEqual(law.mean, math.exp(0.5))
        self.assertAlmostEqual(law.variance, math.e * (math.e - 1.0))

    def test_sampling_validation_and_protocols(self) -> None:
        """Seeded samples reproduce log moments and structural interfaces."""
        with self.assertRaises(ValueError):
            LogNormal(0.0, 0.0)
        law = LogNormal(0.4, 0.3)
        sample = law.sample(10_000, seed=19)
        self.assertEqual(sample, law.sample(10_000, seed=19))
        logarithms = tuple(math.log(value) for value in sample)
        self.assertAlmostEqual(statistics.fmean(logarithms), 0.4, delta=0.01)
        self.assertAlmostEqual(statistics.pstdev(logarithms), 0.3, delta=0.01)
        self.assertIsInstance(law, ContinuousProbabilityLaw)
        self.assertIsInstance(law, LogProbabilityDensity)
        self.assertIsInstance(law, SurvivalFunction)
        self.assertIsInstance(law, QuantileFunction)


class WeibullTests(unittest.TestCase):
    """Verify Weibull shape behavior, inversion, hazard, and sampling."""

    def test_closed_forms_and_exponential_reduction(self) -> None:
        """Shape one reduces exactly to an exponential distribution."""
        law = Weibull(1.0, 2.0)
        self.assertEqual(law.pdf(0.0), 0.5)
        self.assertAlmostEqual(law.cdf(2.0), 1.0 - math.exp(-1.0))
        self.assertAlmostEqual(law.sf(2.0), math.exp(-1.0))
        self.assertAlmostEqual(law.quantile(law.cdf(2.0)), 2.0)
        self.assertEqual(law.mean, 2.0)
        self.assertEqual(law.variance, 4.0)
        self.assertEqual(law.hazard(10.0), 0.5)
        self.assertEqual(Weibull(0.5).pdf(0.0), math.inf)
        self.assertEqual(Weibull(2.0).pdf(0.0), 0.0)

    def test_sampling_validation_and_protocols(self) -> None:
        """Seeded observations recover moments and expose continuous APIs."""
        with self.assertRaises(ValueError):
            Weibull(0.0)
        law = Weibull(2.0, 3.0)
        sample = law.sample(15_000, seed=71)
        self.assertEqual(sample, law.sample(15_000, seed=71))
        self.assertAlmostEqual(statistics.fmean(sample), law.mean, delta=0.04)
        self.assertAlmostEqual(statistics.pvariance(sample), law.variance, delta=0.1)
        self.assertIsInstance(law, ContinuousProbabilityLaw)
        self.assertIsInstance(law, HazardFunction)
        self.assertIsInstance(law, SurvivalFunction)
        self.assertIsInstance(law, QuantileFunction)


class MultivariateNormalTests(unittest.TestCase):
    """Verify vector density, marginals, dependence, and Cholesky sampling."""

    def test_standard_bivariate_density_and_marginals(self) -> None:
        """Identity covariance factorizes into two standard normals."""
        law = MultivariateNormal((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(law.dimension, 2)
        self.assertAlmostEqual(law.pdf((0.0, 0.0)), 1.0 / (2.0 * math.pi))
        self.assertAlmostEqual(law.mahalanobis_squared((1.0, 2.0)), 5.0)
        self.assertEqual(law.marginal(0), Normal())
        self.assertEqual(law.correlation(0, 1), 0.0)

    def test_correlated_sampling_recovers_covariance(self) -> None:
        """A seeded sample recovers configured first and second moments."""
        law = MultivariateNormal((1.0, -2.0), ((4.0, 1.2), (1.2, 9.0)))
        sample = law.sample(20_000, seed=42)
        first = tuple(value[0] for value in sample)
        second = tuple(value[1] for value in sample)
        self.assertEqual(sample, law.sample(20_000, seed=42))
        self.assertAlmostEqual(statistics.fmean(first), 1.0, delta=0.04)
        self.assertAlmostEqual(statistics.fmean(second), -2.0, delta=0.06)
        self.assertAlmostEqual(statistics.covariance(first, second), 1.2, delta=0.12)
        self.assertAlmostEqual(law.correlation(0, 1), 0.2)

    def test_invalid_covariance_and_dimensions_are_rejected(self) -> None:
        """Non-square, asymmetric, singular, and mismatched inputs fail."""
        with self.assertRaises(ValueError):
            MultivariateNormal((0.0, 0.0), ((1.0,), (0.0,)))
        with self.assertRaises(ValueError):
            MultivariateNormal((0.0, 0.0), ((1.0, 0.2), (0.1, 1.0)))
        with self.assertRaises(ValueError):
            MultivariateNormal((0.0, 0.0), ((1.0, 1.0), (1.0, 1.0)))
        law = MultivariateNormal((0.0,), ((1.0,),))
        with self.assertRaises(ValueError):
            law.pdf((0.0, 1.0))


class EstimationAndGoodnessOfFitTests(unittest.TestCase):
    """Verify MLE estimators and empirical-distribution diagnostics."""

    def test_normal_and_lognormal_mle(self) -> None:
        """Closed-form MLEs use the likelihood population divisor."""
        fitted = fit_normal_mle((1.0, 2.0, 3.0))
        self.assertEqual(fitted.mu, 2.0)
        self.assertAlmostEqual(fitted.sigma, math.sqrt(2.0 / 3.0))
        log_fitted = fit_lognormal_mle((1.0, math.e, math.e**2))
        self.assertEqual(log_fitted.mu, 1.0)
        self.assertAlmostEqual(log_fitted.sigma, math.sqrt(2.0 / 3.0))
        with self.assertRaises(ValueError):
            fit_normal_mle((2.0, 2.0))
        with self.assertRaises(ValueError):
            fit_lognormal_mle((0.0, 1.0))

    def test_ks_statistic_and_asymptotic_probability(self) -> None:
        """Uniform quartile points have the hand-computed KS distance."""
        result = kolmogorov_smirnov_test(
            (0.25, 0.5, 0.75),
            Uniform(0.0, 1.0),
        )
        self.assertAlmostEqual(result.statistic, 0.25)
        self.assertGreater(result.p_value, 0.9)
        self.assertLessEqual(result.p_value, 1.0)
        self.assertEqual(result.sample_size, 3)

    def test_anderson_darling_uses_tail_probabilities(self) -> None:
        """Interior observations yield finite discrepancy; null tails diverge."""
        result = anderson_darling_test(
            (0.25, 0.5, 0.75),
            Uniform(0.0, 1.0),
        )
        self.assertGreaterEqual(result.statistic, 0.0)
        self.assertTrue(math.isfinite(result.statistic))
        boundary = anderson_darling_test((0.0, 0.5, 1.0), Uniform(0.0, 1.0))
        self.assertEqual(boundary.statistic, math.inf)


class ConfidenceIntervalTests(unittest.TestCase):
    """Verify explicit z and Wilson confidence-interval constructions."""

    def test_known_scale_normal_mean_interval(self) -> None:
        """Known-scale z interval matches its analytic center and width."""
        interval = normal_mean_confidence_interval(
            (1.0, 2.0, 3.0, 4.0),
            confidence=0.95,
            known_sigma=1.0,
        )
        critical = statistics.NormalDist().inv_cdf(0.975)
        self.assertAlmostEqual(interval.lower, 2.5 - critical / 2.0)
        self.assertAlmostEqual(interval.upper, 2.5 + critical / 2.0)
        self.assertIn("known", interval.method)

    def test_wilson_interval_and_invalid_counts(self) -> None:
        """Wilson bounds are symmetric at one half and remain probabilistic."""
        interval = wilson_score_interval(5, 10)
        self.assertAlmostEqual(interval.lower, 0.236593, places=5)
        self.assertAlmostEqual(interval.upper, 0.763407, places=5)
        edge = wilson_score_interval(0, 10)
        self.assertEqual(edge.lower, 0.0)
        self.assertLess(edge.upper, 0.3)
        with self.assertRaises(ValueError):
            wilson_score_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
