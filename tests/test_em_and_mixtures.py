"""Tests for the generic EM driver, Gaussian mixtures, and censored-normal EM."""

from __future__ import annotations

import math
import random
import unittest
import warnings

from probstat.inference import (
    EMMonotonicityWarning,
    GaussianMixture,
    censored_normal_log_likelihood,
    expectation_maximization,
    fit_censored_normal,
    fit_gaussian_mixture,
    kmeans_plus_plus,
    select_gaussian_mixture,
)
from probstat.optimization import minimize_nelder_mead

TRUE_MIXTURE = GaussianMixture(
    (0.3, 0.5, 0.2),
    ((-4.0, 0.0), (0.0, 3.0), (4.0, -1.0)),
    (
        ((1.0, 0.3), (0.3, 0.5)),
        ((0.6, 0.0), (0.0, 1.5)),
        ((0.8, -0.4), (-0.4, 1.0)),
    ),
)


def assert_monotone(case: unittest.TestCase, trace: tuple[float, ...]) -> None:
    """Assert that an objective trace never decreases beyond round-off."""
    for old, new in zip(trace, trace[1:], strict=False):
        case.assertGreaterEqual(new, old - 1e-9 * (1.0 + abs(old)))


class GenericEMTests(unittest.TestCase):
    """The driver iterates, converges, and detects a broken M-step."""

    @staticmethod
    def _normal_mean_em(
        broken: bool,
    ) -> tuple[object, object]:
        """Return E/M steps for a normal mean with half of the data missing."""
        observed = (1.0, 2.0, 4.0, 5.0)
        missing = 4

        def e_step(mu: float) -> tuple[float, float]:
            """Fill missing values with the current mean; return log-lik."""
            loglik = -0.5 * math.fsum((v - mu) ** 2 for v in observed)
            return math.fsum(observed) + missing * mu, loglik

        def m_step(total: float) -> float:
            """Return the completed-data mean (or a deliberately bad value)."""
            mean = total / (len(observed) + missing)
            return mean + (5.0 if broken else 0.0)

        return e_step, m_step

    def test_converges_to_observed_mean_monotonically(self) -> None:
        """Missing-at-random EM converges to the observed-data MLE."""
        e_step, m_step = self._normal_mean_em(broken=False)
        result = expectation_maximization(0.0, e_step, m_step, tolerance=1e-14)
        self.assertTrue(result.converged)
        self.assertTrue(result.monotone)
        self.assertAlmostEqual(result.parameters, 3.0, places=5)
        assert_monotone(self, result.trace)
        self.assertEqual(len(result.trace), result.iterations + 1)

    def test_decrease_is_reported(self) -> None:
        """A wrong M-step triggers the warning, the error, or a record."""
        e_step, m_step = self._normal_mean_em(broken=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = expectation_maximization(
                3.0, e_step, m_step, max_iterations=5
            )
        self.assertFalse(result.monotone)
        self.assertGreater(result.max_decrease, 0.0)
        self.assertTrue(
            any(issubclass(w.category, EMMonotonicityWarning) for w in caught)
        )
        with self.assertRaises(ArithmeticError):
            expectation_maximization(
                3.0, e_step, m_step, max_iterations=5, on_decrease="raise"
            )
        quiet = expectation_maximization(
            3.0, e_step, m_step, max_iterations=5, on_decrease="ignore"
        )
        self.assertFalse(quiet.monotone)

    def test_rejects_bad_settings(self) -> None:
        """Invalid tolerances and policies are rejected."""
        e_step, m_step = self._normal_mean_em(broken=False)
        with self.assertRaises(ValueError):
            expectation_maximization(0.0, e_step, m_step, tolerance=-1.0)
        with self.assertRaises(ValueError):
            expectation_maximization(0.0, e_step, m_step, max_iterations=0)
        with self.assertRaises(ValueError):
            expectation_maximization(
                0.0, e_step, m_step, on_decrease="loud"  # type: ignore[arg-type]
            )


class GaussianMixtureTests(unittest.TestCase):
    """Mixture fitting, monotonicity, recovery, ordering, and selection."""

    @classmethod
    def setUpClass(cls) -> None:
        """Draw one seeded bivariate dataset shared by the tests."""
        cls.data = TRUE_MIXTURE.sample(600, seed=11)

    def test_log_likelihood_never_decreases(self) -> None:
        """The penalized EM objective is monotone for full and diagonal fits."""
        for covariance_type in ("full", "diag"):
            fit = fit_gaussian_mixture(
                self.data, 3, covariance_type=covariance_type, seed=3,
                tolerance=1e-12,
            )
            self.assertTrue(fit.monotone)
            self.assertGreater(len(fit.trace), 3)
            assert_monotone(self, fit.trace)
        univariate = [p[0] for p in self.data]
        fit = fit_gaussian_mixture(univariate, 4, seed=2, tolerance=1e-12)
        assert_monotone(self, fit.trace)

    def test_recovers_seeded_parameters(self) -> None:
        """Weights, means, and covariances are recovered within tolerance."""
        fit = fit_gaussian_mixture(self.data, 3, seed=3)
        mixture = fit.mixture
        self.assertTrue(fit.converged)
        for estimate, truth in zip(mixture.weights, TRUE_MIXTURE.weights, strict=True):
            self.assertAlmostEqual(estimate, truth, delta=0.05)
        for estimate, truth in zip(mixture.means, TRUE_MIXTURE.means, strict=True):
            for a, b in zip(estimate, truth, strict=True):
                self.assertAlmostEqual(a, b, delta=0.15)
        for estimate, truth in zip(
            mixture.covariances, TRUE_MIXTURE.covariances, strict=True
        ):
            for row_a, row_b in zip(estimate, truth, strict=True):
                for a, b in zip(row_a, row_b, strict=True):
                    self.assertAlmostEqual(a, b, delta=0.25)
        for row in fit.responsibilities:
            self.assertAlmostEqual(math.fsum(row), 1.0, places=12)
        self.assertEqual(fit.parameter_count, 2 + 6 + 9)
        self.assertAlmostEqual(
            fit.bic, fit.parameter_count * math.log(600) - 2.0 * fit.log_likelihood
        )

    def test_canonical_order_removes_label_switching(self) -> None:
        """Different seeds give the same component order and estimates."""
        first = fit_gaussian_mixture(self.data, 3, seed=1, restarts=2).mixture
        second = fit_gaussian_mixture(self.data, 3, seed=99).mixture
        means = [m[0] for m in first.means]
        self.assertEqual(means, sorted(means))
        for a, b in zip(first.means, second.means, strict=True):
            for x, y in zip(a, b, strict=True):
                self.assertAlmostEqual(x, y, places=4)

    def test_bic_selects_true_component_count(self) -> None:
        """BIC chooses K = 3 on well-separated three-component data."""
        selection = select_gaussian_mixture(
            self.data[:400], (1, 2, 3, 4, 5), seed=5, tolerance=1e-7,
            max_iterations=150,
        )
        self.assertEqual(selection.best_components, 3)
        self.assertEqual(selection.criterion, "bic")
        self.assertEqual(min(selection.scores), selection.best.bic)

    def test_univariate_mixture_and_log_pdf(self) -> None:
        """A univariate fit recovers its components and log_pdf is exact."""
        truth = GaussianMixture((0.4, 0.6), ((-2.0,), (3.0,)), (((0.5,),), ((1.0,),)))
        sample = [p[0] for p in truth.sample(800, seed=4)]
        fit = fit_gaussian_mixture(sample, 2, seed=1)
        self.assertAlmostEqual(fit.mixture.means[0][0], -2.0, delta=0.15)
        self.assertAlmostEqual(fit.mixture.means[1][0], 3.0, delta=0.15)
        self.assertAlmostEqual(fit.mixture.weights[0], 0.4, delta=0.05)
        x = 0.7
        manual = math.log(
            0.4 * math.exp(-((x + 2.0) ** 2) / 1.0) / math.sqrt(math.pi)
            + 0.6 * math.exp(-((x - 3.0) ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
        )
        self.assertAlmostEqual(truth.log_pdf(x), manual, places=12)
        self.assertEqual(truth.predict([-2.0, 3.0]), (0, 1))

    def test_regularization_floor_handles_duplicate_points(self) -> None:
        """Repeated values cannot collapse a covariance to zero."""
        rng = random.Random(8)
        data = [5.0] * 30 + [rng.gauss(0.0, 1.0) for _ in range(60)]
        fit = fit_gaussian_mixture(data, 2, seed=0, regularization=1e-4)
        self.assertTrue(math.isfinite(fit.log_likelihood))
        self.assertTrue(all(c[0][0] > 0.0 for c in fit.mixture.covariances))
        self.assertTrue(fit.monotone)

    def test_kmeans_plus_plus_seeding(self) -> None:
        """Seeding is reproducible and returns distinct data points."""
        centers = kmeans_plus_plus(self.data, 3, seed=21)
        self.assertEqual(centers, kmeans_plus_plus(self.data, 3, seed=21))
        self.assertEqual(len(set(centers)), 3)
        self.assertTrue(all(center in self.data for center in centers))
        with self.assertRaises(ValueError):
            kmeans_plus_plus([1.0, 2.0], 3)

    def test_invalid_mixture_is_rejected(self) -> None:
        """Weights must sum to one and covariances must be positive definite."""
        with self.assertRaises(ValueError):
            GaussianMixture((0.5, 0.6), ((0.0,), (1.0,)), (((1.0,),), ((1.0,),)))
        with self.assertRaises(ValueError):
            GaussianMixture((1.0,), ((0.0, 0.0),), (((1.0, 2.0), (2.0, 1.0)),))


class CensoredNormalTests(unittest.TestCase):
    """EM for left/right-censored normal data matches direct maximization."""

    @staticmethod
    def _censored_sample() -> list[tuple[float, float]]:
        """Return 300 normal values censored below 8 and above 13."""
        rng = random.Random(5)
        intervals: list[tuple[float, float]] = []
        for _ in range(300):
            value = rng.gauss(10.0, 2.0)
            if value < 8.0:
                intervals.append((-math.inf, 8.0))
            elif value > 13.0:
                intervals.append((13.0, math.inf))
            else:
                intervals.append((value, value))
        return intervals

    def test_matches_direct_maximum_likelihood(self) -> None:
        """EM and Nelder-Mead on the observed likelihood agree."""
        intervals = self._censored_sample()
        fit = fit_censored_normal(intervals)
        self.assertTrue(fit.em.converged)
        self.assertTrue(fit.em.monotone)
        assert_monotone(self, fit.em.trace)
        self.assertGreater(fit.censored_count, 50)

        def objective(x: tuple[float, ...]) -> float:
            """Negative observed log-likelihood on (mu, log sigma)."""
            return -censored_normal_log_likelihood(intervals, x[0], math.exp(x[1]))

        direct = minimize_nelder_mead(objective, (9.0, 0.5), tolerance=1e-14)
        self.assertAlmostEqual(fit.mean, direct.x[0], delta=1e-5)
        self.assertAlmostEqual(
            fit.standard_deviation, math.exp(direct.x[1]), delta=1e-5
        )
        self.assertAlmostEqual(fit.log_likelihood, -direct.value, delta=1e-7)
        self.assertAlmostEqual(fit.mean, 10.0, delta=0.4)
        self.assertAlmostEqual(fit.standard_deviation, 2.0, delta=0.4)

    def test_uncensored_data_give_the_normal_mle(self) -> None:
        """Without censoring EM returns the sample mean and MLE deviation."""
        values = (1.0, 2.5, 3.0, 4.5, 7.0)
        fit = fit_censored_normal([(v, v) for v in values])
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        self.assertAlmostEqual(fit.mean, mean, places=12)
        self.assertAlmostEqual(fit.standard_deviation, sd, places=12)

    def test_interval_censoring_and_validation(self) -> None:
        """Interval-censored values are allowed; malformed ones are rejected."""
        data = [(1.0, 1.0), (2.0, 2.0), (0.0, 3.0), (2.5, 2.5), (1.5, 4.0)]
        fit = fit_censored_normal(data)
        self.assertTrue(fit.em.monotone)
        self.assertEqual(fit.censored_count, 2)
        with self.assertRaises(ValueError):
            fit_censored_normal([(2.0, 1.0)])
        with self.assertRaises(ValueError):
            fit_censored_normal([(-math.inf, math.inf), (1.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
