"""Tests for observation, regression, conjugate calibration, and prediction."""

from __future__ import annotations

import math
import unittest

from probstat.inference import (
    condition_linear_gaussian,
    fit_simple_linear_regression,
    posterior_predictive,
    posterior_predictive_check,
)
from probstat.inverse import (
    GaussianObservationModel,
    calibrate_linear_gaussian,
    diagnose_linear_identifiability,
)
from probstat.multivariate import MultivariateNormal
from probstat.validation import AdditiveGaussianDiscrepancy


class InverseCalibrationTests(unittest.TestCase):
    """Verify the complete linear-Gaussian inverse-problem teaching chain."""

    def test_observation_model_separates_noise_and_discrepancy(self) -> None:
        """Likelihood uses total variance while retaining its named components."""
        model = GaussianObservationModel(
            (0.0, 1.0),
            (1.0, 3.0),
            observation_variance=0.25,
            discrepancy_variance=0.75,
        )
        def forward(value: float, parameters: tuple[float, ...]) -> float:
            """Return an intercept-plus-slope response for the test."""
            return parameters[0] + parameters[1] * value

        self.assertEqual(model.predictions(forward, (1.0, 2.0)), (1.0, 3.0))
        self.assertEqual(model.residuals(forward, (1.0, 2.0)), (0.0, 0.0))
        self.assertEqual(model.total_error_variance, 1.0)
        self.assertAlmostEqual(
            model.log_likelihood(forward, (1.0, 2.0)),
            -math.log(2.0 * math.pi),
        )

    def test_ordinary_least_squares_recovers_exact_line(self) -> None:
        """OLS recovers intercept and slope and reports perfect fit diagnostics."""
        result = fit_simple_linear_regression(
            (0.0, 1.0, 2.0, 3.0), (1.0, 3.0, 5.0, 7.0)
        )
        self.assertAlmostEqual(result.intercept, 1.0)
        self.assertAlmostEqual(result.slope, 2.0)
        self.assertAlmostEqual(result.residual_sum_squares, 0.0)
        self.assertAlmostEqual(result.r_squared, 1.0)
        self.assertEqual(result.predict((4.0,)), (9.0,))

    def test_gaussian_conditioning_matches_scalar_closed_form(self) -> None:
        """A one-parameter Gaussian update agrees with precision weighting."""
        posterior = condition_linear_gaussian(
            MultivariateNormal((0.0,), ((4.0,),)),
            ((1.0,), (1.0,)),
            (1.0, 3.0),
            observation_variance=1.0,
        )
        self.assertAlmostEqual(posterior.mean[0], 16.0 / 9.0)
        self.assertAlmostEqual(posterior.covariance[0][0], 4.0 / 9.0)
        self.assertEqual(posterior.sample(3, seed=17), posterior.sample(3, seed=17))

    def test_calibration_reports_identifiability_and_prediction_targets(
        self,
    ) -> None:
        """Calibration evidence and latent/noisy predictive variances stay explicit."""
        prior = MultivariateNormal((0.0, 0.0), ((100.0, 0.0), (0.0, 100.0)))
        result = calibrate_linear_gaussian(
            (0.0, 1.0, 2.0, 3.0),
            (1.1, 2.9, 5.2, 6.8),
            observation_variance=0.04,
            prior=prior,
        )
        self.assertTrue(result.identifiability.identifiable)
        self.assertEqual(result.identifiability.rank, 2)
        design = result.design_rows((1.5, 4.0))
        latent = posterior_predictive(
            result.posterior, design, include_observation_noise=False
        )
        observed = posterior_predictive(
            result.posterior, design, include_observation_noise=True
        )
        self.assertAlmostEqual(
            observed.covariance[0][0] - latent.covariance[0][0], 0.04
        )
        self.assertEqual(result.calibration_inputs, (0.0, 1.0, 2.0, 3.0))

    def test_predictive_check_and_discrepancy_are_reproducible(self) -> None:
        """Predictive replication is seeded and discrepancy scope is auditable."""
        posterior = condition_linear_gaussian(
            MultivariateNormal((0.0,), ((1.0,),)),
            ((1.0,), (1.0,), (1.0,)),
            (0.1, -0.1, 0.0),
            observation_variance=0.25,
        )
        discrepancy = AdditiveGaussianDiscrepancy(
            0.2,
            "Unresolved apparatus bias around the calibrated regime.",
            "Inputs between zero and one.",
        )
        predictive = posterior_predictive(
            posterior,
            ((1.0,), (1.0,), (1.0,)),
            discrepancy_variance=discrepancy.variance,
        )
        first = posterior_predictive_check(
            predictive, (0.1, -0.1, 0.0), draws=300, seed=2026
        )
        second = posterior_predictive_check(
            predictive, (0.1, -0.1, 0.0), draws=300, seed=2026
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.tail_probability, 0.0)
        self.assertLessEqual(first.tail_probability, 1.0)
        self.assertAlmostEqual(discrepancy.variance, 0.04)

    def test_latent_joint_prediction_can_be_singular(self) -> None:
        """Repeated latent design rows retain a valid degenerate Gaussian law."""
        posterior = condition_linear_gaussian(
            MultivariateNormal((0.0,), ((1.0,),)),
            ((1.0,),),
            (0.0,),
            observation_variance=1.0,
        )
        predictive = posterior_predictive(
            posterior,
            ((1.0,), (1.0,), (1.0,)),
            include_observation_noise=False,
        )
        sample = predictive.sample(1, seed=7)[0]
        self.assertAlmostEqual(sample[0], sample[1])
        self.assertAlmostEqual(sample[1], sample[2])
        with self.assertRaises(ValueError):
            posterior_predictive_check(predictive, (0.0, 0.0, 0.0), draws=10)

    def test_invalid_and_unidentifiable_domains_fail_explicitly(self) -> None:
        """Shape, variance, and rank failures are rejected rather than repaired."""
        diagnostic = diagnose_linear_identifiability((2.0, 2.0, 2.0))
        self.assertFalse(diagnostic.identifiable)
        self.assertTrue(math.isinf(diagnostic.condition_number))
        with self.assertRaises(ValueError):
            condition_linear_gaussian(
                MultivariateNormal((0.0,), ((1.0,),)),
                ((1.0,),),
                (1.0,),
                observation_variance=0.0,
            )
        with self.assertRaises(ValueError):
            fit_simple_linear_regression((1.0, 1.0, 1.0), (1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
