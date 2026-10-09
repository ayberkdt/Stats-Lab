"""Tests for quadrature, Monte Carlo estimators, and forward propagation."""

from __future__ import annotations

import math
import unittest
from collections.abc import Mapping

from probstat.monte_carlo import (
    convergence_trace,
    estimate_mean,
    estimate_probability,
    estimate_uniform_integral,
)
from probstat.numerics import composite_simpson, composite_trapezoid
from probstat.propagation import expectation_by_quadrature, propagate_samples
from probstat.uq import QuantityOfInterest


class QuadratureTests(unittest.TestCase):
    """Verify deterministic rules, work counts, and numerical boundaries."""

    def test_trapezoid_is_exact_for_linear_integrands(self) -> None:
        """Composite trapezoids integrate an affine function exactly."""
        result = composite_trapezoid(
            lambda x: 2.0 * x + 1.0,
            0.0,
            3.0,
            intervals=12,
        )
        self.assertAlmostEqual(result.value, 12.0)
        self.assertEqual(result.evaluations, 13)
        self.assertEqual(result.intervals, 12)

    def test_simpson_is_exact_for_cubic_integrands(self) -> None:
        """Composite Simpson quadrature integrates a cubic exactly."""
        result = composite_simpson(lambda x: x**3, 0.0, 2.0, intervals=10)
        self.assertAlmostEqual(result.value, 4.0)
        self.assertEqual(result.evaluations, 11)

    def test_quadrature_rejects_invalid_domains_and_values(self) -> None:
        """Reversed bounds, odd Simpson grids, and infinite values fail."""
        with self.assertRaises(ValueError):
            composite_trapezoid(lambda x: x, 1.0, 0.0, intervals=2)
        with self.assertRaises(ValueError):
            composite_simpson(lambda x: x, 0.0, 1.0, intervals=3)
        with self.assertRaises(ValueError):
            composite_simpson(lambda _: math.inf, 0.0, 1.0, intervals=2)

    def test_expectation_integral_combines_transform_and_density(self) -> None:
        """Uniform expectation of the square equals one third."""
        result = expectation_by_quadrature(
            lambda x: x**2,
            lambda _: 1.0,
            0.0,
            1.0,
            intervals=20,
        )
        self.assertAlmostEqual(result.value, 1.0 / 3.0)
        with self.assertRaises(ValueError):
            expectation_by_quadrature(
                lambda x: x,
                lambda _: -1.0,
                0.0,
                1.0,
                intervals=2,
            )


class MonteCarloEstimatorTests(unittest.TestCase):
    """Verify estimator equations, confidence metadata, and convergence traces."""

    def test_mean_variance_standard_error_and_degenerate_sample(self) -> None:
        """Sample statistics and the constant-output boundary are explicit."""
        estimate = estimate_mean((1.0, 2.0, 3.0, 4.0))
        self.assertEqual(estimate.estimate, 2.5)
        self.assertAlmostEqual(estimate.sample_variance, 5.0 / 3.0)
        self.assertAlmostEqual(estimate.standard_error, math.sqrt(5.0 / 12.0))
        self.assertEqual(estimate.sample_size, 4)
        self.assertEqual(estimate.evaluations, 4)
        constant = estimate_mean((2.0, 2.0, 2.0))
        self.assertEqual(constant.standard_error, 0.0)
        self.assertEqual(constant.confidence_interval.lower, 2.0)

    def test_probability_estimator_and_wilson_interval(self) -> None:
        """Empirical event probability retains counts and interval uncertainty."""
        result = estimate_probability((True, False, True, True))
        self.assertEqual(result.probability, 0.75)
        self.assertEqual(result.successes, 3)
        self.assertEqual(result.trials, 4)
        self.assertLess(result.confidence_interval.lower, 0.75)
        self.assertGreater(result.confidence_interval.upper, 0.75)
        with self.assertRaises(TypeError):
            estimate_probability((True, 1))  # type: ignore[arg-type]

    def test_uniform_monte_carlo_integral_is_seeded_and_costed(self) -> None:
        """A seeded square integral agrees with one third within sampling error."""
        result = estimate_uniform_integral(
            lambda x: x**2,
            0.0,
            1.0,
            sample_size=20_000,
            seed=2026,
        )
        repeated = estimate_uniform_integral(
            lambda x: x**2,
            0.0,
            1.0,
            sample_size=20_000,
            seed=2026,
        )
        self.assertEqual(result, repeated)
        self.assertLess(abs(result.estimate - 1.0 / 3.0), 4.0 * result.standard_error)
        self.assertEqual(result.evaluations, 20_000)

    def test_nested_convergence_trace_has_root_n_standard_error(self) -> None:
        """Repeated Bernoulli contributions show decreasing standard error."""
        values = (0.0, 1.0) * 500
        trace = convergence_trace(values, (10, 100, 1000))
        self.assertEqual(tuple(item.sample_size for item in trace), (10, 100, 1000))
        self.assertTrue(all(item.estimate == 0.5 for item in trace))
        self.assertGreater(trace[0].standard_error, trace[1].standard_error)
        self.assertGreater(trace[1].standard_error, trace[2].standard_error)


class ForwardPropagationTests(unittest.TestCase):
    """Verify named QoI propagation, summaries, thresholds, and cost."""

    def test_samples_produce_output_moments_and_threshold_probability(self) -> None:
        """A quadratic model yields hand-computed moments and exceedance rate."""

        def square_model(parameters: Mapping[str, float]) -> Mapping[str, float]:
            return {"energy": parameters["x"] ** 2}

        qoi = QuantityOfInterest("energy", "Squared scalar response", "J")
        samples = tuple({"x": float(value)} for value in (-2, -1, 0, 1, 2))
        result = propagate_samples(
            square_model,
            samples,
            (qoi,),
            thresholds={"energy": 1.0},
        )
        self.assertEqual(result.outputs["energy"], (4.0, 1.0, 0.0, 1.0, 4.0))
        summary = result.summaries["energy"]
        self.assertEqual(summary.mean, 2.0)
        self.assertEqual(summary.variance, 3.5)
        self.assertAlmostEqual(summary.standard_deviation, math.sqrt(3.5))
        self.assertIsNotNone(summary.exceedance)
        assert summary.exceedance is not None
        self.assertEqual(summary.exceedance.probability, 0.4)
        self.assertEqual(result.model_evaluations, 5)
        with self.assertRaises(TypeError):
            result.outputs["energy"] = ()  # type: ignore[index]

    def test_propagation_rejects_invalid_sample_and_threshold_inputs(self) -> None:
        """Variance needs two runs and thresholds must name requested outputs."""
        qoi = QuantityOfInterest("y", "Response")

        def identity(parameters: Mapping[str, float]) -> Mapping[str, float]:
            return {"y": parameters["x"]}

        with self.assertRaises(ValueError):
            propagate_samples(identity, ({"x": 1.0},), (qoi,))
        with self.assertRaises(ValueError):
            propagate_samples(
                identity,
                ({"x": 1.0}, {"x": 2.0}),
                (qoi,),
                thresholds={"other": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
