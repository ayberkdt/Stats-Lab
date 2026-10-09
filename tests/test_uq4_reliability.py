"""Tests for limit-state, rare-event, FORM, and directional reliability APIs."""

from __future__ import annotations

import math
import random
import unittest
from statistics import NormalDist

from probstat.reliability import (
    LimitState,
    estimate_directional_failure_probability,
    estimate_failure_probability,
    estimate_weighted_failure_probability,
    linear_normal_form,
)


class ReliabilityTests(unittest.TestCase):
    """Check reliability definitions, estimators, geometry, and diagnostics."""

    def test_limit_state_uses_closed_failure_domain(self) -> None:
        """Zero and negative margins fail while positive margins remain safe."""
        state = LimitState(lambda vector: 2.0 - vector[0], "capacity", 1)
        self.assertFalse(state.failed((1.0,)))
        self.assertTrue(state.failed((2.0,)))
        self.assertTrue(state.failed((3.0,)))
        with self.assertRaises(ValueError):
            state.evaluate((1.0, 2.0))

    def test_direct_monte_carlo_reports_failure_reliability_and_wilson_interval(
        self,
    ) -> None:
        """Direct sampling retains event counts, complement, uncertainty, and cost."""
        state = LimitState(lambda vector: 3.0 - vector[0] - vector[1], "sum", 2)
        generator = random.Random(7)
        samples = tuple(
            (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
            for _ in range(40_000)
        )
        result = estimate_failure_probability(state, samples)
        reference = NormalDist().cdf(-3.0 / math.sqrt(2.0))
        self.assertAlmostEqual(result.failure.probability, reference, delta=0.002)
        self.assertAlmostEqual(result.reliability, 1.0 - result.failure.probability)
        self.assertEqual(result.evaluations, 40_000)
        self.assertLessEqual(
            result.failure.confidence_interval.lower,
            result.failure.probability,
        )

    def test_zero_failures_still_have_nonzero_wilson_upper_bound(self) -> None:
        """No observed failures is not treated as proof of zero risk."""
        state = LimitState(lambda vector: 10.0 - vector[0], "rare", 1)
        result = estimate_failure_probability(state, ((0.0,),) * 100)
        self.assertEqual(result.failure.probability, 0.0)
        self.assertGreater(result.failure.confidence_interval.upper, 0.0)

    def test_linear_normal_form_matches_exact_affine_probability(self) -> None:
        """Affine standard-normal FORM recovers beta, design point, and exact tail."""
        result = linear_normal_form(3.0, (-1.0, -1.0))
        expected_beta = 3.0 / math.sqrt(2.0)
        self.assertAlmostEqual(result.reliability_index, expected_beta)
        self.assertAlmostEqual(
            result.failure_probability,
            NormalDist().cdf(-expected_beta),
        )
        self.assertAlmostEqual(result.boundary_residual, 0.0, places=14)
        self.assertAlmostEqual(result.design_point[0], 1.5)
        self.assertAlmostEqual(result.design_point[1], 1.5)

    def test_form_signed_index_handles_origin_inside_failure_domain(self) -> None:
        """A negative signed beta correctly represents failure above one half."""
        result = linear_normal_form(-1.0, (1.0,))
        self.assertEqual(result.reliability_index, -1.0)
        self.assertAlmostEqual(result.failure_probability, NormalDist().cdf(1.0))
        with self.assertRaises(ValueError):
            linear_normal_form(1.0, (0.0, 0.0))

    def test_directional_sampling_matches_linear_normal_form(self) -> None:
        """Radial chi-tail averaging agrees with an affine bivariate benchmark."""
        def radial_boundary(direction: tuple[float, ...]) -> float | None:
            projection = direction[0] + direction[1]
            return None if projection <= 0.0 else 3.0 / projection

        result = estimate_directional_failure_probability(
            radial_boundary,
            dimension=2,
            directions=30_000,
            seed=13,
        )
        reference = linear_normal_form(3.0, (-1.0, -1.0))
        self.assertAlmostEqual(
            result.failure_probability,
            reference.failure_probability,
            delta=0.001,
        )
        self.assertEqual(result.evaluations, 30_000)
        self.assertGreater(result.finite_boundaries, 0)

    def test_importance_weighting_estimates_a_rare_normal_tail(self) -> None:
        """Mean-shift importance sampling resolves a five-sigma failure tail."""
        state = LimitState(lambda vector: 5.0 - vector[0], "five sigma", 1)
        generator = random.Random(29)
        samples = tuple((generator.gauss(3.0, 1.0),) for _ in range(60_000))
        weights = tuple(
            math.exp(-0.5 * value[0] ** 2 + 0.5 * (value[0] - 3.0) ** 2)
            for value in samples
        )
        result = estimate_weighted_failure_probability(state, samples, weights)
        reference = NormalDist().cdf(-5.0)
        self.assertAlmostEqual(
            result.estimate.estimate / reference,
            1.0,
            delta=0.12,
        )
        self.assertGreater(result.observed_failures, 500)
        self.assertGreaterEqual(result.weight_effective_sample_size, 1.0)
        self.assertGreater(result.max_normalized_weight, 0.0)

    def test_reliability_estimators_reject_invalid_inputs(self) -> None:
        """Empty samples, mismatched weights, and negative radii fail clearly."""
        state = LimitState(lambda vector: vector[0], "boundary", 1)
        with self.assertRaises(ValueError):
            estimate_failure_probability(state, ())
        with self.assertRaises(ValueError):
            estimate_weighted_failure_probability(state, ((1.0,), (2.0,)), (1.0,))
        with self.assertRaises(ValueError):
            estimate_directional_failure_probability(
                lambda direction: -1.0,
                dimension=2,
                directions=2,
            )


if __name__ == "__main__":
    unittest.main()
