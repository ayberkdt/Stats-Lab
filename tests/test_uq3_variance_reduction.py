"""Tests for advanced Monte Carlo estimators and budget diagnostics."""

from __future__ import annotations

import math
import random
import unittest

from probstat.monte_carlo import (
    allocate_stratified_budget,
    compare_efficiency,
    estimate_antithetic_uniform,
    estimate_conditional_expectation,
    estimate_control_variate,
    estimate_importance_sampling,
    estimate_mean,
    estimate_stratified_uniform,
)


class VarianceReductionTests(unittest.TestCase):
    """Verify estimator identities, gains, cost counts, and failure modes."""

    def test_antithetic_linear_integrand_is_exact(self) -> None:
        """Pairing ``u`` and ``1-u`` integrates a linear function exactly."""
        result = estimate_antithetic_uniform(lambda value: value, pairs=50, seed=7)
        self.assertEqual(result.estimate, 0.5)
        self.assertEqual(result.sample_variance, 0.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.sample_size, 50)
        self.assertEqual(result.evaluations, 100)

    def test_control_variate_reduces_quadratic_sample_variance(self) -> None:
        """A uniform control reduces variance for a correlated square target."""
        generator = random.Random(19)
        controls = tuple(generator.random() for _ in range(2_000))
        values = tuple(value * value for value in controls)
        result = estimate_control_variate(
            values,
            controls,
            known_control_mean=0.5,
        )
        self.assertAlmostEqual(result.estimate.estimate, 1.0 / 3.0, delta=0.01)
        self.assertGreater(result.coefficient, 0.0)
        self.assertLess(result.adjusted_variance, result.raw_variance)

    def test_fixed_control_coefficient_and_input_validation(self) -> None:
        """A fixed coefficient is honored and mismatched samples are rejected."""
        result = estimate_control_variate(
            (1.0, 2.0, 3.0),
            (0.0, 1.0, 2.0),
            known_control_mean=1.0,
            coefficient=1.0,
        )
        self.assertEqual(result.coefficient, 1.0)
        self.assertEqual(result.estimate.estimate, 2.0)
        with self.assertRaises(ValueError):
            estimate_control_variate((1.0, 2.0), (1.0,), known_control_mean=0.0)
        with self.assertRaises(ValueError):
            estimate_control_variate(
                (1.0, 2.0), (3.0, 3.0), known_control_mean=3.0
            )

    def test_conditional_expectation_preserves_mean_and_reduces_variance(self) -> None:
        """Conditioning an indicator on one uniform variable lowers variance."""
        generator = random.Random(31)
        first = tuple(generator.random() for _ in range(5_000))
        second = tuple(generator.random() for _ in range(5_000))
        raw = estimate_mean(
            float(left + right > 1.0)
            for left, right in zip(first, second, strict=True)
        )
        conditional = estimate_conditional_expectation(first)
        self.assertAlmostEqual(conditional.estimate, 0.5, delta=0.02)
        self.assertLess(conditional.sample_variance, raw.sample_variance)

    def test_stratification_reduces_linear_integrand_standard_error(self) -> None:
        """Equal strata improve a smooth linear unit-interval expectation."""
        stratified = estimate_stratified_uniform(
            lambda value: value,
            (100,) * 10,
            seed=11,
        )
        generator = random.Random(11)
        crude = estimate_mean(generator.random() for _ in range(1_000))
        self.assertAlmostEqual(stratified.estimate, 0.5, delta=0.005)
        self.assertLess(stratified.standard_error, crude.standard_error)
        self.assertEqual(stratified.evaluations, 1_000)
        self.assertEqual(len(stratified.stratum_means), 10)

    def test_stratified_sampling_validates_allocations(self) -> None:
        """A stratified design needs non-empty strata with replication."""
        with self.assertRaises(ValueError):
            estimate_stratified_uniform(lambda value: value, (), seed=1)
        with self.assertRaises(ValueError):
            estimate_stratified_uniform(lambda value: value, (2, 1), seed=1)

    def test_importance_sampling_matches_uniform_second_moment(self) -> None:
        """A beta-shaped proposal estimates the uniform second moment."""
        result = estimate_importance_sampling(
            lambda value: value * value,
            lambda value: 1.0 if 0.0 <= value <= 1.0 else 0.0,
            lambda value: 2.0 * value if 0.0 <= value <= 1.0 else 0.0,
            lambda generator: math.sqrt(generator.random()),
            sample_size=20_000,
            seed=23,
        )
        self.assertAlmostEqual(result.estimate.estimate, 1.0 / 3.0, delta=0.005)
        self.assertGreaterEqual(result.effective_sample_size, 1.0)
        self.assertLessEqual(result.effective_sample_size, 20_000.0)
        self.assertGreater(result.max_normalized_weight, 0.0)
        self.assertEqual(result.evaluations, 20_000)

    def test_importance_sampling_rejects_zero_proposal_at_draw(self) -> None:
        """A proposal must dominate the target at every realized point."""
        with self.assertRaises(ValueError):
            estimate_importance_sampling(
                lambda value: value,
                lambda value: 1.0,
                lambda value: 0.0,
                lambda generator: 0.5,
                sample_size=2,
            )

    def test_neyman_allocation_is_exact_and_deterministic(self) -> None:
        """Largest remainders preserve the budget and relative pilot scales."""
        allocation = allocate_stratified_budget((1.0, 2.0, 3.0), 60)
        self.assertEqual(allocation, (11, 20, 29))
        self.assertEqual(sum(allocation), 60)
        self.assertEqual(allocate_stratified_budget((0.0, 0.0, 0.0), 10), (4, 3, 3))
        with self.assertRaises(ValueError):
            allocate_stratified_budget((1.0, 2.0, 3.0), 5)

    def test_efficiency_comparison_accounts_for_evaluation_cost(self) -> None:
        """The variance-cost ratio favors a cheaper, more precise candidate."""
        generator = random.Random(5)
        crude = estimate_mean(generator.random() for _ in range(1_000))
        stratified = estimate_stratified_uniform(
            lambda value: value,
            (100,) * 10,
            seed=5,
        )
        comparison = compare_efficiency(
            crude,
            stratified,
            baseline_label="crude",
            candidate_label="stratified",
        )
        self.assertEqual(comparison.baseline_label, "crude")
        self.assertGreater(comparison.efficiency_gain, 1.0)


if __name__ == "__main__":
    unittest.main()
