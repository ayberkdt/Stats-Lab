"""Tests for local derivatives and variance-based global sensitivity."""

from __future__ import annotations

import random
import unittest

from probstat.sensitivity import (
    central_difference_gradient,
    estimate_sobol_indices,
)


class SensitivityTests(unittest.TestCase):
    """Verify derivative, ANOVA, interaction, ranking, and decision contracts."""

    def test_central_difference_recovers_quadratic_gradient_and_cost(self) -> None:
        """A central stencil differentiates a quadratic at a known point."""
        result = central_difference_gradient(
            lambda vector: vector[0] ** 2 + 3.0 * vector[1],
            (2.0, -1.0),
            (1e-4, 1e-4),
        )
        self.assertAlmostEqual(result.gradient[0], 4.0, places=8)
        self.assertAlmostEqual(result.gradient[1], 3.0, places=8)
        self.assertEqual(result.evaluations, 4)

    def test_additive_model_sobol_indices_match_variance_shares(self) -> None:
        """Independent additive uniforms allocate variance in a one-to-four ratio."""
        sample_a, sample_b = self._uniform_matrices(30_000, 2)
        result = estimate_sobol_indices(
            lambda vector: vector[0] + 2.0 * vector[1],
            sample_a,
            sample_b,
        )
        self.assertAlmostEqual(result.first_order[0], 0.2, delta=0.025)
        self.assertAlmostEqual(result.first_order[1], 0.8, delta=0.025)
        self.assertAlmostEqual(result.total_effect[0], 0.2, delta=0.02)
        self.assertAlmostEqual(result.total_effect[1], 0.8, delta=0.02)
        self.assertAlmostEqual(result.second_order[0][1], 0.0, delta=0.035)
        self.assertEqual(result.ranking(), (1, 0))
        self.assertEqual(result.evaluations, 150_000)

    def test_pure_interaction_has_zero_main_and_unit_pair_effect(self) -> None:
        """A centered product is explained by interaction rather than main effects."""
        sample_a, sample_b = self._uniform_matrices(40_000, 2, seed=41)
        result = estimate_sobol_indices(
            lambda vector: vector[0] * vector[1],
            sample_a,
            sample_b,
        )
        self.assertAlmostEqual(result.first_order[0], 0.0, delta=0.035)
        self.assertAlmostEqual(result.first_order[1], 0.0, delta=0.035)
        self.assertAlmostEqual(result.total_effect[0], 1.0, delta=0.04)
        self.assertAlmostEqual(result.total_effect[1], 1.0, delta=0.04)
        self.assertAlmostEqual(result.second_order[0][1], 1.0, delta=0.06)

    def test_ranking_and_screening_decision_remain_explicit(self) -> None:
        """A nearly inactive third input becomes a fixing candidate by threshold."""
        sample_a, sample_b = self._uniform_matrices(25_000, 3, seed=9)
        result = estimate_sobol_indices(
            lambda vector: vector[0] + 3.0 * vector[1] + 0.01 * vector[2],
            sample_a,
            sample_b,
        )
        decision = result.decision(threshold=0.001)
        self.assertEqual(decision.priority_inputs[:2], (1, 0))
        self.assertEqual(decision.fix_candidates, (2,))

    def test_invalid_domains_fail_without_silent_repair(self) -> None:
        """Bad steps, matrix shapes, and constant responses are rejected."""
        with self.assertRaises(ValueError):
            central_difference_gradient(lambda vector: vector[0], (1.0,), (0.0,))
        with self.assertRaises(ValueError):
            estimate_sobol_indices(lambda vector: sum(vector), ((1.0,),), ((2.0,),))
        with self.assertRaises(ValueError):
            estimate_sobol_indices(
                lambda vector: 1.0,
                ((0.0,), (1.0,)),
                ((2.0,), (3.0,)),
            )

    @staticmethod
    def _uniform_matrices(
        rows: int,
        dimension: int,
        *,
        seed: int = 2026,
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        """Generate two independent seeded uniform matrices on ``[-1,1]^d``."""
        generator = random.Random(seed)
        first = tuple(
            tuple(generator.uniform(-1.0, 1.0) for _ in range(dimension))
            for _ in range(rows)
        )
        second = tuple(
            tuple(generator.uniform(-1.0, 1.0) for _ in range(dimension))
            for _ in range(rows)
        )
        return first, second


if __name__ == "__main__":
    unittest.main()
