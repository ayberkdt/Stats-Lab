"""Tests for stacking, pseudo-BMA, and model-averaged prediction.

Oracles: the closed-form two-model, two-point stacking optimum
``w = -(u d + v b) / (2 u v)`` (stationary point of
``log(w a + (1-w) b) + log(w c + (1-w) d)`` with ``u = a - b``, ``v = c - d``);
the Karush-Kuhn-Tucker conditions of the concave stacking problem checked
independently on the returned weights; a model that dominates on every point
receiving all the weight; the softmax of the elpd for pseudo-BMA; and the
law of total variance for a finite mixture recomputed by hand. A seeded
M-open example (data from a two-component mixture, two single-component
models) shows stacking keeping both models while pseudo-BMA picks one.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.validation import (
    ModelWeights,
    model_averaged_prediction,
    pseudo_bma_weights,
    stacking_weights,
)


def _normal_log_pdf(y: float, mu: float, sigma: float) -> float:
    """Return the normal log density."""
    return -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((y - mu) / sigma) ** 2


class StackingTests(unittest.TestCase):
    """Stacking weights solve the concave log-score problem."""

    def test_two_point_closed_form(self) -> None:
        """Densities (a, b; c, d) = (2, 0.5; 0.5, 1) give w = 5/6."""
        log_p = [[math.log(2.0), math.log(0.5)], [math.log(0.5), math.log(1.0)]]
        result = stacking_weights(log_p, tolerance=1e-13, max_iterations=100000)
        u, v, b, d = 1.5, -0.5, 0.5, 1.0
        expected = -(u * d + v * b) / (2 * u * v)
        self.assertAlmostEqual(expected, 5.0 / 6.0)
        self.assertAlmostEqual(result.weights[0], expected, places=5)
        score = 0.5 * (
            math.log(expected * 2.0 + (1 - expected) * 0.5)
            + math.log(expected * 0.5 + (1 - expected) * 1.0)
        )
        self.assertAlmostEqual(result.log_score, score, places=9)
        self.assertLessEqual(result.optimality_gap, 1e-13)

    def test_kkt_conditions_on_random_problem(self) -> None:
        """At the optimum g_k <= 1 for all k and g_k = 1 where w_k > 0."""
        rng = random.Random(3)
        log_p = [[rng.gauss(0.0, 2.0) for _ in range(4)] for _ in range(40)]
        result = stacking_weights(log_p, tolerance=1e-12, max_iterations=200000)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=12)
        mixtures = [
            sum(w * math.exp(lp) for w, lp in zip(result.weights, row, strict=True))
            for row in log_p
        ]
        for k in range(4):
            g = sum(
                math.exp(row[k]) / m for row, m in zip(log_p, mixtures, strict=True)
            )
            g /= len(log_p)
            self.assertLessEqual(g, 1.0 + 1e-8)
            if result.weights[k] > 1e-4:
                self.assertAlmostEqual(g, 1.0, places=6)

    def test_dominant_model_takes_all_weight(self) -> None:
        """A model with higher density at every point gets weight one."""
        log_p = [[-1.0, -3.0, -2.0], [-0.5, -1.5, -0.9], [-2.0, -2.5, -4.0]]
        result = stacking_weights(log_p, tolerance=1e-10)
        self.assertGreater(result.weights[0], 1.0 - 1e-6)

    def test_zero_density_entries_are_allowed(self) -> None:
        """A model with zero density at a point is down-weighted, not fatal."""
        log_p = [[-math.inf, -1.0], [-0.1, -1.0], [-0.2, -1.2]]
        result = stacking_weights(log_p)
        self.assertTrue(math.isfinite(result.log_score))
        self.assertGreater(result.weights[1], 0.0)

    def test_stacking_keeps_both_models_in_the_m_open_case(self) -> None:
        """Mixture data: stacking near 1/2 each, pseudo-BMA concentrates."""
        rng = random.Random(21)
        data = [
            rng.gauss(-2.0, 1.0) if rng.random() < 0.5 else rng.gauss(2.0, 1.0)
            for _ in range(400)
        ]
        log_p = [
            [_normal_log_pdf(y, -2.0, 1.0), _normal_log_pdf(y, 2.0, 1.0)] for y in data
        ]
        stacked = stacking_weights(log_p)
        bma = pseudo_bma_weights(log_p)
        self.assertAlmostEqual(stacked.weights[0], 0.5, delta=0.08)
        self.assertGreater(max(bma.weights), 0.99)
        self.assertGreater(stacked.log_score, bma.log_score)


class PseudoBmaTests(unittest.TestCase):
    """Pseudo-BMA and pseudo-BMA+ follow their definitions."""

    def test_pseudo_bma_is_softmax_of_elpd(self) -> None:
        """Weights equal exp(elpd_k) / sum exp(elpd_j)."""
        log_p = [[-1.0, -1.2], [-0.8, -0.7], [-1.5, -1.6]]
        result = pseudo_bma_weights(log_p)
        elpd = (-3.3, -3.5)
        self.assertEqual(result.elpd, (-3.3, -3.5))
        expected = math.exp(elpd[0]) / (math.exp(elpd[0]) + math.exp(elpd[1]))
        self.assertAlmostEqual(result.weights[0], expected, places=12)
        self.assertEqual(result.method, "pseudo-bma")

    def test_bayesian_bootstrap_regularizes(self) -> None:
        """Pseudo-BMA+ lies strictly inside (0, 1) where pseudo-BMA saturates."""
        rng = random.Random(5)
        log_p = [[rng.gauss(-1.0, 1.0), rng.gauss(-1.3, 1.0)] for _ in range(60)]
        plain = pseudo_bma_weights(log_p)
        plus = pseudo_bma_weights(log_p, n_bootstrap=500, seed=2)
        self.assertEqual(plus.method, "pseudo-bma+")
        self.assertAlmostEqual(sum(plus.weights), 1.0, places=12)
        self.assertLess(max(plus.weights), max(plain.weights))
        self.assertGreater(min(plus.weights), min(plain.weights))
        again = pseudo_bma_weights(log_p, n_bootstrap=500, seed=2)
        self.assertEqual(plus, again)


class AveragedPredictionTests(unittest.TestCase):
    """The mixture moments follow the law of total variance."""

    def test_variance_decomposition(self) -> None:
        """Mean, within, and between variance equal the hand formulas."""
        result = model_averaged_prediction([0.25, 0.75], [1.0, 3.0], [0.5, 2.0])
        self.assertAlmostEqual(result.mean, 2.5)
        self.assertAlmostEqual(result.within_model_variance, 0.25 * 0.5 + 0.75 * 2.0)
        self.assertAlmostEqual(
            result.between_model_variance, 0.25 * 1.5**2 + 0.75 * 0.5**2
        )
        self.assertAlmostEqual(
            result.total_variance,
            result.within_model_variance + result.between_model_variance,
        )
        self.assertAlmostEqual(result.model_form_share, 0.75 / (1.625 + 0.75))

    def test_mixture_variance_matches_simulation_free_second_moment(self) -> None:
        """Total variance equals E[Y^2] - E[Y]^2 of the mixture."""
        weights = ModelWeights((0.2, 0.3, 0.5), "stacking", 0.0, (0.0,) * 3, 1, 0.0)
        means, variances = [0.0, 2.0, -1.0], [1.0, 0.25, 4.0]
        result = model_averaged_prediction(weights, means, variances)
        second = sum(
            w * (v + m**2)
            for w, m, v in zip(weights.weights, means, variances, strict=True)
        )
        self.assertAlmostEqual(result.total_variance, second - result.mean**2)
        degenerate = model_averaged_prediction([1.0, 0.0], [1.0, 2.0], [0.0, 0.0])
        self.assertEqual(degenerate.model_form_share, 0.0)


class InvalidInputTests(unittest.TestCase):
    """Malformed inputs raise."""

    def test_invalid_inputs(self) -> None:
        """Bad matrices, settings, and weights are rejected."""
        with self.assertRaises(ValueError):
            stacking_weights([])
        with self.assertRaises(ValueError):
            stacking_weights([[0.0]])
        with self.assertRaises(ValueError):
            stacking_weights([[0.0, 1.0], [0.0]])
        with self.assertRaises(ValueError):
            stacking_weights([[-math.inf, -math.inf]])
        with self.assertRaises(ValueError):
            stacking_weights([[math.nan, 0.0]])
        with self.assertRaises(ValueError):
            stacking_weights([[0.0, 1.0]], max_iterations=0)
        with self.assertRaises(ValueError):
            stacking_weights([[0.0, 1.0]], tolerance=0.0)
        with self.assertRaises(ValueError):
            pseudo_bma_weights([[0.0, 1.0]], n_bootstrap=-1)
        with self.assertRaises(ValueError):
            model_averaged_prediction([0.5, 0.6], [0.0, 1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            model_averaged_prediction([0.5, 0.5], [0.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            model_averaged_prediction([0.5, 0.5], [0.0, 1.0], [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
