"""Tests for adaptive conformal inference (Gibbs and Candes 2021).

Oracles: the deterministic long-run bound of Proposition 4.1 and the range
``alpha_t in [-gamma, 1 + gamma]`` of Lemma 4.1, which hold for every data
sequence (checked on adversarial and shifting sequences); a hand-computed
three-step trace; ``gamma = 0`` reproducing the fixed conformal order
statistic; and a seeded variance-shift experiment where fixed-level conformal
under-covers after the shift and ACI does not.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.conformal import (
    AdaptiveConformalPredictor,
    adaptive_conformal_intervals,
    conformal_quantile,
)


def _shifting_sequence(seed: int, size: int) -> tuple[list[float], list[float]]:
    """Responses whose noise scale jumps from 1 to 4 halfway; predictions 0."""
    rng = random.Random(seed)
    observations = [
        rng.gauss(0.0, 1.0 if t < size // 2 else 4.0) for t in range(size)
    ]
    return [0.0] * size, observations


class GuaranteeTests(unittest.TestCase):
    """The distribution-free guarantees hold on arbitrary sequences."""

    def test_long_run_bound_holds_on_adversarial_sequences(self) -> None:
        """The realized miss rate obeys Proposition 4.1 for every gamma."""
        rng = random.Random(4)
        sequences = (
            [rng.gauss(0.0, 1.0 + t / 50.0) for t in range(600)],
            [(-1.0) ** t * (t % 37) for t in range(600)],
            [1000.0 if t % 7 == 0 else 0.0 for t in range(600)],
            [rng.paretovariate(0.8) for _ in range(600)],
        )
        for observations in sequences:
            for gamma in (0.001, 0.01, 0.05, 0.3):
                result = adaptive_conformal_intervals(
                    [0.0] * len(observations), observations, alpha=0.1, gamma=gamma
                )
                gap = abs(result.miscoverage_rate - 0.1)
                self.assertLessEqual(gap, result.long_run_bound + 1e-9)
                for level in result.alphas:
                    self.assertGreaterEqual(level, -gamma - 1e-9)
                    self.assertLessEqual(level, 1.0 + gamma + 1e-9)

    def test_bound_formula(self) -> None:
        """Bound equals (max(a1, 1-a1) + gamma) / (gamma T)."""
        result = adaptive_conformal_intervals(
            [0.0] * 50, [1.0] * 50, alpha=0.2, gamma=0.05, initial_alpha=0.3
        )
        self.assertAlmostEqual(result.long_run_bound, (0.7 + 0.05) / (0.05 * 50))
        frozen = adaptive_conformal_intervals([0.0] * 5, [1.0] * 5, gamma=0.0)
        self.assertEqual(frozen.long_run_bound, math.inf)

    def test_recovers_coverage_after_a_variance_shift(self) -> None:
        """After the shift ACI covers near 1 - alpha; fixed conformal does not."""
        predictions, observations = _shifting_sequence(seed=12, size=4000)
        adaptive = adaptive_conformal_intervals(
            predictions, observations, alpha=0.1, gamma=0.02, window=300
        )
        fixed = adaptive_conformal_intervals(
            predictions, observations, alpha=0.1, gamma=0.0
        )
        late = slice(2500, 4000)
        adaptive_miss = sum(adaptive.errors[late]) / 1500
        fixed_miss = sum(fixed.errors[late]) / 1500
        se = math.sqrt(0.1 * 0.9 / 1500)
        self.assertLess(abs(adaptive_miss - 0.1), 4 * se)
        self.assertGreater(fixed_miss, 0.1 + 4 * se)


class MechanicsTests(unittest.TestCase):
    """The update, quantile, and interval mechanics match hand computation."""

    def test_hand_computed_trace(self) -> None:
        """Three steps from calibration scores (1, 2, 3), alpha 0.25, gamma 0.1."""
        predictor = AdaptiveConformalPredictor(
            0.25, 0.1, calibration_scores=(1.0, 2.0, 3.0)
        )
        # k = ceil(0.75 * 4) = 3 -> q = 3.
        self.assertEqual(predictor.radius(), 3.0)
        interval = predictor.interval(10.0)
        self.assertEqual((interval.lower, interval.upper), (7.0, 13.0))
        self.assertTrue(predictor.update(10.0, 15.0))  # score 5 > 3: miss
        self.assertAlmostEqual(predictor.current_alpha, 0.25 + 0.1 * (0.25 - 1.0))
        # alpha = 0.175, history (1, 2, 3, 5): k = ceil(0.825 * 5) = 5 > 4.
        self.assertEqual(predictor.radius(), math.inf)
        self.assertFalse(predictor.update(0.0, 100.0))
        self.assertAlmostEqual(predictor.current_alpha, 0.175 + 0.1 * 0.25)
        # alpha = 0.2, history (1, 2, 3, 5, 100): k = ceil(0.8 * 6) = 5 -> 100.
        self.assertEqual(predictor.radius(), 100.0)
        self.assertEqual(predictor.errors, (True, False))

    def test_gamma_zero_matches_fixed_conformal_quantile(self) -> None:
        """With gamma = 0 each radius is the split conformal quantile of the past."""
        rng = random.Random(8)
        observations = [rng.gauss(0.0, 1.0) for _ in range(60)]
        result = adaptive_conformal_intervals(
            [0.0] * 60, observations, alpha=0.2, gamma=0.0
        )
        for t in range(1, 60):
            expected = conformal_quantile([abs(y) for y in observations[:t]], 0.2)
            self.assertEqual(result.intervals[t].upper, expected)

    def test_boundary_levels_give_whole_line_and_empty_set(self) -> None:
        """alpha_t >= 1 gives the empty set and alpha_t <= 0 the whole line."""
        empty = AdaptiveConformalPredictor(0.5, 0.1, initial_alpha=1.0)
        interval = empty.interval(0.0)
        self.assertLess(interval.upper, interval.lower)
        self.assertTrue(empty.update(0.0, 0.0))
        whole = AdaptiveConformalPredictor(0.5, 0.1, initial_alpha=0.0)
        interval = whole.interval(2.0)
        self.assertEqual((interval.lower, interval.upper), (-math.inf, math.inf))
        self.assertFalse(whole.update(0.0, 1e300))

    def test_scaled_scores_and_window(self) -> None:
        """Normalized scores scale the radius; the window drops old scores."""
        predictor = AdaptiveConformalPredictor(
            0.4, 0.0, calibration_scores=(100.0, 1.0, 1.0), window=2
        )
        # The window keeps (1, 1): k = ceil(0.6 * 3) = 2 -> q = 1.
        self.assertEqual(predictor.radius(), 1.0)
        interval = predictor.interval(5.0, scale=3.0)
        self.assertEqual((interval.lower, interval.upper), (2.0, 8.0))
        self.assertFalse(predictor.update(5.0, 7.0, scale=3.0))

    def test_local_coverage(self) -> None:
        """Trailing-window coverage averages the most recent indicators."""
        result = adaptive_conformal_intervals(
            [0.0] * 4, [1.0, 1.0, 1.0, 1.0], alpha=0.5, gamma=0.0,
            calibration_scores=(2.0,),
        )
        coverage = result.local_coverage(2)
        self.assertEqual(len(coverage), 4)
        for value in coverage:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        manual = [
            1.0 - sum(result.errors[max(0, t - 1) : t + 1]) / min(t + 1, 2)
            for t in range(4)
        ]
        self.assertEqual(list(coverage), manual)


class InvalidInputTests(unittest.TestCase):
    """Invalid settings and data raise."""

    def test_invalid_inputs(self) -> None:
        """Bad levels, steps, windows, scores, and sequences are rejected."""
        with self.assertRaises(ValueError):
            AdaptiveConformalPredictor(1.0)
        with self.assertRaises(ValueError):
            AdaptiveConformalPredictor(0.1, -0.1)
        with self.assertRaises(ValueError):
            AdaptiveConformalPredictor(0.1, initial_alpha=1.5)
        with self.assertRaises(ValueError):
            AdaptiveConformalPredictor(0.1, window=0)
        with self.assertRaises(ValueError):
            AdaptiveConformalPredictor(0.1, calibration_scores=(-1.0,))
        with self.assertRaises(TypeError):
            AdaptiveConformalPredictor(0.1, window=2.5)  # type: ignore[arg-type]
        predictor = AdaptiveConformalPredictor(0.1)
        with self.assertRaises(ValueError):
            predictor.interval(0.0, scale=0.0)
        with self.assertRaises(ValueError):
            predictor.update(0.0, math.nan)
        with self.assertRaises(ValueError):
            adaptive_conformal_intervals([], [])
        with self.assertRaises(ValueError):
            adaptive_conformal_intervals([0.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            adaptive_conformal_intervals([0.0], [1.0], scales=[1.0, 1.0])
        result = adaptive_conformal_intervals([0.0], [1.0])
        with self.assertRaises(ValueError):
            result.local_coverage(0)


if __name__ == "__main__":
    unittest.main()
