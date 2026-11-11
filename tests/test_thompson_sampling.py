"""Tests for the Beta-Bernoulli bandit with Thompson sampling.

Oracles: the exact probability ``P(theta_1 > theta_2)`` for independent Beta
laws with integer parameters (the second CDF is a binomial tail, integrated
against the first density by Simpson's rule), which is the Thompson
selection probability; the conjugate pseudo-count update recomputed from the
recorded actions and rewards; the regret of uniform random play; and the
known failure of the greedy rule to explore.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.risk import BanditRun, bernoulli_bandit


def _beta_exceeds(a1: int, b1: int, a2: int, b2: int) -> float:
    """Return P(X > Y), X ~ Beta(a1, b1), Y ~ Beta(a2, b2), integer parameters."""

    def density(x: float) -> float:
        """Beta(a1, b1) density."""
        log_norm = math.lgamma(a1 + b1) - math.lgamma(a1) - math.lgamma(b1)
        if x <= 0.0 or x >= 1.0:
            return math.exp(log_norm) if (x == 0.0 and a1 == 1) or (
                x == 1.0 and b1 == 1
            ) else 0.0
        return math.exp(log_norm + (a1 - 1) * math.log(x) + (b1 - 1) * math.log1p(-x))

    def cdf(x: float) -> float:
        """Beta(a2, b2) CDF as a binomial upper tail."""
        n = a2 + b2 - 1
        return sum(
            math.comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(a2, n + 1)
        )

    steps = 2000
    h = 1.0 / steps
    total = 0.0
    for i in range(steps + 1):
        x = i * h
        weight = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += weight * density(x) * cdf(x)
    return total * h / 3.0


class ThompsonSelectionTests(unittest.TestCase):
    """Thompson sampling chooses each action with its probability of being best."""

    def test_first_choice_frequency_matches_exact_probability(self) -> None:
        """Over independent runs, P(first action = 0) = P(Beta(3,1) > Beta(1,2))."""
        expected = _beta_exceeds(3, 1, 1, 2)
        runs = 4000
        hits = sum(
            bernoulli_bandit(
                [0.5, 0.5], 1, prior=[(3.0, 1.0), (1.0, 2.0)], seed=index
            ).actions[0]
            == 0
            for index in range(runs)
        )
        se = math.sqrt(expected * (1 - expected) / runs)
        self.assertAlmostEqual(hits / runs, expected, delta=4 * se)

    def test_probability_optimal_matches_exact_probability(self) -> None:
        """The posterior probability-of-optimality estimate matches the oracle."""
        run = BanditRun(
            policy="thompson",
            actions=(0,),
            rewards=(1,),
            posterior_alpha=(4.0, 2.0),
            posterior_beta=(3.0, 5.0),
            pulls=(1, 0),
            cumulative_regret=None,
        )
        estimate = run.probability_optimal(20000, seed=3)
        expected = _beta_exceeds(4, 3, 2, 5)
        se = math.sqrt(expected * (1 - expected) / 20000)
        self.assertAlmostEqual(estimate[0], expected, delta=4 * se)
        self.assertAlmostEqual(sum(estimate), 1.0, places=12)

    def test_oracle_symmetry(self) -> None:
        """The quadrature oracle satisfies P(X > Y) + P(Y > X) = 1."""
        self.assertAlmostEqual(
            _beta_exceeds(3, 2, 2, 4) + _beta_exceeds(2, 4, 3, 2), 1.0, places=6
        )
        self.assertAlmostEqual(_beta_exceeds(1, 1, 1, 1), 0.5, places=8)


class LearningTests(unittest.TestCase):
    """Posterior bookkeeping and regret behave as the theory says."""

    def test_conjugate_update_from_recorded_history(self) -> None:
        """Final pseudo-counts equal the prior plus successes and failures."""
        run = bernoulli_bandit(
            [0.3, 0.6, 0.5], 300, prior=[(1.0, 1.0), (2.0, 3.0), (0.5, 0.5)], seed=9
        )
        priors = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5)]
        for arm, (a0, b0) in enumerate(priors):
            successes = sum(
                r for x, r in zip(run.actions, run.rewards, strict=True) if x == arm
            )
            pulls = run.actions.count(arm)
            self.assertEqual(run.pulls[arm], pulls)
            self.assertEqual(run.posterior_alpha[arm], a0 + successes)
            self.assertEqual(run.posterior_beta[arm], b0 + pulls - successes)
        self.assertEqual(run.total_reward, sum(run.rewards))
        self.assertEqual(sum(run.pulls), 300)

    def test_regret_is_small_and_sublinear(self) -> None:
        """Thompson regret is far below uniform play and grows sublinearly."""
        probabilities = [0.2, 0.5, 0.45]
        horizon = 1500
        uniform = horizon * (0.5 - sum(probabilities) / 3)
        finals: list[float] = []
        first_half: list[float] = []
        for seed in range(10):
            run = bernoulli_bandit(probabilities, horizon, seed=seed)
            assert run.cumulative_regret is not None
            finals.append(run.cumulative_regret[-1])
            first_half.append(run.cumulative_regret[horizon // 2 - 1])
        mean_final = sum(finals) / len(finals)
        mean_first = sum(first_half) / len(first_half)
        self.assertLess(mean_final, 0.25 * uniform)
        self.assertLess(mean_final - mean_first, mean_first)

    def test_greedy_rule_under_explores(self) -> None:
        """Greedy play locks onto the worse action more often than Thompson."""
        probabilities = [0.6, 0.9]
        greedy = thompson = 0.0
        for seed in range(20):
            greedy_run = bernoulli_bandit(
                probabilities, 400, policy="greedy", seed=seed
            )
            ts_run = bernoulli_bandit(probabilities, 400, seed=seed)
            assert greedy_run.cumulative_regret is not None
            assert ts_run.cumulative_regret is not None
            greedy += greedy_run.cumulative_regret[-1]
            thompson += ts_run.cumulative_regret[-1]
        self.assertGreater(greedy, 2.0 * thompson)

    def test_callable_environment_and_reproducibility(self) -> None:
        """An external reward function works; equal seeds give equal runs."""

        def reward(action: int, rng: random.Random) -> int:
            """Return a success for action 1 and a failure for action 0."""
            return 1 if action == 1 else 0

        run = bernoulli_bandit(reward, 200, n_actions=2, seed=5)
        self.assertIsNone(run.cumulative_regret)
        self.assertGreater(run.pulls[1], 180)
        again = bernoulli_bandit(reward, 200, n_actions=2, seed=5)
        self.assertEqual(run, again)
        self.assertGreater(run.posterior_means[1], run.posterior_means[0])


class InvalidInputTests(unittest.TestCase):
    """Invalid settings raise."""

    def test_invalid_inputs(self) -> None:
        """Bad horizons, probabilities, priors, policies, and rewards are rejected."""
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5, 0.5], 0)
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5, 1.5], 10)
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5], 10)
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5, 0.5], 10, policy="ucb")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5, 0.5], 10, prior=(0.0, 1.0))
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5, 0.5], 10, prior=[(1.0, 1.0)] * 3)
        with self.assertRaises(ValueError):
            bernoulli_bandit([0.5, 0.5], 10, n_actions=3)
        with self.assertRaises(ValueError):
            bernoulli_bandit(lambda arm, rng: 1, 10)
        with self.assertRaises(TypeError):
            bernoulli_bandit(lambda arm, rng: 2, 10, n_actions=2)
        run = bernoulli_bandit([0.5, 0.5], 3, seed=1)
        with self.assertRaises(ValueError):
            run.probability_optimal(0)


if __name__ == "__main__":
    unittest.main()
