"""Tests for Sobol indices of stochastic simulators with replications.

Oracle model, inputs ``X_1..X_4 ~ U(-1, 1)`` independent (variance 1/3):
``Y = X_1 + 2 X_2 + 3 X_2 X_3 + (0.5 + 0.5 X_1^2) eps``, ``eps ~ N(0, 1)``.
The mean response ``m = X_1 + 2 X_2 + 3 X_2 X_3`` has ``Var(m) = 8/3`` with
``V_1 = 1/3``, ``V_2 = 4/3``, ``V_3 = V_4 = 0`` and total effects
``T_1 = 1/3``, ``T_2 = 7/3``, ``T_3 = 1``, ``T_4 = 0``; the noise variance is
``E[(0.5 + 0.5 X_1^2)^2] = 1/4 + 1/6 + 1/20 = 7/15``. Tolerances are four
standard errors estimated from independent seeded repetitions. A zero-noise
simulator must reproduce the deterministic pick-freeze numerators exactly.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.sensitivity import estimate_sobol_indices, stochastic_sobol_indices

VAR_M = 8.0 / 3.0
NOISE = 7.0 / 15.0
FIRST = (1.0 / 8.0, 4.0 / 8.0, 0.0, 0.0)
TOTAL = (1.0 / 8.0, 7.0 / 8.0, 3.0 / 8.0, 0.0)


def _simulator(x: tuple[float, ...], seed: int) -> float:
    """Evaluate the heteroscedastic oracle model with one seeded noise draw."""
    eps = random.Random(seed).gauss(0.0, 1.0)
    return x[0] + 2 * x[1] + 3 * x[1] * x[2] + (0.5 + 0.5 * x[0] ** 2) * eps


def _uniform_matrix(rng: random.Random, rows: int, columns: int) -> list[list[float]]:
    """Draw a matrix of independent U(-1, 1) inputs."""
    return [[rng.uniform(-1.0, 1.0) for _ in range(columns)] for _ in range(rows)]


class StochasticSobolOracleTests(unittest.TestCase):
    """Estimates agree with the closed-form decomposition."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run five independent replicated studies of the oracle model."""
        cls.results = []
        for repeat in range(5):
            rng = random.Random(100 + repeat)
            cls.results.append(
                stochastic_sobol_indices(
                    _simulator,
                    _uniform_matrix(rng, 1500, 4),
                    _uniform_matrix(rng, 1500, 4),
                    replications=2,
                    seed=repeat,
                )
            )

    def _check(self, values: list[float], expected: float, floor: float) -> None:
        """Mean of repeats within four standard errors (with a small floor)."""
        mean = statistics.fmean(values)
        se = statistics.stdev(values) / math.sqrt(len(values))
        self.assertAlmostEqual(mean, expected, delta=max(4.0 * se, floor))

    def test_variance_components(self) -> None:
        """Var(m), E[Var(Y | X)], and the noise index match the oracle."""
        self._check([r.mean_variance for r in self.results], VAR_M, 0.02)
        self._check([r.noise_variance for r in self.results], NOISE, 0.005)
        self._check(
            [r.noise_index for r in self.results], NOISE / (VAR_M + NOISE), 0.005
        )
        for result in self.results:
            self.assertAlmostEqual(
                result.output_variance, result.mean_variance + result.noise_variance
            )

    def test_mean_response_indices(self) -> None:
        """First-order and total indices of m match their closed forms."""
        for i in range(4):
            self._check([r.first_order[i] for r in self.results], FIRST[i], 0.01)
            self._check([r.total_effect[i] for r in self.results], TOTAL[i], 0.01)

    def test_output_shares_and_inert_input(self) -> None:
        """Output shares rescale by Var(m)/Var(Y); the inert input's total is 0."""
        share = VAR_M / (VAR_M + NOISE)
        self._check(
            [r.first_order_output[1] for r in self.results], FIRST[1] * share, 0.01
        )
        naive_bias = NOISE / (2 * VAR_M)
        mean_inert = statistics.fmean(r.total_effect[3] for r in self.results)
        self.assertLess(abs(mean_inert), naive_bias / 4)

    def test_cost_accounting(self) -> None:
        """Evaluations equal N (d + 2) R."""
        result = self.results[0]
        self.assertEqual(result.evaluations, 1500 * 6 * 2)
        self.assertEqual((result.sample_size, result.replications), (1500, 2))
        self.assertAlmostEqual(
            result.unexplained_mean_variance, 1.0 - sum(result.first_order)
        )


class DeterministicLimitTests(unittest.TestCase):
    """Without intrinsic noise the method reduces to ordinary pick-freeze."""

    def test_zero_noise_reproduces_deterministic_numerators(self) -> None:
        """V_i and T_i numerators equal those of estimate_sobol_indices."""

        def model(x: tuple[float, ...]) -> float:
            """Evaluate the Ishigami-type deterministic mean response."""
            first = math.sin(x[0])
            return first + 7 * math.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * first

        rng = random.Random(4)
        a = [[rng.uniform(-math.pi, math.pi) for _ in range(3)] for _ in range(200)]
        b = [[rng.uniform(-math.pi, math.pi) for _ in range(3)] for _ in range(200)]
        stochastic = stochastic_sobol_indices(
            lambda x, seed: model(x), a, b, replications=2, seed=1
        )
        deterministic = estimate_sobol_indices(model, a, b)
        self.assertEqual(stochastic.noise_variance, 0.0)
        self.assertEqual(stochastic.noise_index, 0.0)
        for i in range(3):
            self.assertAlmostEqual(
                stochastic.first_order[i] * stochastic.mean_variance,
                deterministic.first_order[i] * deterministic.output_variance,
                places=10,
            )
            self.assertAlmostEqual(
                stochastic.total_effect[i] * stochastic.mean_variance,
                deterministic.total_effect[i] * deterministic.output_variance,
                places=10,
            )

    def test_reproducible_by_seed(self) -> None:
        """Equal seeds give identical results."""
        rng = random.Random(2)
        a = _uniform_matrix(rng, 50, 4)
        b = _uniform_matrix(rng, 50, 4)
        first = stochastic_sobol_indices(_simulator, a, b, seed=8)
        second = stochastic_sobol_indices(_simulator, a, b, seed=8)
        self.assertEqual(first, second)


class InvalidInputTests(unittest.TestCase):
    """Invalid designs and settings raise."""

    def test_invalid_inputs(self) -> None:
        """Malformed matrices, replications, outputs, and models are rejected."""
        a = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.9]]
        b = [[0.2, 0.1], [0.6, 0.7], [0.8, 0.3]]
        with self.assertRaises(TypeError):
            stochastic_sobol_indices(None, a, b)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            stochastic_sobol_indices(_simulator, a, b[:2])
        with self.assertRaises(ValueError):
            stochastic_sobol_indices(_simulator, a[:1], b[:1])
        with self.assertRaises(ValueError):
            stochastic_sobol_indices(_simulator, a, b, replications=1)
        with self.assertRaises(ValueError):
            stochastic_sobol_indices(_simulator, [[]], [[]])
        with self.assertRaises(ValueError):
            stochastic_sobol_indices(lambda x, seed: math.nan, a, b, seed=1)
        with self.assertRaises(ValueError):
            stochastic_sobol_indices(lambda x, seed: 1.0, a, b, seed=1)


if __name__ == "__main__":
    unittest.main()
