"""Tests for Sobol and Morris analysis on ProbabilisticModel."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Normal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.sensitivity import morris_screening, sobol_analysis
from probstat.uq import QuantityOfInterest

ISHIGAMI_FIRST = (0.3139, 0.4424, 0.0)
ISHIGAMI_TOTAL = (0.5576, 0.4424, 0.2437)


def ishigami_model() -> ProbabilisticModel:
    """Return the Ishigami benchmark with uniform inputs on [-pi, pi]."""
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )

    def model(p):
        """Evaluate the Ishigami function."""
        return {
            "y": math.sin(p["x1"])
            + 7.0 * math.sin(p["x2"]) ** 2
            + 0.1 * p["x3"] ** 4 * math.sin(p["x1"])
        }

    return ProbabilisticModel(inputs, model, (QuantityOfInterest("y", "Ishigami"),))


def linear_model(weights=(1.0, 2.0, 0.0)) -> ProbabilisticModel:
    """Return an additive model y = sum w_i x_i with standard normal inputs."""
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Normal(0.0, 1.0)) for i in range(len(weights)))
    )

    def model(p):
        """Evaluate the weighted sum."""
        return {"y": sum(w * p[f"x{i}"] for i, w in enumerate(weights))}

    return ProbabilisticModel(inputs, model, (QuantityOfInterest("y", "sum"),))


class ModelSobolTests(unittest.TestCase):
    """Variance-based indices on the central model."""

    def test_ishigami_indices_and_bootstrap_coverage(self) -> None:
        """Estimates are near the analytic values and intervals cover them."""
        result = sobol_analysis(ishigami_model(), "y", 2048, seed=3, bootstrap=120)
        self.assertEqual(result.evaluations, 2048 * 5)
        for index in range(3):
            first = result.first_order[index]
            total = result.total_effect[index]
            self.assertAlmostEqual(first.estimate, ISHIGAMI_FIRST[index], delta=0.04)
            self.assertAlmostEqual(total.estimate, ISHIGAMI_TOTAL[index], delta=0.04)
            self.assertLessEqual(first.lower, first.upper)
            self.assertLessEqual(total.lower, ISHIGAMI_TOTAL[index] + 0.02)
            self.assertGreaterEqual(total.upper, ISHIGAMI_TOTAL[index] - 0.02)
        self.assertEqual(result.ranking()[0], "x1")

    def test_linear_model_exact_shares(self) -> None:
        """Additive model: S_i = S_Ti = w_i^2 / sum w^2; a zero weight is negligible."""
        result = sobol_analysis(linear_model(), "y", 1024, seed=1, bootstrap=60)
        mapping = result.as_mapping()
        self.assertAlmostEqual(mapping["x0"][1], 0.2, delta=0.03)
        self.assertAlmostEqual(mapping["x1"][1], 0.8, delta=0.03)
        self.assertIn("x2", result.negligible(0.01))

    def test_dependent_inputs_rejected(self) -> None:
        """Sobol indices are refused for a non-identity copula."""
        base = linear_model((1.0, 1.0))
        dependent = JointInput(base.inputs.inputs, ((1.0, 0.5), (0.5, 1.0)))
        model = ProbabilisticModel(dependent, base.model, base.quantities)
        with self.assertRaises(ValueError):
            sobol_analysis(model, "y", 64, seed=1)


class MorrisTests(unittest.TestCase):
    """Elementary-effect screening."""

    def test_ranking_and_cost(self) -> None:
        """Morris ranks a strong linear input above a weak one and costs r(d+1)."""
        result = morris_screening(linear_model((3.0, 0.5, 0.0)), "y", 20, seed=2)
        self.assertEqual(result.evaluations, 20 * 4)
        self.assertEqual(result.ranking()[0], "x0")
        self.assertAlmostEqual(result.mu_star[2], 0.0, places=12)

    def test_invalid_levels(self) -> None:
        """Odd level counts are rejected."""
        with self.assertRaises(ValueError):
            morris_screening(linear_model(), "y", 5, levels=5)


if __name__ == "__main__":
    unittest.main()
