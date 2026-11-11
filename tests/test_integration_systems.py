"""Regression tests for interactions between independently built features."""

from __future__ import annotations

import unittest

from probstat.distributions import Normal
from probstat.modeling import ClaytonCopula, JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import ModelLimitState, form_hlrf
from probstat.sensitivity import sobol_analysis
from probstat.uq import QuantityOfInterest


def _margin(point):
    """Return a linear margin of two inputs."""
    return {"g": 3.0 - point["a"] - point["b"]}


class FormAutodiffCopulaTests(unittest.TestCase):
    """Autodiff FORM must not apply the Nataf Jacobian to Rosenblatt maps."""

    def test_clayton_inputs_fall_back_to_finite_differences(self) -> None:
        """The automatic gradient falls back and agrees with finite differences."""
        inputs = JointInput(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            copula=ClaytonCopula(2.0, 2),
        )
        model = ProbabilisticModel(inputs, _margin, (QuantityOfInterest("g", "g"),))
        state = ModelLimitState(model, "g", 0.0, failure="below")
        reference = form_hlrf(state, gradient="finite-difference")
        automatic = form_hlrf(state, gradient="auto")
        self.assertEqual(automatic.gradient_method, "finite-difference")
        self.assertAlmostEqual(automatic.beta, reference.beta, 10)


def _offset_model(offset: float) -> ProbabilisticModel:
    """Return ``offset + a + 0.3 b`` on independent normal inputs."""
    inputs = JointInput.independent(
        RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))
    )

    def function(point):
        """Return the shifted linear output."""
        return {"y": offset + point["a"] + 0.3 * point["b"]}

    return ProbabilisticModel(inputs, function, (QuantityOfInterest("y", "y"),))


class SobolOffsetInvarianceTests(unittest.TestCase):
    """First-order Sobol estimates must not depend on the output mean."""

    def test_large_offset_leaves_indices_unchanged(self) -> None:
        """A constant 1e6 offset changes nothing beyond rounding."""
        base = sobol_analysis(_offset_model(0.0), "y", 512, design="mc", seed=4)
        shifted = sobol_analysis(_offset_model(1e6), "y", 512, design="mc", seed=4)
        for name in ("a", "b"):
            first_base, total_base = base.as_mapping()[name]
            first_shift, total_shift = shifted.as_mapping()[name]
            self.assertAlmostEqual(first_base, first_shift, 6)
            self.assertAlmostEqual(total_base, total_shift, 6)
        # Exact indices: 1 / 1.09 and 0.09 / 1.09.
        self.assertLess(abs(shifted.as_mapping()["a"][0] - 1.0 / 1.09), 0.08)


if __name__ == "__main__":
    unittest.main()
