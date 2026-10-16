"""Regression tests for interactions between independently built features."""

from __future__ import annotations

import unittest

from probstat.distributions import Normal
from probstat.modeling import ClaytonCopula, JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import ModelLimitState, form_hlrf
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


if __name__ == "__main__":
    unittest.main()
