"""Tests for the published benchmark models in ``probstat.modeling``."""

from __future__ import annotations

import math
import unittest

from probstat.modeling import (
    FLOOD_INPUT_NAMES,
    FLOOD_NOMINAL,
    FOUR_BRANCH_REFERENCE_PROBABILITY,
    ISHIGAMI_FIRST_ORDER,
    ISHIGAMI_TOTAL,
    flood_cost,
    flood_model,
    flood_overflow,
    four_branch,
    four_branch_model,
    ishigami,
    ishigami_model,
)


class FloodModelTests(unittest.TestCase):
    """The flood model reproduces the published formulas and input laws."""

    def test_nominal_point(self) -> None:
        """At the nominal point the dike is 10.97 m above the water."""
        height = (1355.0 / (30.5 * 300.0 * math.sqrt(5.0 / 5000.0))) ** 0.6
        expected = 50.0 + height - 8.0 - 55.5
        self.assertAlmostEqual(flood_overflow(FLOOD_NOMINAL), expected, places=12)
        damage = 0.2 + 0.8 * (1.0 - math.exp(-1000.0 / expected**4))
        self.assertAlmostEqual(flood_cost(FLOOD_NOMINAL), damage + 0.4, places=12)

    def test_cost_branches(self) -> None:
        """An overflow costs one, and a dike above 8 m adds its height / 20."""
        point = dict(FLOOD_NOMINAL, Q=1.0e5, Hd=8.5)
        self.assertGreater(flood_overflow(point), 0.0)
        self.assertAlmostEqual(flood_cost(point), 1.0 + 8.5 / 20.0, places=12)

    def test_invalid_geometry_raises(self) -> None:
        """A river bed rising downstream is rejected."""
        with self.assertRaises(ValueError):
            flood_overflow(dict(FLOOD_NOMINAL, Zm=49.0))

    def test_input_laws_match_r(self) -> None:
        """Truncated Gumbel and normal quantiles equal R's evd-based values."""
        model = flood_model()
        self.assertEqual(
            tuple(item.name for item in model.inputs.inputs), FLOOD_INPUT_NAMES
        )
        discharge = model.inputs.inputs[0].distribution
        strickler = model.inputs.inputs[1].distribution
        # R 4.6.2: qgumbel((1-p)*pgumbel(500,1013,555)+p*pgumbel(3000,1013,555),
        # 1013, 555) and the matching qnorm expression for N(30, 8) above 15.
        for p, q_ref, k_ref in (
            (0.01, 523.656076185862, 16.0030812346677),
            (0.5, 1259.35419003734, 30.3048432765244),
            (0.99, 2841.459145747592, 48.7032560466816),
        ):
            self.assertAlmostEqual(discharge.quantile(p) / q_ref, 1.0, places=12)
            self.assertAlmostEqual(strickler.quantile(p) / k_ref, 1.0, places=12)

    def test_model_outputs(self) -> None:
        """The model returns both outputs through the shared contract."""
        values = flood_model().evaluate(dict(FLOOD_NOMINAL))
        self.assertAlmostEqual(values["overflow"], flood_overflow(FLOOD_NOMINAL))
        self.assertAlmostEqual(values["cost"], flood_cost(FLOOD_NOMINAL))


class FourBranchTests(unittest.TestCase):
    """The four-branch limit state has its two pairs of design points."""

    def test_design_point_distances(self) -> None:
        """The inner branches vanish at radius 3, the outer ones at 3.5."""
        root = 1.0 / math.sqrt(2.0)
        for sign in (1.0, -1.0):
            self.assertAlmostEqual(
                four_branch(3.0 * sign * root, 3.0 * sign * root), 0.0
            )
            self.assertAlmostEqual(
                four_branch(3.5 * sign * root, -3.5 * sign * root), 0.0
            )
        self.assertAlmostEqual(four_branch(0.0, 0.0), 3.0)

    def test_model_and_reference(self) -> None:
        """The model evaluates the function and keeps the offset fixed."""
        model = four_branch_model(6.0)
        self.assertAlmostEqual(
            model.evaluate({"u1": 1.0, "u2": -2.0})["g"], four_branch(1.0, -2.0, 6.0)
        )
        self.assertTrue(2.2e-3 < FOUR_BRANCH_REFERENCE_PROBABILITY < 2.25e-3)
        with self.assertRaises(ValueError):
            four_branch_model(0.0)


class IshigamiTests(unittest.TestCase):
    """The Ishigami constants equal the analytic variance decomposition."""

    def test_indices(self) -> None:
        """First-order and total indices follow from the closed-form variances."""
        a, b, pi = 7.0, 0.1, math.pi
        variance = a * a / 8 + b * pi**4 / 5 + b * b * pi**8 / 18 + 0.5
        v1 = 0.5 * (1.0 + b * pi**4 / 5) ** 2
        v2 = a * a / 8
        v13 = b * b * pi**8 * (1.0 / 18.0 - 1.0 / 50.0)
        exact_first = (v1 / variance, v2 / variance, 0.0)
        exact_total = ((v1 + v13) / variance, v2 / variance, v13 / variance)
        for value, exact in zip(ISHIGAMI_FIRST_ORDER, exact_first, strict=True):
            self.assertAlmostEqual(value, exact, places=4)
        for value, exact in zip(ISHIGAMI_TOTAL, exact_total, strict=True):
            self.assertAlmostEqual(value, exact, places=4)

    def test_model(self) -> None:
        """The model evaluates the function with its fixed weights."""
        value = ishigami_model().evaluate({"x1": 1.0, "x2": 2.0, "x3": 0.5})["y"]
        self.assertAlmostEqual(value, ishigami(1.0, 2.0, 0.5))
        with self.assertRaises(ValueError):
            ishigami_model(a=math.inf)


if __name__ == "__main__":
    unittest.main()
