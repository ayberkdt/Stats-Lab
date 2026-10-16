"""Nested aleatory-epistemic propagation and probability boxes (ADR-0064).

Oracle: ``Y = X + E`` with ``X ~ N(0, 1)`` aleatory and ``E`` in ``[a, b]``
epistemic. Then ``F_Y(y | e) = Phi(y - e)`` decreases in ``e``, so the p-box
is bounded by ``Phi(y - b)`` (lower) and ``Phi(y - a)`` (upper), attained at
the interval corners. A one-dimensional Latin hypercube of ``n`` points has
exactly one point per stratum, so its empirical CDF is within ``1/n`` of
``Phi`` everywhere; with common random numbers every outer CDF is a shifted
copy of the same empirical CDF.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Normal, Uniform
from probstat.modeling import (
    ClaytonCopula,
    IntervalInput,
    JointInput,
    ProbabilisticModel,
    RandomInput,
)
from probstat.propagation import nested_propagation
from probstat.uq import QuantityOfInterest


def _phi(z: float) -> float:
    """Return the standard normal CDF."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _additive_model(joint: JointInput) -> ProbabilisticModel:
    """Return ``Y = X + E`` with ``E`` read from inputs or fixed parameters."""
    return ProbabilisticModel(
        joint, lambda p: {"y": p["x"] + p["e"]}, (QuantityOfInterest("y", "sum"),)
    )


class IntervalPBoxTests(unittest.TestCase):
    """Analytic p-box of ``X + [a, b]``."""

    def setUp(self) -> None:
        """Run one nested propagation with a 1-D LHS inner loop."""
        self.a, self.b, self.n = -0.5, 1.0, 2000
        self.model = _additive_model(
            JointInput.independent(RandomInput("x", Normal(0.0, 1.0)))
        )
        self.result = nested_propagation(
            self.model,
            "y",
            intervals=(IntervalInput("e", self.a, self.b),),
            outer_size=5,
            inner_size=self.n,
            thresholds=(2.0,),
            probabilities=(0.9,),
            seed=1,
        )

    def test_pbox_bounds_are_shifted_normal_cdfs(self) -> None:
        """Lower/upper envelopes equal ``Phi(y-b)``/``Phi(y-a)`` within ``1/n``."""
        for y in (-2.5, -1.0, 0.0, 0.7, 2.0, 3.5):
            lower, upper = self.result.pbox.bounds(y)
            with self.subTest(y=y):
                self.assertLessEqual(
                    abs(lower - _phi(y - self.b)), 1.0 / self.n + 1e-12
                )
                self.assertLessEqual(
                    abs(upper - _phi(y - self.a)), 1.0 / self.n + 1e-12
                )
        for level, low, high in zip(
            self.result.pbox.levels,
            self.result.pbox.lower_cdf,
            self.result.pbox.upper_cdf,
            strict=True,
        ):
            self.assertLessEqual(low, high)
            self.assertLessEqual(abs(low - _phi(level - self.b)), 1.0 / self.n + 1e-12)

    def test_statistic_intervals(self) -> None:
        """Mean, exceedance and quantile intervals match the analytic bounds."""
        mean = self.result.statistics["mean"]
        self.assertAlmostEqual(mean.width, self.b - self.a, delta=1e-12)
        self.assertAlmostEqual(mean.lower, self.a, delta=0.01)
        self.assertEqual(self.result.epistemic_points[mean.argmin]["e"], self.a)
        self.assertEqual(self.result.epistemic_points[mean.argmax]["e"], self.b)
        exceed = self.result.statistics["P(Y>2)"]
        self.assertLessEqual(
            abs(exceed.lower - (1.0 - _phi(2.0 - self.a))), 1.0 / self.n
        )
        self.assertLessEqual(
            abs(exceed.upper - (1.0 - _phi(2.0 - self.b))), 1.0 / self.n
        )
        box_low, box_high = self.result.pbox.exceedance_bounds(2.0)
        self.assertAlmostEqual(box_low, exceed.lower, delta=1e-15)
        self.assertAlmostEqual(box_high, exceed.upper, delta=1e-15)
        q = self.result.statistics["q0.9"]
        z90 = 1.2815515655446004
        self.assertAlmostEqual(q.lower, self.a + z90, delta=0.01)
        self.assertAlmostEqual(q.upper, self.b + z90, delta=0.01)

    def test_cost_and_bookkeeping(self) -> None:
        """Evaluations equal outer points times inner size and match the model."""
        result = self.result
        self.assertEqual(result.point_kinds.count("corner"), 2)
        self.assertEqual(result.point_kinds.count("design"), 5)
        self.assertEqual(result.outer_size, 7)
        self.assertEqual(result.model_evaluations, 7 * self.n)
        self.assertEqual(self.model.evaluations, 7 * self.n)
        self.assertEqual(result.epistemic_names, ("e",))
        self.assertEqual(result.aleatory_names, ("x",))
        for point in result.epistemic_points:
            self.assertTrue(self.a <= point["e"] <= self.b)
        self.assertAlmostEqual(result.cdf(0, 100.0), 1.0, delta=0.0)

    def test_reproducible(self) -> None:
        """The same seed gives identical outputs."""
        again = nested_propagation(
            _additive_model(JointInput.independent(RandomInput("x", Normal(0.0, 1.0)))),
            "y",
            intervals=(IntervalInput("e", self.a, self.b),),
            outer_size=5,
            inner_size=self.n,
            thresholds=(2.0,),
            probabilities=(0.9,),
            seed=1,
        )
        self.assertEqual(again.outputs, self.result.outputs)


class EpistemicDistributionTests(unittest.TestCase):
    """Epistemic random inputs give a second-order distribution of statistics."""

    def test_mean_tracks_epistemic_value(self) -> None:
        """With common random numbers ``mean_i - e_i`` is the same for every ``i``."""
        joint = JointInput.independent(
            RandomInput("x", Normal(0.0, 1.0)),
            RandomInput("e", Normal(3.0, 0.5), nature="epistemic"),  # type: ignore[arg-type]
        )
        result = nested_propagation(
            _additive_model(joint), "y", outer_size=40, inner_size=400, seed=5
        )
        offsets = [
            mean - point["e"]
            for mean, point in zip(
                result.statistics["mean"].values, result.epistemic_points, strict=True
            )
        ]
        self.assertLess(max(offsets) - min(offsets), 1e-12)
        self.assertEqual(set(result.point_kinds), {"sample"})
        median = result.statistics["mean"].quantile(0.5)
        self.assertAlmostEqual(median, 3.0 + offsets[0], delta=0.1)

    def test_independent_inner_streams(self) -> None:
        """Without common random numbers the inner samples differ between points."""
        joint = JointInput.independent(
            RandomInput("x", Normal(0.0, 1.0)),
            RandomInput("e", Normal(3.0, 0.5), nature="epistemic"),  # type: ignore[arg-type]
        )
        result = nested_propagation(
            _additive_model(joint),
            "y",
            outer_size=3,
            inner_size=50,
            common_random_numbers=False,
            seed=5,
        )
        offsets = [
            mean - point["e"]
            for mean, point in zip(
                result.statistics["mean"].values, result.epistemic_points, strict=True
            )
        ]
        self.assertGreater(max(offsets) - min(offsets), 1e-6)


class RefinementTests(unittest.TestCase):
    """Compass refinement finds interior extremes that corners miss."""

    def test_interior_minimum(self) -> None:
        """``E[X + (E - 0.3)^2]`` over ``E in [0, 1]`` is smallest at ``E = 0.3``."""
        model = ProbabilisticModel(
            JointInput.independent(RandomInput("x", Normal(0.0, 1.0))),
            lambda p: {"y": p["x"] + (p["e"] - 0.3) ** 2},
            (QuantityOfInterest("y", "response"),),
        )
        corners_only = nested_propagation(
            model,
            "y",
            intervals=(IntervalInput("e", 0.0, 1.0),),
            outer_size=0,
            inner_size=200,
            probabilities=(),
            seed=2,
        )
        offset = corners_only.statistics["mean"].values[0] - 0.09
        self.assertAlmostEqual(
            corners_only.statistics["mean"].lower, 0.09 + offset, delta=1e-12
        )
        refined = nested_propagation(
            model,
            "y",
            intervals=(IntervalInput("e", 0.0, 1.0),),
            outer_size=0,
            inner_size=200,
            probabilities=(),
            refine=20,
            seed=2,
        )
        self.assertLess(refined.statistics["mean"].lower - offset, 1e-4)
        best = refined.epistemic_points[refined.statistics["mean"].argmin]["e"]
        self.assertAlmostEqual(best, 0.3, delta=0.01)
        self.assertIn("refine", refined.point_kinds)
        self.assertEqual(refined.model_evaluations, refined.outer_size * 200)


class ValidationTests(unittest.TestCase):
    """Argument and structure checks."""

    def test_errors(self) -> None:
        """Missing groups, clashes and dependent groups are refused."""
        aleatory_only = _additive_model(
            JointInput.independent(RandomInput("x", Normal(0.0, 1.0)))
        )
        with self.assertRaises(ValueError):
            nested_propagation(aleatory_only, "y")
        with self.assertRaises(ValueError):
            nested_propagation(
                aleatory_only, "y", intervals=(IntervalInput("x", 0.0, 1.0),)
            )
        with self.assertRaises(KeyError):
            nested_propagation(
                aleatory_only, "z", intervals=(IntervalInput("e", 0.0, 1.0),)
            )
        with self.assertRaises(ValueError):
            nested_propagation(
                aleatory_only,
                "y",
                intervals=(IntervalInput("e", 0.0, 1.0),),
                inner_design="grid",
            )
        dependent = _additive_model(
            JointInput(
                (
                    RandomInput("x", Normal(0.0, 1.0)),
                    RandomInput("e", Uniform(0.0, 1.0), nature="epistemic"),  # type: ignore[arg-type]
                ),
                copula=ClaytonCopula(2.0),
            )
        )
        with self.assertRaises(ValueError):
            nested_propagation(dependent, "y")
        correlated = _additive_model(
            JointInput(
                (
                    RandomInput("x", Normal(0.0, 1.0)),
                    RandomInput("e", Uniform(0.0, 1.0), nature="epistemic"),  # type: ignore[arg-type]
                ),
                ((1.0, 0.2), (0.2, 1.0)),
            )
        )
        with self.assertRaises(ValueError):
            nested_propagation(correlated, "y")
        with self.assertRaises(TypeError):
            nested_propagation(aleatory_only, "y", intervals=((0.0, 1.0),))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
