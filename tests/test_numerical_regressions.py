"""Regression tests for numerical defects found by independent review."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Normal
from probstat.inference import NormalInverseGamma, student_t_cdf, student_t_quantile
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import ModelLimitState, form_hlrf, series_system_bounds
from probstat.risk import BasicEvent, FaultTree, and_gate, or_gate
from probstat.uq import QuantityOfInterest


class FaultTreeRareEventTests(unittest.TestCase):
    """OR gates and bounds keep relative accuracy for rare events."""

    def test_or_gate_of_tiny_probabilities(self) -> None:
        """P(A or B) = 2p - p^2 stays accurate down to p = 1e-17."""
        for p in (1e-10, 1e-13, 1e-17):
            tree = FaultTree(or_gate("top", BasicEvent("A", p), BasicEvent("B", p)))
            exact = 2.0 * p - p * p
            self.assertAlmostEqual(tree.top_event_probability() / exact, 1.0, places=12)
            self.assertAlmostEqual(tree.min_cut_upper_bound() / exact, 1.0, places=12)
            fussell = tree.importance()["A"].fussell_vesely
            self.assertAlmostEqual(fussell, 0.5, places=6)
            self.assertLessEqual(fussell, 1.0)

    def test_many_repeated_or_gates_expand_without_blow_up(self) -> None:
        """AND of 17 OR(A, B) gates has exactly the cut sets {A} and {B}."""
        a, b = BasicEvent("A", 0.1), BasicEvent("B", 0.2)
        gates = [or_gate(f"G{i}", a, b) for i in range(17)]
        tree = FaultTree(and_gate("top", *gates))
        self.assertEqual(
            set(tree.minimal_cut_sets), {frozenset({"A"}), frozenset({"B"})}
        )
        self.assertAlmostEqual(tree.top_event_probability(), 0.1 + 0.2 - 0.02)


class SeriesBoundTests(unittest.TestCase):
    """Uni-modal series bounds do not collapse for very rare components."""

    def test_unimodal_upper_at_beta_eight_and_a_half(self) -> None:
        """Two independent components at beta = 8.5 keep upper >= lower."""
        inputs = JointInput.independent(
            RandomInput("x", Normal(0.0, 1.0)), RandomInput("y", Normal(0.0, 1.0))
        )
        model = ProbabilisticModel(
            inputs,
            lambda p: {"gx": p["x"], "gy": p["y"]},
            (QuantityOfInterest("gx", "x"), QuantityOfInterest("gy", "y")),
        )
        forms = [
            form_hlrf(ModelLimitState(model, name, 8.5, failure="above"))
            for name in ("gx", "gy")
        ]
        bounds = series_system_bounds(forms)
        component = 0.5 * math.erfc(8.5 / math.sqrt(2.0))
        self.assertGreater(bounds.unimodal_upper, 0.0)
        self.assertAlmostEqual(bounds.unimodal_upper / (2.0 * component), 1.0, places=6)
        self.assertGreaterEqual(bounds.unimodal_upper, bounds.unimodal_lower)


class StudentTAccuracyTests(unittest.TestCase):
    """Student-t evaluation keeps relative accuracy in tails and for large nu."""

    def test_far_lower_tail_quantile(self) -> None:
        """Quantiles at p = 1e-17 invert back to p with relative accuracy."""
        for nu in (1.0, 30.0):
            q = student_t_quantile(1e-17, nu)
            self.assertAlmostEqual(student_t_cdf(q, nu) / 1e-17, 1.0, places=6)
        self.assertAlmostEqual(
            student_t_quantile(1e-17, 1.0), -1.0 / math.tan(math.pi * 1e-17), delta=1e3
        )

    def test_center_for_large_degrees_of_freedom(self) -> None:
        """For nu = 1e7 the CDF near zero matches the normal density slope."""
        expected = 0.5 + 1e-5 / math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(student_t_cdf(1e-5, 1e7), expected, places=10)


class NormalInverseGammaIntervalTests(unittest.TestCase):
    """Noise-variance intervals for vague inverse-gamma priors."""

    def test_small_shapes_do_not_crash(self) -> None:
        """Shapes down to 0.001 give ordered bounds (possibly infinite)."""
        for shape in (0.1, 0.05, 0.01, 0.001):
            law = NormalInverseGamma((0.0,), ((1.0,),), shape, shape)
            low, high = law.noise_variance_interval(0.95)
            self.assertGreater(low, 0.0)
            self.assertLessEqual(low, high)

    def test_known_upper_bound_for_shape_one_tenth(self) -> None:
        """InvGamma(0.1, 0.1) 97.5% quantile is about 1.727e15."""
        law = NormalInverseGamma((0.0,), ((1.0,),), 0.1, 0.1)
        self.assertAlmostEqual(
            law.noise_variance_interval(0.95)[1] / 1.7266e15, 1.0, delta=1e-3
        )


if __name__ == "__main__":
    unittest.main()


class CalibrationOverflowTests(unittest.TestCase):
    """Extreme proposals for log-sigma are rejected, not raised."""

    def test_huge_log_sigma_gives_minus_infinity(self) -> None:
        """log_posterior at log sigma = 710 returns -inf."""
        from probstat.distributions import LogNormal, Uniform
        from probstat.inverse import CalibrationProblem, Experiment, NoiseModel

        model = ProbabilisticModel(
            JointInput.independent(RandomInput("k", Uniform(0.0, 4.0))),
            lambda p: {"y": p["k"] * p["t"]},
            (QuantityOfInterest("y", "response"),),
            fixed={"t": 1.0},
        )
        problem = CalibrationProblem(
            model,
            (Experiment({"y": 2.1}, {"t": 1.0}), Experiment({"y": 3.9}, {"t": 2.0})),
            (NoiseModel("y", standard_deviation_prior=LogNormal(-1.0, 1.0)),),
        )
        self.assertEqual(problem.log_posterior((1.5, 710.0)), -math.inf)
        self.assertEqual(problem.standard_log_posterior((0.0, 710.0)), -math.inf)
        self.assertTrue(math.isfinite(problem.log_posterior((2.0, math.log(0.2)))))
