"""Regression tests of the workflow, generative, and figure defect review (ADR-0094).

Oracles: direct recomputation with :func:`probstat.sensitivity.sobol_analysis`
on the documented stage stream (identical numbers, counted model runs);
the type-7 quantile identity ``quantile(-y, p) = -quantile(y, 1 - p)``;
50-digit decimal evaluations (:mod:`probstat.numerics.reference`) of the
planar and radial log-determinants; exact constant data for ActNorm; and
the limits stored in a :class:`probstat.risk.ReturnLevel`.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pickle
import unittest
import warnings
from decimal import Decimal, localcontext

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.generative import ActNorm, NormalizingFlow, PlanarFlow, RadialFlow
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.numerics import reference_exp, reference_log
from probstat.sensitivity import sobol_analysis
from probstat.statistics import quantile
from probstat.uq import EvaluationBudgetExceeded, QuantityOfInterest
from probstat.workflow import UQStudy

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
DIGITS = 50


def three_outputs(point):
    """Return a linear, a product, and a constant QoI of ``(x, y)``."""
    return {"a": point["x"] + 2.0 * point["y"], "b": point["x"] * point["y"], "c": 3.0}


def three_output_model() -> ProbabilisticModel:
    """Return ``x ~ U(0, 1)``, ``y ~ N(0, 1)`` with QoIs ``a``, ``b``, ``c``."""
    inputs = JointInput.independent(
        RandomInput("x", Uniform(0.0, 1.0)), RandomInput("y", Normal(0.0, 1.0))
    )
    quantities = tuple(QuantityOfInterest(name, name) for name in "abc")
    return ProbabilisticModel(inputs, three_outputs, quantities, name="three")


def capacity_model() -> ProbabilisticModel:
    """Return ``g = R - S`` with lognormal resistance and normal load."""
    inputs = JointInput.independent(
        RandomInput("R", LogNormal(math.log(5.0), 0.1)),
        RandomInput("S", Normal(3.0, 0.8)),
    )
    return ProbabilisticModel(
        inputs,
        lambda p: {"g": p["R"] - p["S"]},
        (QuantityOfInterest("g", "margin"),),
        name="capacity",
    )


def decimal_softplus(value: float) -> Decimal:
    """Return ``log(1 + exp(value))`` to 50 digits."""
    with localcontext() as context:
        context.prec = DIGITS + 10
        if value > 0.0:
            tail = reference_exp(-value, digits=DIGITS)
            return Decimal(value) + reference_log(1 + tail, digits=DIGITS)
        return reference_log(1 + reference_exp(value, digits=DIGITS), digits=DIGITS)


def decimal_tanh(value: float) -> Decimal:
    """Return ``tanh(value)`` to 50 digits."""
    with localcontext() as context:
        context.prec = DIGITS + 10
        e = reference_exp(2 * Decimal(value), digits=DIGITS)
        return (e - 1) / (e + 1)


class SensitivityStageTests(unittest.TestCase):
    """``UQStudy.sensitivity`` uses one shared design for every QoI (ADR-0065)."""

    def test_one_design_serves_every_quantity_at_one_cost(self) -> None:
        """A second QoI reuses the design: same numbers, zero new runs."""
        model = three_output_model()
        study = UQStudy(model, seed=5)
        first = study.sensitivity("a", 64)
        second = study.sensitivity("b", 64)
        stream = study._root.child("sensitivity", 0)
        oracle = three_output_model()
        expected_a = sobol_analysis(oracle, "a", 64, seed=stream, bootstrap=200)
        expected_b = sobol_analysis(oracle, "b", 64, seed=stream, bootstrap=200)
        self.assertEqual(first.as_mapping(), expected_a.as_mapping())
        self.assertEqual(second.as_mapping(), expected_b.as_mapping())
        self.assertEqual(first.first_order, expected_a.first_order)
        self.assertEqual(second.total_effect, expected_b.total_effect)
        cost = 64 * (model.dimension + 2)
        self.assertEqual([r.evaluations for r in study.records], [cost, 0])
        self.assertEqual(model.evaluations, cost)
        self.assertEqual(study.records[1].seed, study.records[0].seed)
        self.assertIn("shared", str(study.records[1].summary["design"]))
        self.assertEqual(study.records[0].summary["also estimated by this design"], "b")

    def test_constant_quantity_is_skipped_without_extra_runs(self) -> None:
        """A zero-variance QoI does not block the others and raises alone."""
        model = three_output_model()
        study = UQStudy(model, seed=5)
        study.sensitivity("a", 64)
        self.assertEqual(model.evaluations, 64 * 4)
        self.assertNotIn("c", str(study.records[0].summary))
        with self.assertRaises(ValueError):
            study.sensitivity("c", 64)

    def test_requested_quantity_matches_the_single_quantity_call(self) -> None:
        """One-QoI studies keep their seeded numbers exactly."""
        study = UQStudy(capacity_model(), seed=7)
        result = study.sensitivity("g", 64)
        expected = sobol_analysis(
            capacity_model(), "g", 64,
            seed=study._root.child("sensitivity", 0), bootstrap=200,
        )  # fmt: skip
        self.assertEqual(result.first_order, expected.first_order)
        self.assertEqual(result.total_effect, expected.total_effect)
        self.assertEqual(study.records[0].evaluations, 64 * 4)

    def test_summary_states_the_interval_method_and_limits(self) -> None:
        """Replicate intervals are named as such, with their limits."""
        study = UQStudy(capacity_model(), seed=7)
        result = study.sensitivity("g", 64)
        summary = study.records[-1].summary
        self.assertEqual(result.interval_method, "replicates")
        self.assertIn("8 independent randomizations of 8 rows", summary["intervals"])
        self.assertEqual(summary["S[R].low"], result.first_order[0].lower)
        self.assertEqual(summary["ST[S].high"], result.total_effect[1].upper)
        mc = UQStudy(capacity_model(), seed=7)
        boot = mc.sensitivity("g", 64, design="mc", bootstrap=25)
        self.assertEqual(boot.interval_method, "bootstrap")
        self.assertIn("25 row bootstrap", mc.records[-1].summary["intervals"])
        none = UQStudy(capacity_model(), seed=7)
        none.sensitivity("g", 64, bootstrap=0)
        self.assertEqual(none.records[-1].summary["intervals"], "none")
        self.assertNotIn("S[R].low", none.records[-1].summary)

    def test_new_input_law_or_settings_run_a_new_design(self) -> None:
        """Reuse requires the same law hash, size, design, and bootstrap."""
        model = three_output_model()
        study = UQStudy(model, seed=5)
        study.sensitivity("a", 64)
        study.sensitivity("b", 32)
        self.assertEqual(study.records[-1].evaluations, 32 * 4)
        study.with_inputs(
            JointInput.independent(
                RandomInput("x", Uniform(0.0, 2.0)), RandomInput("y", Normal(0.0, 1.0))
            )
        )
        study.sensitivity("b", 64)
        self.assertEqual(study.records[-1].evaluations, 64 * 4)

    def test_shared_design_needs_no_budget(self) -> None:
        """The known cost of a shared design is zero, so it is not refused."""
        study = UQStudy(three_output_model(), seed=5, evaluation_budget=64 * 4)
        study.sensitivity("a", 64)
        study.sensitivity("b", 64)
        with self.assertRaises(EvaluationBudgetExceeded):
            study.sensitivity("b", 32)

    def test_same_seed_gives_identical_json_and_pickle_round_trip(self) -> None:
        """The record (provenance aside) depends only on the seed."""

        def run() -> UQStudy:
            study = UQStudy(three_output_model(), seed=11)
            study.sensitivity("b", 32)
            study.sensitivity("a", 32)
            return study

        first, second = run(), run()
        one, two = first.to_dict(), second.to_dict()
        one.pop("provenance")
        two.pop("provenance")
        self.assertEqual(json.dumps(one), json.dumps(two))
        restored = pickle.loads(pickle.dumps(first))
        restored.sensitivity("a", 32)
        self.assertEqual(restored.records[-1].evaluations, 0)


class ExtremesStageTests(unittest.TestCase):
    """The threshold rule of a lower-tail POT stage names the right quantile."""

    def test_lower_tail_rule_names_the_complementary_quantile(self) -> None:
        """A lower-tail threshold is the ``1 - p`` quantile of the QoI."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = UQStudy(capacity_model(), seed=7)
            result = study.extremes("g", size=500, tail="lower")
        summary = study.records[-1].summary
        self.assertEqual(result.sample_size, 500)
        self.assertEqual(summary["threshold_rule"], "q0.05 of g")
        sample = study._output_sample(
            study._root.child("extremes", 0), "g", 500, "mc", 4096, False
        )[0]["g"].values
        self.assertAlmostEqual(summary["threshold"], quantile(sample, 0.05), places=12)
        upper = UQStudy(capacity_model(), seed=7)
        upper.extremes("g", size=500)
        self.assertEqual(upper.records[-1].summary["threshold_rule"], "q0.95 of g")


class FlowBoundaryTests(unittest.TestCase):
    """Planar and radial log-determinants near the invertibility boundary."""

    def planar_oracle(self, raw: float, z: float) -> float:
        """Return ``log(1 + (1 - tanh^2 z) c)`` with ``c = -1 + softplus(raw)``."""
        with localcontext() as context:
            context.prec = DIGITS + 10
            c = -1 + decimal_softplus(raw)
            t = decimal_tanh(z)
            return float(reference_log(1 + (1 - t * t) * c, digits=DIGITS))

    def test_planar_log_determinant_near_the_boundary(self) -> None:
        """``w'u`` below -37 no longer rounds the margin to zero."""
        flow = PlanarFlow(1)
        for raw, z in ((-36.0, 0.0), (-40.0, 0.0), (-40.0, 0.3), (5.0, 0.5)):
            parameters = (raw, 1.0, 0.0)
            x, log_det = flow.forward(parameters, [z])
            self.assertAlmostEqual(
                log_det, self.planar_oracle(raw, z), delta=1e-12 * max(1, abs(log_det))
            )
            if raw > -37.0 or z != 0.0:
                back, inverse_log_det = flow.inverse(parameters, x)
                self.assertAlmostEqual(back[0], z, delta=1e-12)
                self.assertAlmostEqual(inverse_log_det, -log_det, delta=1e-9)

    def test_planar_log_determinant_with_a_large_slope(self) -> None:
        """``1 - tanh^2`` no longer rounds to zero at ``|w'z + b| > 19``."""
        flow = PlanarFlow(1)
        _, log_det = flow.forward((1e10, 1.0, 0.0), [20.0])
        oracle = self.planar_oracle(1e10, 20.0)
        self.assertGreater(oracle, 1e-7)
        # A log-determinant near zero is exact to the rounding of its
        # argument near one (2.2e-16); the defect was an error of 1.7e-7.
        self.assertAlmostEqual(log_det, oracle, delta=1e-15)

    def test_radial_log_determinant_near_the_boundary(self) -> None:
        """``beta`` rounding to ``-alpha`` no longer breaks the determinant."""
        flow = RadialFlow(2)
        for raw in (-36.0, -40.0):
            for z in ((0.0, 0.0), (1e-3, 0.0), (0.5, 0.2)):
                parameters = (0.0, 0.0, 0.0, raw)
                x, log_det = flow.forward(parameters, list(z))
                with localcontext() as context:
                    context.prec = DIGITS + 10
                    alpha = decimal_softplus(0.0)
                    beta = -alpha + decimal_softplus(raw)
                    r = (Decimal(z[0]) ** 2 + Decimal(z[1]) ** 2).sqrt()
                    h = 1 / (alpha + r)
                    oracle = float(
                        reference_log(1 + beta * h, digits=DIGITS)
                        + reference_log(1 + beta * alpha * h * h, digits=DIGITS)
                    )
                self.assertAlmostEqual(log_det, oracle, delta=1e-12 * abs(oracle))
                back, inverse_log_det = flow.inverse(parameters, x)
                self.assertAlmostEqual(inverse_log_det, -log_det, delta=1e-12 * 80)
                for got, want in zip(back, z, strict=True):
                    self.assertAlmostEqual(got, want, delta=1e-12)


class ActNormInitializationTests(unittest.TestCase):
    """Data initialization refuses a coordinate that is constant."""

    def test_constant_coordinate_is_refused(self) -> None:
        """A rounded mean no longer hides a constant column."""
        flow = NormalizingFlow((ActNorm(2),))
        start = flow.initial_parameters(0)
        for value in (0.1, 0.7, 1e-300, 3.0):
            data = [(value, 1.0), (value, 2.0), (value, 3.0)]
            with self.assertRaisesRegex(ValueError, "zero spread"):
                flow.data_initialization(start, data)

    def test_varying_coordinates_are_standardized(self) -> None:
        """Genuine spread still gives mean and log standard deviation."""
        flow = NormalizingFlow((ActNorm(1),))
        parameters = flow.data_initialization(
            flow.initial_parameters(0), [(0.1,), (0.1 + 1e-9,), (0.1 + 2e-9,)]
        )
        self.assertAlmostEqual(parameters[0], 0.1 + 1e-9, delta=1e-15)
        self.assertAlmostEqual(
            parameters[1], math.log(1e-9 * math.sqrt(2.0 / 3.0)), delta=1e-5
        )


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class ReturnLevelTitleTests(unittest.TestCase):
    """The return-level title states the interval as its limits are."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend once."""
        import matplotlib

        matplotlib.use("Agg")

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def title_of(self, lower: float, upper: float, method: str) -> tuple[str, str]:
        """Return the title and subtitle for one shape of interval limits."""
        from probstat.risk import ReturnLevel
        from probstat.visualization.extremes_plots import plot_return_levels

        levels = [
            ReturnLevel(t, 5.0 + math.log(t), math.nan, lower, upper, 0.95, method)
            for t in (2.0, 10.0, 100.0)
        ]
        axes = plot_return_levels(levels, [4.0, 5.0, 6.0], headline_period=100.0)
        texts = " | ".join(item.get_text() for item in axes.texts)
        # The title block wraps long titles; compare the words in order.
        return " ".join(axes.get_title(loc="left").split()), texts

    def test_point_estimates_are_not_called_unbounded(self) -> None:
        """PWM levels (``nan`` limits) have no interval and no band."""
        title, texts = self.title_of(math.nan, math.nan, "point")
        self.assertIn("no confidence interval is available", title)
        self.assertNotIn("unbounded", title)
        self.assertNotIn("band:", texts + title)

    def test_open_lower_limit_is_named(self) -> None:
        """An infinite lower limit is stated as the lower one."""
        title, _ = self.title_of(-math.inf, 20.0, "profile")
        self.assertIn("lower limit is unbounded", title)
        self.assertIn("ends at 20", title)
        title, _ = self.title_of(1.0, math.inf, "profile")
        self.assertIn("upper limit is unbounded", title)
        title, _ = self.title_of(1.0, 30.0, "delta")
        self.assertIn("interval spans 1 to 30", title)


if __name__ == "__main__":
    unittest.main()
