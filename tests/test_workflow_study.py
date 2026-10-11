"""Tests for the integrated UQStudy workflow."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest

from probstat.distributions import LogNormal, Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.risk import step_loss
from probstat.uq import QuantityOfInterest
from probstat.workflow import UQStudy


def capacity_model() -> ProbabilisticModel:
    """Return R - S with lognormal resistance and normal load."""
    inputs = JointInput.independent(
        RandomInput("R", LogNormal(math.log(5.0), 0.1)),
        RandomInput("S", Normal(3.0, 0.8)),
    )

    def model(p):
        """Return the safety margin."""
        return {"g": p["R"] - p["S"]}

    return ProbabilisticModel(
        inputs, model, (QuantityOfInterest("g", "margin"),), name="capacity"
    )


class StudyTests(unittest.TestCase):
    """End-to-end behaviour of the study object."""

    def test_stages_share_one_model_and_report(self) -> None:
        """Every stage is recorded with its cost and appears in the report."""
        study = UQStudy(capacity_model(), seed=11)
        propagation = study.propagate(1024, design="sobol")
        sobol = study.sensitivity("g", 256, bootstrap=40)
        comparison = study.reliability(
            "g", 0.0, methods=("form", "mc"), monte_carlo_samples=4000
        )
        risk = study.risk("g", step_loss(-0.0, 100.0), 4000)
        self.assertGreater(propagation["g"].mean, 1.5)
        self.assertEqual(sobol.ranking()[0], "S")
        form_pf = comparison.row("FORM").probability
        mc_pf = comparison.row("MC").probability
        self.assertLess(abs(form_pf - mc_pf), 0.01)
        self.assertGreaterEqual(risk.expected_loss.estimate, 0.0)
        stages = [record.stage for record in study.records]
        self.assertEqual(stages, ["propagation", "sensitivity", "reliability", "risk"])
        self.assertEqual(
            study.total_evaluations, sum(r.evaluations for r in study.records)
        )
        report = study.report()
        for stage in stages:
            self.assertIn(stage, report)
        self.assertIn("Forward-model evaluations", report)

    def test_same_seed_reproduces_study(self) -> None:
        """Two studies with the same root seed give identical results."""
        first = UQStudy(capacity_model(), seed=5).propagate(128, design="lhs")
        second = UQStudy(capacity_model(), seed=5).propagate(128, design="lhs")
        self.assertEqual(first["g"].values, second["g"].values)

    def test_repeated_stage_uses_new_stream(self) -> None:
        """Running a stage twice draws from distinct child streams."""
        study = UQStudy(capacity_model(), seed=3)
        study.propagate(64, design="mc")
        study.propagate(64, design="mc")
        seeds = {record.seed for record in study.records}
        self.assertEqual(len(seeds), 2)
        with self.assertRaises(KeyError):
            study.latest("risk")

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None, "matplotlib is optional"
    )
    def test_study_figures_are_written(self) -> None:
        """Each supported stage produces a saved figure."""
        import matplotlib

        matplotlib.use("Agg")
        from probstat.visualization.study import save_study_figures

        study = UQStudy(capacity_model(), seed=2)
        study.propagate(256)
        study.screening("g", 6)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_study_figures(study, directory)
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))


class CalibrationAndEmulationTests(unittest.TestCase):
    """Backward modeling and surrogates inside the same study."""

    def test_calibrate_then_predict_closes_the_loop(self) -> None:
        """Calibration recovers a decay rate and prediction uses the posterior."""
        from probstat.distributions import Uniform
        from probstat.inverse import Experiment, NoiseModel

        inputs = JointInput.independent(
            RandomInput("A", Uniform(2.0, 8.0)), RandomInput("k", Uniform(0.05, 1.0))
        )

        def decay(p):
            """Return the decayed amplitude at time t."""
            return {"y": p["A"] * math.exp(-p["k"] * p["t"])}

        model = ProbabilisticModel(
            inputs, decay, (QuantityOfInterest("y", "amplitude"),), fixed={"t": 0.0}
        )
        times = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)
        noise_free = (5.0 * math.exp(-0.3 * t) for t in times)
        offsets = (0.04, -0.03, 0.05, -0.02, 0.01, -0.04)
        experiments = tuple(
            Experiment({"y": value + offset}, {"t": t})
            for value, offset, t in zip(noise_free, offsets, times, strict=True)
        )
        study = UQStudy(model, seed=4)
        posterior = study.calibrate(
            experiments,
            (NoiseModel("y", standard_deviation=0.05),),
            draws=600,
            warmup=600,
            chains=2,
        )
        means = posterior.mean()
        self.assertAlmostEqual(means["k"], 0.3, delta=0.03)
        self.assertAlmostEqual(means["A"], 5.0, delta=0.2)
        prediction = study.predict(conditions=({"t": 10.0},), max_draws=200)
        expected = 5.0 * math.exp(-3.0)
        self.assertAlmostEqual(prediction.predictions[0].mean, expected, delta=0.06)
        self.assertEqual(
            [record.stage for record in study.records], ["calibration", "prediction"]
        )

    def test_emulated_study_matches_original_reliability(self) -> None:
        """A PCE emulator reproduces the failure probability far more cheaply."""
        original = UQStudy(capacity_model(), seed=8)
        emulated = original.emulate("g", method="pce", degree=3)
        emulator_record = original.latest("emulation")
        self.assertLess(emulator_record.summary["loo_error"], 1e-3)
        reference = original.reliability(
            "g", 0.0, methods=("mc",), monte_carlo_samples=20_000
        )
        cheap = emulated.reliability(
            "g", 0.0, methods=("mc",), monte_carlo_samples=20_000
        )
        self.assertAlmostEqual(
            cheap.row("MC").probability, reference.row("MC").probability, delta=0.004
        )
        self.assertLess(emulator_record.evaluations, 200)

    def test_emulator_on_bounded_inputs_survives_large_designs(self) -> None:
        """Uniform inputs get a Legendre basis, so far-tail design points stay sane."""
        from probstat.distributions import Uniform

        inputs = JointInput.independent(
            RandomInput("a", Uniform(-1.0, 1.0)), RandomInput("b", Uniform(-1.0, 1.0))
        )

        def model(p):
            """Return a smooth bounded response."""
            return {"y": math.sin(3.0 * p["a"]) + p["b"] ** 2}

        study = UQStudy(
            ProbabilisticModel(inputs, model, (QuantityOfInterest("y", "y"),)), seed=1
        )
        emulated = study.emulate("y", method="pce", degree=7, size=120)
        self.assertEqual(study.latest("emulation").summary["basis"], "legendre")
        mean = emulated.propagate(8192, design="sobol")["y"].mean
        self.assertAlmostEqual(mean, 1.0 / 3.0, delta=0.01)

    def test_validate_stage_gives_scoped_verdicts(self) -> None:
        """Unbiased data are consistent; biased data are not."""
        import random

        from probstat.validation import ValidationExperiment

        inputs = JointInput.independent(RandomInput("k", Normal(2.0, 0.1)))

        def spring(p):
            """Return the extension of a spring under a load."""
            return {"x": p["k"] * p["load"]}

        model = ProbabilisticModel(
            inputs, spring, (QuantityOfInterest("x", "extension"),), fixed={"load": 1.0}
        )
        rng = random.Random(3)

        def campaign(bias):
            """Return three experiments drawn from the model plus a bias."""
            return [
                ValidationExperiment(
                    f"load-{load}",
                    [
                        rng.gauss(2.0, 0.1) * load + rng.gauss(0.0, 0.05) + bias
                        for _ in range(8)
                    ],
                    {"load": load},
                    measurement_uncertainty=0.05,
                    provenance="synthetic",
                )
                for load in (1.0, 2.0, 3.0)
            ]

        study = UQStudy(model, seed=6)
        good = study.validate("x", campaign(0.0), tolerance=0.3, sample_size=800)
        bad = study.validate("x", campaign(1.0), tolerance=0.3, sample_size=800)
        self.assertTrue(good.consistent)
        self.assertFalse(bad.consistent)
        self.assertNotIn("validated", good.summary().lower())
        self.assertEqual(study.latest("validation").summary["consistent"], "False")


if __name__ == "__main__":
    unittest.main()
