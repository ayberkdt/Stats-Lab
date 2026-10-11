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


if __name__ == "__main__":
    unittest.main()
