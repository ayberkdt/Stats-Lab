"""Tests for the Kennedy-O'Hagan, scoring, and figure integration of UQStudy."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import tempfile
import unittest

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.foundations.random_streams import resolve_stream
from probstat.inverse import (
    CalibrationProblem,
    DiscrepancyCalibration,
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    NoiseModel,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import (
    WORKFLOW_GUIDE,
    QuantityOfInterest,
    StepStatus,
    UQWorkflowRecord,
    WorkflowStep,
)
from probstat.validation import (
    ValidationExperiment,
    assess_validation,
    ensemble_pit,
    pit_coverage,
)
from probstat.workflow import DiscrepancyStudyPrediction, UQStudy

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
DATA_X = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
NOISE = 0.03


def decay(p):
    """Return ``4 exp(-k x)``."""
    return {"y": 4.0 * math.exp(-p["k"] * p["x"])}


def decay_model() -> ProbabilisticModel:
    """Return the decay simulator with a lognormal prior on ``k``."""
    return ProbabilisticModel(
        JointInput.independent(RandomInput("k", LogNormal(math.log(0.5), 0.5))),
        decay,
        (QuantityOfInterest("y", "y"),),
        fixed={"x": 0.0},
        name="decay",
    )


def biased_data() -> tuple[Experiment, ...]:
    """Return observations of the simulator at k = 0.4 plus a smooth bias."""
    return tuple(
        Experiment({"y": 4.0 * math.exp(-0.4 * x) + 0.25 * math.sin(2.5 * x)}, {"x": x})
        for x in DATA_X
    )


def process() -> DiscrepancyProcess:
    """Return a squared-exponential discrepancy with hyperparameter priors."""
    return DiscrepancyProcess(
        ("x",),
        standard_deviation_prior=LogNormal(math.log(0.2), 0.75),
        length_scale_priors=(LogNormal(math.log(0.5), 0.5),),
    )


def koh_study(seed: int) -> UQStudy:
    """Return a decay study after a short Kennedy-O'Hagan calibration."""
    study = UQStudy(decay_model(), seed=seed)
    study.calibrate_with_discrepancy(
        biased_data(),
        "y",
        KnownNoise(NOISE),
        process(),
        draws=200,
        warmup=200,
        chains=2,
        grid=11,
        grid_draws=50,
    )
    return study


class DefaultCalibrationTests(unittest.TestCase):
    """The plain calibrate stage is unchanged by the new options."""

    def test_calibrate_reproduces_direct_problem(self) -> None:
        """Calibrate equals CalibrationProblem.sample_posterior on its stream."""
        noise = (NoiseModel("y", standard_deviation=NOISE),)
        study = UQStudy(decay_model(), seed=21)
        posterior = study.calibrate(
            biased_data(), noise, draws=150, warmup=150, chains=2
        )
        direct = CalibrationProblem(
            decay_model(), biased_data(), noise
        ).sample_posterior(
            draws=150,
            warmup=150,
            chains=2,
            seed=resolve_stream(21).child("calibration", 0),
        )
        self.assertEqual(posterior.mcmc.pooled(), direct.mcmc.pooled())
        self.assertEqual(
            set(study.latest("calibration").summary),
            {"experiments", "mean[k]", "sd[k]", "rhat[k]"},
        )
        prediction = study.predict(conditions=({"x": 1.0},), max_draws=50)
        self.assertNotIsInstance(prediction, DiscrepancyStudyPrediction)


class DiscrepancyStageTests(unittest.TestCase):
    """Kennedy-O'Hagan calibration as a recorded study stage."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one short discrepancy calibration."""
        cls.study = koh_study(3)
        cls.analysis = cls.study.latest("discrepancy_calibration").result

    def test_reproduces_direct_calibration_on_child_stream(self) -> None:
        """The chains equal DiscrepancyCalibration on root.child(kind, 0)."""
        direct = DiscrepancyCalibration(
            decay_model(), biased_data(), "y", KnownNoise(NOISE), process()
        )
        posterior = direct.sample_posterior(
            draws=200,
            warmup=200,
            chains=2,
            seed=resolve_stream(3).child("discrepancy_calibration", 0),
        )
        self.assertEqual(
            self.analysis.posterior.vector.pooled(), posterior.vector.pooled()
        )
        report = direct.identifiability(posterior)
        self.assertEqual(
            self.analysis.identifiability.max_abs_cross_correlation,
            report.max_abs_cross_correlation,
        )
        record = self.study.latest("discrepancy_calibration")
        summary = record.summary
        self.assertEqual(
            summary["max |rho(theta, psi)|"], report.max_abs_cross_correlation
        )
        self.assertEqual(summary["sd ratio[k]"], report.standard_deviation_ratios["k"])
        self.assertIn("identifiability", summary)
        self.assertIn("mean[sigma_delta]", summary)
        self.assertEqual(
            record.seed, resolve_stream(3).child("discrepancy_calibration", 0).seed
        )
        self.assertGreater(record.evaluations, 0)
        self.assertEqual(self.analysis.grid_input, "x")
        self.assertEqual(len(self.analysis.grid), 11)
        self.assertEqual(self.analysis.grid[0].condition["x"], 0.0)
        self.assertEqual(self.analysis.grid[-1].condition["x"], 4.0)

    def test_predict_separates_simulator_and_corrected(self) -> None:
        """Predict after KOH reports both interval kinds and matches the problem."""
        study = koh_study(3)
        result = study.predict(conditions=({"x": 1.2}, {"x": 4.5}), include_noise=True)
        assert isinstance(result, DiscrepancyStudyPrediction)
        analysis = study.latest("discrepancy_calibration").result
        direct = analysis.calibration.predict(
            analysis.posterior,
            ({"x": 1.2}, {"x": 4.5}),
            max_draws=1000,
            observation_standard_deviations=(NOISE, NOISE),
        )
        self.assertEqual(result.predictions, direct)
        summary = study.latest("prediction").summary
        low, high = direct[1].interval(0.95, kind="corrected")
        self.assertEqual(summary["corrected q_low[y#1]"], low)
        self.assertEqual(summary["corrected q_high[y#1]"], high)
        for key in (
            "simulator mean[y#0]",
            "simulator q_low[y#0]",
            "delta mean[y#0]",
            "observed q_high[y#1]",
        ):
            self.assertIn(key, summary)
        self.assertEqual(study.latest("prediction").evaluations, result.evaluations)
        with self.assertRaises(ValueError):
            study.predict(("other",))

    def test_design_after_discrepancy_needs_noise_and_uses_its_data(self) -> None:
        """Design raises without noise and uses the KOH experiments as baseline."""
        study = koh_study(5)
        candidates = ({"x": 5.0}, {"x": 6.0})
        with self.assertRaisesRegex(ValueError, "calibrate_with_discrepancy"):
            study.design(candidates, 1)
        result = study.design(
            candidates,
            1,
            noise=(NoiseModel("y", standard_deviation=NOISE),),
            method="linearized",
        )
        self.assertEqual(len(result.baseline), len(DATA_X))
        self.assertEqual(tuple(dict(c) for c in result.candidates), candidates)

    def test_invalid_arguments_raise(self) -> None:
        """Bad levels and grid sizes are rejected before sampling."""
        study = UQStudy(decay_model(), seed=1)
        with self.assertRaises(ValueError):
            study.calibrate_with_discrepancy(
                biased_data(), "y", KnownNoise(NOISE), process(), level=1.0
            )
        with self.assertRaises(ValueError):
            study.calibrate_with_discrepancy(
                biased_data(), "y", KnownNoise(NOISE), process(), grid=-1
            )
        self.assertEqual(study.records, ())

    def test_stage_kind_maps_to_calibration(self) -> None:
        """record_stages routes the new kind to the calibration step."""
        guide = WORKFLOW_GUIDE[WorkflowStep.CALIBRATION]
        self.assertIn("discrepancy_calibration", guide.study_stages)
        record = UQWorkflowRecord.start(
            "decay", (QuantityOfInterest("y", "y"),)
        ).record_stages(self.study.records)
        self.assertEqual(
            record.entry(WorkflowStep.CALIBRATION).status, StepStatus.COMPLETE
        )


class ValidationScoringTests(unittest.TestCase):
    """CRPS, PIT, and coverage evidence in the validate summary."""

    def test_summary_contains_scores_and_keeps_the_verdict(self) -> None:
        """Scores match the scoring modules; the assessment equals the direct one."""
        inputs = JointInput.independent(
            RandomInput("a", Normal(1.0, 0.1)), RandomInput("b", Uniform(0.5, 1.5))
        )

        def line(p):
            """Return ``a + b t``."""
            return {"y": p["a"] + p["b"] * p["t"]}

        model = ProbabilisticModel(
            inputs, line, (QuantityOfInterest("y", "y"),), fixed={"t": 0.0}
        )
        experiments = tuple(
            ValidationExperiment(
                f"t={t:g}",
                tuple(1.0 + t + 0.02 * k for k in (-1, 0, 1)),
                {"t": t},
                measurement_uncertainty=0.05,
                provenance="synthetic",
            )
            for t in (0.5, 1.0, 2.0)
        )
        study = UQStudy(model, seed=8)
        result = study.validate("y", experiments, tolerance=0.5, sample_size=400)
        stream = resolve_stream(8).child("validation", 0)
        direct = assess_validation(
            model, "y", experiments, tolerance=0.5, sample_size=400, seed=stream
        )
        self.assertEqual(result.consistent, direct.consistent)
        self.assertEqual(result.u_pooling.area, direct.u_pooling.area)
        summary = study.latest("validation").summary
        for index, item in enumerate(result.experiments, start=1):
            self.assertEqual(
                summary[f"crps[{index}:{item.experiment.name}]"], item.crps
            )
        ensembles = [
            item.observable_predictions
            for item in result.experiments
            for _ in item.experiment.observations
        ]
        observations = [
            y for item in result.experiments for y in item.experiment.observations
        ]
        pit = ensemble_pit(ensembles, observations, seed=stream.child("pit"))
        self.assertAlmostEqual(summary["pit.mean"], sum(pit) / len(pit), places=14)
        coverage = pit_coverage(pit, (0.5, 0.9))
        self.assertEqual(summary["coverage[50%]"], coverage[0].rate)
        self.assertEqual(summary["coverage[90%]"], coverage[1].rate)
        self.assertIn(f"{coverage[1].covered}/9", summary["coverage[90%].interval"])
        self.assertGreater(summary["crps.mean"], 0.0)
        self.assertEqual(summary["consistent"], str(direct.consistent))


class StoredFigureDataTests(unittest.TestCase):
    """Tail exceedances and design candidates are kept on the results."""

    def test_tail_keeps_exceedances_in_qoi_units(self) -> None:
        """Exceedances match the rate count and lie beyond the threshold."""
        model = dataclasses.replace(decay_model(), fixed={"x": 1.0})
        study = UQStudy(model, seed=4)
        study.propagate(1024, design="mc")
        for tail in ("upper", "lower"):
            result = study.extremes("y", tail=tail, threshold_quantile=0.9)
            self.assertEqual(len(result.exceedances), result.fit.rate.exceedances)
            if tail == "upper":
                self.assertTrue(all(v > result.threshold for v in result.exceedances))
                self.assertEqual(list(result.exceedances), sorted(result.exceedances))
            else:
                self.assertTrue(all(v < result.threshold for v in result.exceedances))
                self.assertEqual(
                    list(result.exceedances), sorted(result.exceedances, reverse=True)
                )


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class NewStageFigureTests(unittest.TestCase):
    """Figures of the design-point, AK-MCS, calibration, and KOH stages."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one study with every stage that gained a figure."""
        import matplotlib

        matplotlib.use("Agg")

        def deviation(p):
            """Return ``|x1 + x2|``, failing on both sides of zero."""
            return {
                "d": abs(p["x1"] + p["x2"]),
                "y": decay({"k": p["k"], "x": p["x"]})["y"],
            }

        inputs = JointInput.independent(
            RandomInput("x1", Normal(0.0, 1.0)),
            RandomInput("x2", Normal(0.0, 1.0)),
            RandomInput("k", LogNormal(math.log(0.5), 0.5)),
        )
        model = ProbabilisticModel(
            inputs,
            deviation,
            (QuantityOfInterest("d", "d"), QuantityOfInterest("y", "y")),
            fixed={"x": 0.0},
            name="figures",
        )
        cls.study = UQStudy(model, seed=13)
        cls.points = cls.study.design_points(
            "d", 4.0, failure="above", max_points=2, importance_samples=0,
            start=(0.4, 0.3, 0.0),
        )  # fmt: skip
        cls.active = cls.study.active_reliability(
            "d", 3.0, failure="above", population=1500, target_cov=0.6
        )
        cls.koh = koh_study(3)
        cls.plain = UQStudy(decay_model(), seed=2)
        cls.plain.calibrate(
            biased_data(),
            (NoiseModel("y", standard_deviation=NOISE),),
            draws=150,
            warmup=150,
            chains=2,
        )

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_design_point_plot_titles_the_system_probability(self) -> None:
        """The title quotes the multi-point FORM probability and the ratio."""
        from probstat.visualization.stage_plots import plot_design_points

        self.assertEqual(len(self.points.points.forms), 2)
        axes = plot_design_points(self.points, input_names=("x1", "x2", "k"))
        title = axes.get_title(loc="left")
        self.assertIn(f"{self.points.system.probability:.2e}", title)
        self.assertTrue(title.startswith("2 design points"))
        self.assertTrue(any("plane of" in t.get_text() for t in axes.texts))
        with self.assertRaises(ValueError):
            plot_design_points(self.points, input_names=("x1",))

    def test_active_learning_plot_has_two_panels(self) -> None:
        """Estimate and stopping statistic share the horizontal axis."""
        from probstat.visualization.stage_plots import plot_active_learning

        figure = plot_active_learning(self.active)
        self.assertEqual(len(figure.axes), 2)
        self.assertIn(
            f"{self.active.probability:.2e}", figure.axes[0].get_title(loc="left")
        )

    def test_posterior_plot_names_the_best_informed_parameter(self) -> None:
        """The title states the posterior/prior spread ratio."""
        from probstat.visualization.stage_plots import plot_posterior_marginals

        posterior = self.plain.latest("calibration").result
        figure = plot_posterior_marginals(posterior, prior=self.plain.model.inputs)
        self.assertIn("The data narrow k to", figure.axes[0].get_title(loc="left"))
        alone = plot_posterior_marginals(posterior)
        self.assertIn("posterior mean of k", alone.axes[0].get_title(loc="left"))
        with self.assertRaises(ValueError):
            plot_posterior_marginals(posterior, max_panels=0)

    def test_study_html_renders_every_new_stage(self) -> None:
        """study_html and save_study_figures draw the new stage figures."""
        from probstat.visualization.study import save_study_figures, study_html

        reliability = study_html(self.study)
        self.assertEqual(reliability.count("data:image/png;base64,"), 4)
        self.assertEqual(study_html(self.koh).count("data:image/png;base64,"), 4)
        self.assertEqual(study_html(self.plain).count("data:image/png;base64,"), 2)
        with tempfile.TemporaryDirectory() as directory:
            names = [p.name for p in save_study_figures(self.study, directory)]
            names += [p.name for p in save_study_figures(self.koh, directory)]
            names += [p.name for p in save_study_figures(self.plain, directory)]
        for fragment in (
            "design_points-u-space",
            "active_reliability-learning",
            "discrepancy_calibration-predictions",
            "discrepancy_calibration-discrepancy",
            "calibration-posterior",
        ):
            self.assertTrue(any(fragment in name for name in names), fragment)


if __name__ == "__main__":
    unittest.main()
