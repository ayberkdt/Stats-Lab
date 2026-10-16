"""Tests for the extremes, information, and design stages of ``UQStudy``."""

from __future__ import annotations

import math
import unittest

from probstat.design import (
    BayesianDesignProblem,
    expected_information_gain,
    greedy_design,
)
from probstat.distributions import LogNormal, Normal, Uniform
from probstat.foundations.random_streams import resolve_stream
from probstat.information import ksg_mutual_information
from probstat.inverse import CalibrationProblem, Experiment, NoiseModel
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import propagate
from probstat.risk import fit_pot
from probstat.statistics import quantile
from probstat.uq import (
    WORKFLOW_GUIDE,
    QuantityOfInterest,
    StepStatus,
    UQWorkflowRecord,
    WorkflowStep,
)
from probstat.workflow import UQStudy

NOISE = (NoiseModel("y", standard_deviation=0.05),)
CANDIDATES = tuple({"t": t} for t in (0.5, 2.0, 4.0, 8.0, 12.0))


def capacity_model() -> ProbabilisticModel:
    """Return ``g = R - S`` with lognormal resistance and normal load."""
    inputs = JointInput.independent(
        RandomInput("R", LogNormal(math.log(5.0), 0.1)),
        RandomInput("S", Normal(3.0, 0.8)),
    )

    def margin(p):
        """Return the safety margin."""
        return {"g": p["R"] - p["S"]}

    return ProbabilisticModel(
        inputs, margin, (QuantityOfInterest("g", "margin"),), name="capacity"
    )


def decay_model() -> ProbabilisticModel:
    """Return an exponential decay ``A exp(-k t)`` with time as a condition."""
    inputs = JointInput.independent(
        RandomInput("A", Uniform(2.0, 8.0)), RandomInput("k", Uniform(0.05, 1.0))
    )

    def decay(p):
        """Return the decayed amplitude at time t."""
        return {"y": p["A"] * math.exp(-p["k"] * p["t"])}

    return ProbabilisticModel(
        inputs, decay, (QuantityOfInterest("y", "amplitude"),), fixed={"t": 0.0}
    )


def decay_experiments() -> tuple[Experiment, ...]:
    """Return two early-time observations of the decay with A=5, k=0.3."""
    return tuple(
        Experiment({"y": 5.0 * math.exp(-0.3 * t) + offset}, {"t": t})
        for t, offset in ((0.0, 0.04), (1.0, -0.03))
    )


def calibrated_study(seed: int) -> UQStudy:
    """Return a decay study calibrated on the two early observations."""
    study = UQStudy(decay_model(), seed=seed)
    study.calibrate(decay_experiments(), NOISE, draws=200, warmup=200, chains=2)
    return study


class ExtremesStageTests(unittest.TestCase):
    """Peaks-over-threshold tail analysis inside the study."""

    def test_reused_propagation_reproduces_direct_fit(self) -> None:
        """The stage fits the latest propagation sample exactly like fit_pot."""
        study = UQStudy(capacity_model(), seed=11)
        study.propagate(4096, design="mc")
        result = study.extremes("g", tail="lower", level=0.0, reuse=True)
        values = study.latest("propagation").result["g"].values
        oriented = tuple(-v for v in values)
        cut = quantile(oriented, 0.95)
        direct = fit_pot(oriented, cut, observations_per_year=1.0)
        self.assertEqual(result.fit.gpd.sigma, direct.gpd.sigma)
        self.assertEqual(result.fit.gpd.xi, direct.gpd.xi)
        self.assertEqual(result.threshold, -cut)
        level = direct.return_level(100.0)
        self.assertEqual(result.return_levels[0].level, -level.level)
        self.assertEqual(result.return_levels[0].lower, -level.upper)
        record = study.latest("extremes")
        self.assertEqual(record.evaluations, 0)
        for key in (
            "threshold",
            "exceedances",
            "xi",
            "sigma",
            "x[T=100]",
            "x[T=1000].low",
            "empirical x[T=100]",
            "P_pot",
            "P_empirical",
            "P_pot/P_empirical",
        ):
            self.assertIn(key, record.summary)
        # The POT tail probability lies inside the empirical Wilson interval.
        assert result.empirical is not None
        interval = result.empirical.confidence_interval
        self.assertTrue(interval.lower <= result.pot_probability <= interval.upper)
        self.assertAlmostEqual(record.summary["P_pot/P_empirical"], 1.0, delta=0.3)

    def test_fresh_sample_uses_stage_stream_and_is_charged(self) -> None:
        """Without a propagation the stage draws from root.child('extremes', 0)."""
        study = UQStudy(capacity_model(), seed=5)
        result = study.extremes("g", size=1024, return_periods=(200.0,))
        stream = resolve_stream(5).child("extremes", 0)
        sample = propagate(capacity_model(), 1024, design="mc", seed=stream)
        values = sample["g"].values
        direct = fit_pot(values, quantile(values, 0.95), observations_per_year=1.0)
        self.assertEqual(result.fit.gpd.xi, direct.gpd.xi)
        self.assertEqual(result.return_levels[0].level, direct.return_level(200).level)
        record = study.latest("extremes")
        self.assertEqual(record.evaluations, 1024)
        self.assertEqual(record.seed, stream.seed)
        self.assertEqual(study.total_evaluations, 1024)

    def test_invalid_requests_raise(self) -> None:
        """Bad quantities, tails, quantiles, and levels are rejected clearly."""
        study = UQStudy(capacity_model(), seed=1)
        study.propagate(512, design="mc")
        with self.assertRaises(ValueError):
            study.extremes("missing")
        with self.assertRaises(ValueError):
            study.extremes("g", threshold_quantile=1.0)
        with self.assertRaises(ValueError):
            study.extremes("g", tail="sideways")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "inside the threshold"):
            study.extremes("g", level=0.0, reuse=True)
        with self.assertRaisesRegex(ValueError, "independent draws"):
            study.extremes("g", design="sobol")
        with self.assertRaisesRegex(ValueError, "either size"):
            study.extremes("g", size=100, reuse=True)
        # No evaluation was spent, so no stage was recorded.
        self.assertEqual([r.stage for r in study.records], ["propagation"])
        # A fresh sample that is spent before the error is still accounted.
        with self.assertRaisesRegex(ValueError, "inside the threshold"):
            study.extremes("g", level=0.0, size=600)
        aborted = study.records[-1]
        self.assertEqual(aborted.status, "aborted")
        self.assertEqual(aborted.evaluations, 600)
        self.assertIn("inside the threshold", aborted.error or "")
        self.assertEqual(study.total_evaluations, 1112)


class InformationStageTests(unittest.TestCase):
    """Mutual-information sensitivity inside the study."""

    def test_reproduces_ksg_on_propagation_prefix(self) -> None:
        """With reuse, the stage equals KSG on the first max_points MC runs."""
        study = UQStudy(capacity_model(), seed=11)
        sample = study.propagate(1024, design="mc")
        result = study.information_sensitivity("g", max_points=300, reuse=True)
        outputs = sample["g"].values[:300]
        for name, value in result.as_mapping().items():
            expected = ksg_mutual_information(
                sample.input_column(name)[:300], outputs, k=3
            )
            self.assertEqual(value, expected)
        self.assertEqual(result.ranking(), ("S", "R"))
        record = study.latest("information")
        self.assertEqual(record.evaluations, 0)
        self.assertEqual(record.summary["ranking"], "S > R")
        self.assertIn("MI[R]", record.summary)
        self.assertEqual(record.summary["unit"], "nats")

    def test_fresh_sample_and_bits(self) -> None:
        """A fresh sample is charged; repeated runs use new streams and units."""
        study = UQStudy(capacity_model(), seed=2)
        nats = study.information_sensitivity("g", max_points=200)
        self.assertEqual(study.latest("information").evaluations, 200)
        stream = resolve_stream(2).child("information", 0)
        sample = propagate(capacity_model(), 200, design="mc", seed=stream)
        expected = ksg_mutual_information(
            sample.input_column("S"), sample["g"].values, k=3
        )
        self.assertEqual(nats.as_mapping()["S"], expected)
        bits = study.information_sensitivity("g", max_points=200, unit="bits")
        self.assertNotEqual(study.records[0].seed, study.records[1].seed)
        second = propagate(
            capacity_model(),
            200,
            design="mc",
            seed=resolve_stream(2).child("information", 1),
        )
        expected_bits = ksg_mutual_information(
            second.input_column("S"), second["g"].values, k=3
        ) / math.log(2.0)
        self.assertAlmostEqual(bits.as_mapping()["S"], expected_bits, places=12)
        self.assertEqual(study.latest("information").summary["unit"], "bits")
        with self.assertRaises(ValueError):
            study.information_sensitivity("g", max_points=1)


class DesignStageTests(unittest.TestCase):
    """Bayesian experimental design for the study's calibration problem."""

    def test_reproduces_direct_greedy_design(self) -> None:
        """The stage equals greedy_design plus a final EIG on its child streams."""
        study = calibrated_study(4)
        before = study.total_evaluations
        result = study.design(CANDIDATES, 2, samples=8)
        problem = BayesianDesignProblem.from_calibration(
            CalibrationProblem(decay_model(), decay_experiments(), NOISE)
        )
        baseline = tuple(item.conditions for item in decay_experiments())
        stream = resolve_stream(4).child("design", 0)
        direct = greedy_design(
            problem,
            CANDIDATES,
            2,
            baseline=baseline,
            samples=8,
            seed=stream.child("greedy"),
        )
        final = expected_information_gain(
            problem,
            baseline + direct.selected,
            samples=8,
            seed=stream.child("final"),
        )
        self.assertEqual(result.selection.indices, direct.indices)
        self.assertEqual(result.selection.information_gains, direct.information_gains)
        self.assertEqual(result.estimate.value, final.value)
        self.assertEqual(result.estimate.standard_error, final.standard_error)
        self.assertEqual(len(result.design), 4)
        record = study.latest("design")
        self.assertEqual(
            record.evaluations,
            direct.model_evaluations + final.model_evaluations,
        )
        self.assertEqual(study.total_evaluations, before + record.evaluations)
        for key in ("condition[1]", "eig[2]", "gain[1]", "eig", "eig_se"):
            self.assertIn(key, record.summary)
        self.assertGreater(record.summary["eig_se"], 0.0)

    def test_late_times_are_informative_about_decay_rate(self) -> None:
        """With only early data, the linearized design picks a later time."""
        study = calibrated_study(4)
        result = study.design(CANDIDATES, 1, method="linearized")
        self.assertGreater(result.selection.selected[0]["t"], 1.0)
        self.assertEqual(result.estimate.standard_error, 0.0)
        self.assertGreater(result.selection.increments[0], 0.0)

    def test_requires_calibration_or_noise(self) -> None:
        """Without calibration the stage needs explicit noise models."""
        study = UQStudy(decay_model(), seed=3)
        with self.assertRaisesRegex(ValueError, "calibration problem"):
            study.design(CANDIDATES, 1)
        result = study.design(CANDIDATES, 1, noise=NOISE, method="linearized")
        self.assertEqual(result.baseline, ())
        self.assertEqual(study.latest("design").summary["baseline_conditions"], 0)


class ChainedStudyTests(unittest.TestCase):
    """New stages chain with the existing ones, report, and map to steps."""

    def test_report_records_and_determinism(self) -> None:
        """A chained study reports every stage and repeats under its seed."""

        def run() -> UQStudy:
            """Run a chained study on the decay model."""
            study = calibrated_study(9)
            study.propagate(512, design="mc")
            study.extremes("y", threshold_quantile=0.9, return_periods=(50.0,))
            study.information_sensitivity("y", max_points=200)
            study.design(CANDIDATES, 1, samples=4)
            return study

        first, second = run(), run()
        report = first.report()
        for heading in ("extremes: POT upper tail", "information:", "design:"):
            self.assertIn(heading, report)
        self.assertEqual(
            [dict(r.summary) for r in first.records],
            [dict(r.summary) for r in second.records],
        )
        self.assertEqual(first.total_evaluations, second.total_evaluations)
        record = UQWorkflowRecord.start(
            "decay", (QuantityOfInterest("y", "amplitude"),)
        ).record_stages(first.records)
        for step in (WorkflowStep.PROPAGATION, WorkflowStep.CALIBRATION):
            self.assertEqual(record.entry(step).status, StepStatus.COMPLETE)
        evidence = " ".join(record.entry(WorkflowStep.CALIBRATION).evidence)
        self.assertIn("'design'", evidence)

    def test_guide_maps_new_stage_kinds(self) -> None:
        """The workflow guide routes the new kinds to propagation or calibration."""
        propagation = WORKFLOW_GUIDE[WorkflowStep.PROPAGATION]
        calibration = WORKFLOW_GUIDE[WorkflowStep.CALIBRATION]
        self.assertIn("extremes", propagation.study_stages)
        self.assertIn("information", propagation.study_stages)
        self.assertIn("design", calibration.study_stages)
        self.assertIn("probstat.workflow.UQStudy.design", calibration.library_calls)


if __name__ == "__main__":
    unittest.main()


def _four_branch_study(seed: int) -> UQStudy:
    """Return a study of the four-branch series system (beta = 3 per branch)."""
    from probstat.distributions import Normal as _Normal
    from probstat.modeling import JointInput as _Joint
    from probstat.modeling import ProbabilisticModel as _Model
    from probstat.modeling import RandomInput as _Input
    from probstat.uq import QuantityOfInterest as _QoI

    root2 = 2.0**0.5

    def branches(p: dict[str, float]) -> dict[str, float]:
        """Return the minimum of the four branch margins."""
        a, b = p["x1"], p["x2"]
        return {
            "g": min(
                3.0 + 0.1 * (a - b) ** 2 - (a + b) / root2,
                3.0 + 0.1 * (a - b) ** 2 + (a + b) / root2,
                (a - b) + 6.0 / root2,
                (b - a) + 6.0 / root2,
            )
        }

    inputs = _Joint.independent(
        _Input("x1", _Normal(0.0, 1.0)), _Input("x2", _Normal(0.0, 1.0))
    )
    model = _Model(inputs, branches, (_QoI("g", "four-branch margin", "1"),))
    return UQStudy(model, seed=seed)


class ReliabilityStageTests(unittest.TestCase):
    """Multiple design points and AK-MCS run as recorded study stages."""

    def test_design_points_stage_finds_all_branches(self) -> None:
        """Four design points at beta = 3; multi-point FORM is 4x single FORM."""
        study = _four_branch_study(7)
        analysis = study.design_points(
            "g", 0.0, max_points=4, importance_samples=1000, start=(0.3, 0.1)
        )
        self.assertEqual(len(analysis.points.forms), 4)
        for beta in analysis.points.betas:
            self.assertAlmostEqual(beta, 3.0, places=3)
        single = analysis.points.forms[0].probability
        self.assertGreater(analysis.system.probability, 3.5 * single)
        assert analysis.mixture is not None
        record = study.latest("design_points")
        self.assertIn("pf[multi-point FORM]", record.summary)
        self.assertGreater(record.evaluations, 0)

    def test_active_reliability_matches_same_population_monte_carlo(self) -> None:
        """AK-MCS classifies the population (almost) like the true model."""
        study = _four_branch_study(3)
        result = study.active_reliability(
            "g", 0.0, population=2000, target_cov=0.6, compare_with_monte_carlo=True
        )
        # U >= 2 leaves each uncertain point misclassified with probability
        # at most Phi(-2), so a handful of errors is allowed, not zero.
        self.assertLessEqual(result.misclassified, 0.005 * result.population)
        self.assertLessEqual(
            abs(result.probability - result.reference_probability),
            result.misclassified / result.population + 1e-15,
        )
        self.assertLess(result.evaluations, 100)
        summary = study.latest("active_reliability").summary
        self.assertEqual(summary["misclassified"], float(result.misclassified))
        kinds = [record.stage for record in study.records]
        self.assertEqual(kinds, ["active_reliability"])
