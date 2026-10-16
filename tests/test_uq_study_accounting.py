"""UQStudy accounting, chaining, persistence, and stage correctness (ADR-0058).

Oracles: the conjugate normal posterior (closed form) for posterior
adoption; exact Normal moments and quantiles; stage-order independence by
direct recomputation on the documented streams.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import pickle
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from probstat.distributions import LogNormal, Normal
from probstat.foundations.random_streams import resolve_stream
from probstat.information import ksg_mutual_information
from probstat.inverse import Experiment, NoiseModel
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import propagate
from probstat.uq import (
    EvaluationBudgetExceeded,
    QuantityOfInterest,
    StepStatus,
    UQWorkflowRecord,
    WorkflowStep,
)
from probstat.workflow import (
    EmpiricalMarginal,
    StageRecord,
    UQStudy,
    input_law_hash,
    posterior_joint_input,
)

QOIS = (QuantityOfInterest("g", "margin"),)


def margin(point):
    """Return the safety margin ``R - S``."""
    return {"g": point["R"] - point["S"]}


def brittle_margin(point):
    """Return ``R - S`` but fail when the load exceeds 4.5."""
    if point["S"] > 4.5:
        raise ArithmeticError("load step diverged")
    return {"g": point["R"] - point["S"]}


def capacity_model(function=margin, **options) -> ProbabilisticModel:
    """Return ``g = R - S`` with lognormal resistance and normal load."""
    inputs = JointInput.independent(
        RandomInput("R", LogNormal(math.log(5.0), 0.1)),
        RandomInput("S", Normal(3.0, 0.8)),
    )
    return ProbabilisticModel(inputs, function, QOIS, name="capacity", **options)


def line(point):
    """Return ``y = theta`` and ``z = theta + load``."""
    return {"y": point["theta"], "z": point["theta"] + point["load"]}


def line_model() -> ProbabilisticModel:
    """Return the calibration model with prior ``theta ~ N(1, 0.5)``."""
    inputs = JointInput.independent(
        RandomInput("theta", Normal(1.0, 0.5)), RandomInput("load", Normal(0.0, 1.0))
    )
    return ProbabilisticModel(
        inputs,
        line,
        (QuantityOfInterest("y", "observed"), QuantityOfInterest("z", "predicted")),
        name="line",
    )


OBSERVED = (1.98, 2.02, 1.99, 2.01)
EXPERIMENTS = tuple(Experiment({"y": value}, {}) for value in OBSERVED)
NOISE = (NoiseModel("y", standard_deviation=0.1),)
#: Conjugate normal posterior of theta: precision 1/0.25 + 4/0.01 = 404.
POSTERIOR_PRECISION = 1.0 / 0.25 + len(OBSERVED) / 0.01
POSTERIOR_MEAN = (1.0 / 0.25 + sum(OBSERVED) / 0.01) / POSTERIOR_PRECISION
POSTERIOR_SD = 1.0 / math.sqrt(POSTERIOR_PRECISION)


class FailureAccountingTests(unittest.TestCase):
    """0.1, 0.3, P1.1: failures in stages are reported and charged."""

    def test_skip_policy_stage_reports_failure_fraction(self) -> None:
        """The propagation stage keeps good runs and reports the failures."""
        study = UQStudy(capacity_model(brittle_margin, on_failure="skip"), seed=4)
        with self.assertWarns(UserWarning):
            result = study.propagate(2000, design="mc")
        record = study.latest("propagation")
        self.assertGreater(record.failed_evaluations, 0)
        self.assertEqual(record.failed_evaluations, len(result.failures))
        self.assertEqual(record.evaluations + record.failed_evaluations, 2000)
        self.assertAlmostEqual(
            record.summary["failure fraction"], record.failed_evaluations / 2000
        )
        self.assertTrue(any("biased" in note for note in record.warnings))
        report = study.report()
        self.assertIn("**Failed evaluations:", report)
        self.assertIn("> **Warning:**", report)
        self.assertEqual(study.total_failed_evaluations, record.failed_evaluations)

    def test_stage_that_raises_is_recorded_with_its_cost(self) -> None:
        """0.3: evaluations of an aborted stage stay in the study accounts."""
        model = capacity_model(brittle_margin)
        study = UQStudy(model, seed=4)
        study.propagate(8, design="mc")
        with self.assertRaises(ArithmeticError):
            study.propagate(2000, design="mc")
        aborted = study.records[-1]
        self.assertEqual(aborted.status, "aborted")
        self.assertIsNone(aborted.result)
        self.assertEqual(aborted.failed_evaluations, 1)
        self.assertGreater(aborted.evaluations, 0)
        self.assertEqual(study.total_evaluations, model.evaluations)
        self.assertIn("ArithmeticError: load step diverged", aborted.error or "")
        self.assertIn("status: **aborted**", study.report())
        # latest() returns the last *complete* record; the stream advanced.
        self.assertEqual(study.latest("propagation"), study.records[0])
        self.assertNotEqual(study.stream("propagation").seed, aborted.seed)
        evidence = UQWorkflowRecord.start("c", QOIS).record_stages(study.records)
        self.assertEqual(len(evidence.entry(WorkflowStep.PROPAGATION).evidence), 1)

    def test_invalid_arguments_leave_no_record(self) -> None:
        """A stage rejected before any run is not recorded."""
        study = UQStudy(capacity_model(), seed=1)
        with self.assertRaises(ValueError):
            study.propagate(1)
        self.assertEqual(study.records, ())


class BudgetTests(unittest.TestCase):
    """Optional evaluation budget."""

    def test_known_cost_is_refused_before_running(self) -> None:
        """Stages with an up-front cost above the remainder run nothing."""
        study = UQStudy(capacity_model(), seed=1, evaluation_budget=1000)
        with self.assertRaisesRegex(EvaluationBudgetExceeded, "nothing was run"):
            study.propagate(1001, design="mc")
        self.assertEqual(study.model.evaluations, 0)
        self.assertEqual(study.records, ())
        study.propagate(600, design="mc")
        self.assertEqual(study.remaining_budget, 400)
        with self.assertRaises(EvaluationBudgetExceeded):
            study.sensitivity("g", 128)  # 128 (d + 2) = 512 > 400
        with self.assertRaises(EvaluationBudgetExceeded):
            study.reliability("g", 0.0, methods=("mc",), monte_carlo_samples=500)
        self.assertEqual(len(study.records), 1)

    def test_unknown_cost_is_stopped_and_recorded(self) -> None:
        """A sequential stage is stopped at the limit and recorded."""
        study = UQStudy(capacity_model(), seed=1, evaluation_budget=300)
        with self.assertRaises(EvaluationBudgetExceeded):
            study.propagate_until("g", target_relative_error=1e-4, batch=100)
        record = study.records[-1]
        self.assertEqual(record.status, "budget-exceeded")
        self.assertEqual(record.evaluations, 300)
        self.assertEqual(study.remaining_budget, 0)
        self.assertEqual(study.total_evaluations, 300)
        self.assertIn("Evaluation budget: 300 (0 remaining)", study.report())


class ChainingTests(unittest.TestCase):
    """0.4, P1.2: the posterior becomes the input law of later stages."""

    @classmethod
    def setUpClass(cls) -> None:
        """Calibrate once (2 x 1000 draws) and adopt the posterior of theta."""
        cls.study = UQStudy(line_model(), seed=3)
        cls.prior_hash = cls.study.input_law[1]
        cls.study.calibrate(EXPERIMENTS, NOISE, draws=1000, warmup=500, chains=2)
        cls.law = cls.study.adopt_posterior(["theta"])
        cls.forward = cls.study.propagate(4096, design="sobol")

    def test_later_stages_use_the_posterior(self) -> None:
        """Propagation after adoption matches the conjugate posterior."""
        y = self.forward["y"]
        self.assertLess(abs(y.mean - POSTERIOR_MEAN), 0.01)
        self.assertLess(abs(y.standard_deviation / POSTERIOR_SD - 1.0), 0.2)
        # The non-calibrated input keeps its prior law: z = theta + N(0, 1).
        z = self.forward["z"]
        self.assertLess(abs(z.mean - POSTERIOR_MEAN), 0.05)
        self.assertLess(abs(z.standard_deviation - 1.0), 0.05)
        self.assertEqual(self.law.inputs[1], line_model().inputs.inputs[1])

    def test_records_name_their_input_law(self) -> None:
        """Each record states the prior or posterior label and hash."""
        calibration, change, propagation = self.study.records
        self.assertEqual(calibration.input_law, "prior")
        self.assertEqual(calibration.input_law_hash, self.prior_hash)
        self.assertEqual(change.stage, "input_law")
        self.assertTrue(propagation.input_law.startswith("posterior #1"))
        self.assertEqual(propagation.input_law_hash, input_law_hash(self.law))
        self.assertNotEqual(propagation.input_law_hash, self.prior_hash)
        self.assertEqual(calibration.settings["draws"], 1000)
        self.assertEqual(calibration.settings["chains"], 2)
        report = self.study.report()
        self.assertIn("Input law: posterior #1 (gaussian_copula)", report)
        self.assertIn(f"input law: prior ({self.prior_hash})", report)
        evidence = UQWorkflowRecord.start("line", line_model().quantities)
        evidence = evidence.record_stages(self.study.records)
        self.assertEqual(
            evidence.entry(WorkflowStep.INPUT_CHARACTERIZATION).status,
            StepStatus.COMPLETE,
        )

    def test_empirical_method_and_with_inputs(self) -> None:
        """The empirical route and an explicit law change are both recorded."""
        study = UQStudy(line_model(), seed=3)
        study.calibrate(EXPERIMENTS, NOISE, draws=500, warmup=300, chains=2)
        law = study.adopt_posterior(method="empirical")
        theta = law.inputs[0].distribution
        self.assertIsInstance(theta, EmpiricalMarginal)
        self.assertLess(abs(theta.mean - POSTERIOR_MEAN), 0.02)
        study.with_inputs(line_model().inputs, label="prior again")
        expected = ("prior again", input_law_hash(line_model().inputs))
        self.assertEqual(study.input_law, expected)
        with self.assertRaises(ValueError):
            study.with_inputs(JointInput.independent(RandomInput("x", Normal(0, 1))))
        with self.assertRaises(RuntimeError):
            UQStudy(line_model()).adopt_posterior()

    def test_posterior_joint_input_keeps_dependence(self) -> None:
        """Correlated draws give a Gaussian-copula law with that correlation."""
        prior = line_model().inputs
        sample = JointInput(prior.inputs, ((1.0, 0.8), (0.8, 1.0))).sample(1500, seed=2)
        draws = [{"theta": a, "load": b} for a, b in sample]
        law = posterior_joint_input(prior, draws, candidates=("Normal",))
        assert law.correlation is not None
        self.assertLess(abs(law.correlation[0][1] - 0.8), 0.05)
        self.assertEqual(type(law.inputs[0].distribution).__name__, "Normal")


class EmpiricalMarginalTests(unittest.TestCase):
    """EQ-WF-EMPIRICAL-001 against hand-computed values."""

    def test_moments_cdf_and_quantile(self) -> None:
        """Points (0, 1, 3): each gap holds probability 1/2 uniformly."""
        law = EmpiricalMarginal((3.0, 0.0, 1.0, 1.0))
        self.assertEqual(law.points, (0.0, 1.0, 3.0))
        self.assertAlmostEqual(law.mean, 0.5 * 0.5 + 0.5 * 2.0)
        second = 0.5 * (1.0 / 3.0) + 0.5 * (1.0 + 3.0 + 9.0) / 3.0
        self.assertAlmostEqual(law.variance, second - 1.25**2)
        self.assertEqual(law.cdf(-1.0), 0.0)
        self.assertAlmostEqual(law.cdf(0.5), 0.25)
        self.assertAlmostEqual(law.cdf(2.0), 0.75)
        self.assertAlmostEqual(law.pdf(2.0), 0.25)
        self.assertAlmostEqual(law.quantile(0.75), 2.0)
        self.assertEqual(law.quantile(1.0), 3.0)
        with self.assertRaises(ValueError):
            EmpiricalMarginal((1.0, 1.0))


class SurrogateStudyTests(unittest.TestCase):
    """Emulator stages appear in the parent and are labelled as surrogate."""

    def test_child_study_is_reported_with_surrogate_evaluations(self) -> None:
        """The parent report and JSON hold the child; totals stay separate."""
        study = UQStudy(capacity_model(), seed=5)
        surrogate = study.emulate("g", degree=2)
        surrogate.propagate(5000, design="mc")
        self.assertEqual(surrogate.evaluation_kind, "surrogate")
        self.assertEqual(surrogate.records[0].evaluation_kind, "surrogate")
        self.assertEqual(study.children, ((1, surrogate),))
        self.assertEqual(study.total_evaluations, study.records[0].evaluations)
        self.assertLess(study.total_evaluations, 5000)
        report = study.report()
        self.assertIn("### Surrogate study:", report)
        self.assertIn("Surrogate evaluations: 5,000", report)
        payload = json.loads(study.to_json())
        child = payload["surrogate_studies"][0]
        self.assertEqual(child["after_stage"], 1)
        self.assertEqual(child["study"]["stages"][0]["evaluation_kind"], "surrogate")


class PersistenceTests(unittest.TestCase):
    """P1.3: JSON export, provenance, and pickling."""

    def test_json_record_is_complete_and_strict(self) -> None:
        """No NaN tokens; provenance, settings, and the law hash are present."""
        study = UQStudy(capacity_model(), seed=21)
        study.propagate(256, design="lhs")
        study.form("g", 0.0)
        study.extremes("g", size=512, return_periods=(50.0, 1e6))
        text = study.to_json()
        payload = json.loads(text)
        info = payload["provenance"]
        for key in ("library_version", "python_version", "platform", "created_utc"):
            self.assertTrue(info[key])
        self.assertEqual(info["root_seed"], 21)
        law_hash = input_law_hash(study.model.inputs)
        self.assertEqual(payload["input_law"]["hash"], law_hash)
        stages = payload["stages"]
        kinds = [s["stage"] for s in stages]
        self.assertEqual(kinds, ["propagation", "form", "extremes"])
        self.assertEqual(stages[0]["settings"]["design"], "lhs")
        self.assertEqual(stages[1]["summary"]["converged"], "yes")
        self.assertIn("n/a", stages[2]["summary"]["empirical x[T=1e+06]"])
        self.assertEqual(stages[0]["result_type"], "PropagationStudy")
        with tempfile.TemporaryDirectory() as directory:
            path = study.save_json(Path(directory) / "out" / "study.json")
            self.assertEqual(json.loads(path.read_text()), payload)
        record = StageRecord("x", "x", {"v": math.inf}, 0, 1, None)
        self.assertEqual(json.loads(record.to_json())["summary"]["v"], "inf")

    def test_study_and_records_pickle(self) -> None:
        """A study on a module-level model round-trips through pickle."""
        study = UQStudy(capacity_model(), seed=8, cache=True)
        study.propagate(128, design="mc")
        study.reliability("g", 0.0, methods=("form", "mc"), monte_carlo_samples=500)
        study.sensitivity("g", 64, bootstrap=10)
        restored = pickle.loads(pickle.dumps(study))
        self.assertEqual(restored.report(), study.report())
        self.assertEqual(
            restored.records[0].result["g"].values, study.records[0].result["g"].values
        )
        restored.propagate(64, design="mc")
        self.assertEqual(len(restored.records), 4)
        record = pickle.loads(pickle.dumps(study.records[1]))
        self.assertEqual(record, study.records[1])


class FormFlagTests(unittest.TestCase):
    """0.12: FORM convergence is recorded and unconverged runs warn."""

    def test_flags_are_recorded_when_converged(self) -> None:
        """form, reliability, and design_points state 'converged: yes'."""
        study = UQStudy(capacity_model(), seed=2)
        study.form("g", 0.0)
        study.reliability("g", 0.0, methods=("form", "sorm"))
        study.design_points("g", 0.0, max_points=1, importance_samples=0)
        self.assertEqual(study.records[0].summary["converged"], "yes")
        self.assertEqual(study.records[1].summary["FORM converged"], "yes")
        self.assertEqual(study.records[2].summary["FORM converged[1]"], "yes")
        self.assertEqual([r.warnings for r in study.records], [(), (), ()])

    def test_repeat_of_form_costs_no_evaluation(self) -> None:
        """The convergence check in reliability is served from the stage cache."""
        alone = UQStudy(capacity_model(), seed=2)
        rows = alone.reliability("g", 0.0, methods=("form",))
        record = alone.latest("reliability")
        self.assertEqual(record.evaluations, rows.row("FORM").evaluations)
        self.assertEqual(record.cache_hits, rows.row("FORM").evaluations)

    def test_unconverged_form_warns_in_record_and_report(self) -> None:
        """A non-converged search is flagged instead of silently reported."""
        from probstat.reliability import form_hlrf

        def stalled(*args, **kwargs):
            """Run FORM, then mark it as stopped by the iteration limit."""
            result = form_hlrf(*args, **kwargs)
            return dataclasses.replace(
                result, converged=False, message="maximum iterations reached"
            )

        study = UQStudy(capacity_model(), seed=2)
        with mock.patch("probstat.workflow.study.form_hlrf", stalled):
            with self.assertWarnsRegex(UserWarning, "did not converge"):
                study.form("g", 0.0)
            with self.assertWarnsRegex(UserWarning, "unconverged design point"):
                study.reliability("g", 0.0, methods=("form",))
        self.assertEqual(
            study.records[0].summary["converged"], "no (maximum iterations reached)"
        )
        self.assertTrue(study.records[1].warnings)
        self.assertIn("> **Warning:** FORM for g did not converge", study.report())


class StageOrderTests(unittest.TestCase):
    """0.11: sample-based stages do not reuse quasi-random samples."""

    def test_extremes_and_information_do_not_depend_on_stage_order(self) -> None:
        """With or without an earlier Sobol propagation the results agree."""
        first = UQStudy(capacity_model(), seed=9)
        first.propagate(1024, design="sobol")
        tail_a = first.extremes("g", tail="lower", size=2048)
        info_a = first.information_sensitivity("g", max_points=256)
        second = UQStudy(capacity_model(), seed=9)
        info_b = second.information_sensitivity("g", max_points=256)
        tail_b = second.extremes("g", tail="lower", size=2048)
        self.assertEqual(tail_a.fit, tail_b.fit)
        self.assertEqual(info_a.mutual_information, info_b.mutual_information)
        self.assertEqual(tail_a.source, "fresh mc sample")

    def test_reuse_refuses_quasi_random_samples(self) -> None:
        """reuse=True falls back to a fresh MC sample after a Sobol propagation."""
        study = UQStudy(capacity_model(), seed=9)
        study.propagate(512, design="sobol")
        info = study.information_sensitivity("g", max_points=200, reuse=True)
        self.assertIn("only plain Monte Carlo samples are reused", info.source)
        expected = propagate(
            capacity_model(),
            200,
            design="mc",
            seed=resolve_stream(9).child("information", 0),
        )
        self.assertEqual(info.sample_size, 200)
        self.assertEqual(study.latest("information").evaluations, 200)
        direct = tuple(
            ksg_mutual_information(
                expected.input_column(name), expected["g"].values, k=3
            )
            for name in expected.input_names
        )
        self.assertEqual(info.mutual_information, direct)


class EngineIntegrationTests(unittest.TestCase):
    """P1.5 inside the study: executor and shared cache."""

    def test_executor_does_not_change_results(self) -> None:
        """A thread pool gives the same propagation as sequential runs."""
        plain = UQStudy(capacity_model(), seed=6).propagate(300, design="lhs")
        with ThreadPoolExecutor(max_workers=3) as pool:
            parallel = UQStudy(capacity_model(), seed=6, executor=pool).propagate(
                300, design="lhs", batch=16
            )
        self.assertEqual(parallel["g"].values, plain["g"].values)
        self.assertEqual(parallel.model_evaluations, 300)

    def test_shared_cache_reuses_runs_across_stages(self) -> None:
        """design_points repeats the FORM search of form at no cost."""
        study = UQStudy(capacity_model(), seed=6, cache=True)
        form = study.form("g", 0.0)
        study.design_points("g", 0.0, max_points=1, importance_samples=0)
        record = study.latest("design_points")
        self.assertGreaterEqual(record.cache_hits, form.evaluations)
        self.assertEqual(record.evaluations, 0)
        self.assertIn("Cache hits (not evaluations)", study.report())


class HtmlReportTests(unittest.TestCase):
    """0.19 and P1.3: HTML without Matplotlib and with provenance."""

    def test_html_degrades_without_matplotlib_and_shows_root_seed(self) -> None:
        """The page is written without figures and says why."""
        from probstat.visualization import study as page

        study = UQStudy(capacity_model(), seed=31)
        study.propagate(128, design="mc")
        with mock.patch(
            "probstat.visualization.theme.matplotlib_available", return_value=False
        ):
            text = page.study_html(study)
            with tempfile.TemporaryDirectory() as directory:
                path = page.write_study_html(study, Path(directory) / "s.html")
                self.assertTrue(path.exists())
        self.assertIn("Figures omitted: Matplotlib is not installed", text)
        self.assertNotIn("data:image/png", text)
        self.assertIn("root seed 31", text)
        self.assertIn("reproducible from the root seed 31", text)
        self.assertIn("Settings: size=128", text)
        self.assertIn("Python", text)

    @unittest.skipIf(
        importlib.util.find_spec("matplotlib") is not None, "Matplotlib is installed"
    )
    def test_figures_without_matplotlib_raise_actionable_error(self) -> None:
        """Figure functions name the extra to install."""
        from probstat.visualization.study import save_study_figures
        from probstat.visualization.theme import PORCELAIN, chart_theme

        with self.assertRaisesRegex(ImportError, r"\[visualization\]"):
            with chart_theme(PORCELAIN):
                pass
        with self.assertRaisesRegex(ImportError, r"\[visualization\]"):
            save_study_figures(UQStudy(capacity_model()), tempfile.gettempdir())


if __name__ == "__main__":
    unittest.main()
