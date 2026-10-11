"""Tests for component predictions, workflow records, and the worked example."""

from __future__ import annotations

import importlib
import importlib.util
import math
import random
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from probstat.uq import (
    WORKFLOW_GUIDE,
    ProblemDirection,
    QuantityOfInterest,
    StepStatus,
    UQWorkflowRecord,
    WorkflowEntry,
    WorkflowStep,
    verify_solution,
)
from probstat.validation import (
    AdditiveGaussianDiscrepancy,
    CredibilityRecord,
    DiscrepancyExtrapolation,
    ValidationDomain,
    ValidationExperiment,
    asme_validation_comparison,
    estimate_discrepancy,
    predict_with_uncertainty_components,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "uq_workflow_lab.py"


def gaussian_samples(mean: float, sd: float, size: int, seed: int) -> list[float]:
    """Return seeded Gaussian samples."""
    generator = random.Random(seed)
    return [generator.gauss(mean, sd) for _ in range(size)]


def numerical_statement(half_width: float):
    """Return a monotone solution-verification statement with a given U."""
    # phi(h) = 1 + c h^2 on h = 0.1, 0.05, 0.025 gives GCI = 1.25 c h1^2 / 1.
    coefficient = half_width / (1.25 * 0.025**2)
    spacings = (0.1, 0.05, 0.025)
    return verify_solution("y", spacings, [1.0 + coefficient * h**2 for h in spacings])


def domain_on_time() -> ValidationDomain:
    """Return a validation domain spanning t in [0, 2]."""
    experiments = [
        ValidationExperiment(f"t{t}", [1.0], {"t": t}, provenance="test")
        for t in (0.0, 1.0, 2.0)
    ]
    return ValidationDomain.from_experiments(experiments)


class DiscrepancyTests(unittest.TestCase):
    """Estimation and extrapolation of the model-form component."""

    def test_moment_estimate_from_comparisons(self) -> None:
        """sigma^2 = mean(E^2) - mean(u_val^2) with the bias reported."""
        comparisons = [
            asme_validation_comparison(s, 0.0, data_uncertainty=0.1)
            for s in (0.3, -0.3, 0.3)
        ]
        estimate = estimate_discrepancy(comparisons, applicability_domain="t in [0,2]")
        self.assertAlmostEqual(estimate.discrepancy.variance, 0.09 - 0.01, places=12)
        self.assertAlmostEqual(estimate.mean_error, 0.1, places=12)
        self.assertFalse(estimate.truncated)
        small = [asme_validation_comparison(0.01, 0.0, data_uncertainty=0.1)]
        truncated = estimate_discrepancy(small, applicability_domain="t = 0")
        self.assertTrue(truncated.truncated)
        self.assertEqual(truncated.discrepancy.standard_deviation, 0.0)
        with self.assertRaises(ValueError):
            estimate_discrepancy(comparisons)
        with self.assertRaises(ValueError):
            estimate_discrepancy([], applicability_domain="x")
        with self.assertRaises(TypeError):
            estimate_discrepancy([1.0], applicability_domain="x")  # type: ignore[list-item]

    def test_variance_grows_linearly_with_scaled_distance(self) -> None:
        """At a validation point d=0; two ranges away sigma^2 triples (kappa=1)."""
        base = AdditiveGaussianDiscrepancy(0.2, "test", "t in [0, 2]")
        rule = DiscrepancyExtrapolation(base, domain_on_time(), growth_rate=1.0)
        inside = rule.at({"t": 1.0})
        self.assertEqual(inside.distance, 0.0)
        self.assertAlmostEqual(inside.standard_deviation, 0.2)
        outside = rule.at({"t": 6.0})
        self.assertAlmostEqual(outside.distance, 2.0)
        self.assertAlmostEqual(outside.variance, 0.04 * 3.0, places=12)
        self.assertTrue(outside.domain_check is not None)
        self.assertTrue(outside.domain_check.extrapolation)  # type: ignore[union-attr]
        constant = DiscrepancyExtrapolation(base, domain_on_time(), growth_rate=0.0)
        self.assertAlmostEqual(constant.at({"t": 6.0}).standard_deviation, 0.2)
        with self.assertRaises(ValueError):
            DiscrepancyExtrapolation(base, domain_on_time(), growth_rate=-1.0)
        with self.assertRaises(TypeError):
            DiscrepancyExtrapolation(0.2, domain_on_time(), growth_rate=1.0)  # type: ignore[arg-type]

    def test_zero_range_dimension_cannot_be_extrapolated(self) -> None:
        """Leaving a dimension that had no validation spread is an error."""
        experiments = [
            ValidationExperiment(
                f"e{t}", [1.0], {"t": t, "load": 1.0}, provenance="test"
            )
            for t in (0.0, 1.0)
        ]
        domain = ValidationDomain.from_experiments(experiments)
        base = AdditiveGaussianDiscrepancy(0.1, "test", "load = 1")
        rule = DiscrepancyExtrapolation(base, domain, growth_rate=1.0)
        with self.assertRaises(ValueError):
            rule.at({"t": 0.5, "load": 2.0})


class ComponentPredictionTests(unittest.TestCase):
    """Separate reporting and explicit combination of the three components."""

    def test_normal_treatment_matches_variance_addition(self) -> None:
        """Independent Gaussian parts give sd sqrt(1 + 4 + 2.25)."""
        samples = gaussian_samples(10.0, 1.0, 20_000, seed=1)
        numerical = numerical_statement(1.959963984540054 * 1.5)
        discrepancy = AdditiveGaussianDiscrepancy(2.0, "test", "everywhere")
        result = predict_with_uncertainty_components(
            "y",
            {"t": 1.0},
            samples,
            numerical=numerical,
            numerical_treatment="normal",
            discrepancy=discrepancy,
            seed=7,
        )
        self.assertAlmostEqual(result.numerical_standard_uncertainty or 0.0, 1.5)
        self.assertEqual(
            set(result.variance_budget), {"parametric", "numerical", "model_form"}
        )
        self.assertAlmostEqual(result.variance_budget["numerical"], 2.25, places=9)
        self.assertAlmostEqual(result.variance_budget["model_form"], 4.0)
        self.assertAlmostEqual(math.fsum(result.variance_shares().values()), 1.0)
        expected = 1.959963984540054 * math.sqrt(1.0 + 4.0 + 2.25)
        low, high = result.combined_interval
        self.assertAlmostEqual(0.5 * (high - low), expected, delta=0.1)
        self.assertEqual(result.excluded, ())
        self.assertIn("constant discrepancy", result.discrepancy.rule)  # type: ignore[union-attr]

    def test_interval_treatment_widens_by_exactly_u(self) -> None:
        """The epistemic interval shifts both ends by U_num without a distribution."""
        samples = gaussian_samples(0.0, 1.0, 4000, seed=2)
        discrepancy = AdditiveGaussianDiscrepancy(0.5, "test", "everywhere")
        without = predict_with_uncertainty_components(
            "y", {}, samples, discrepancy=discrepancy, seed=3
        )
        numerical = numerical_statement(0.25)
        with_interval = predict_with_uncertainty_components(
            "y", {}, samples, numerical=numerical, discrepancy=discrepancy, seed=3
        )
        self.assertAlmostEqual(
            with_interval.combined_interval[0], without.combined_interval[0] - 0.25
        )
        self.assertAlmostEqual(
            with_interval.combined_interval[1], without.combined_interval[1] + 0.25
        )
        self.assertNotIn("numerical", with_interval.variance_budget)
        self.assertIsNone(with_interval.numerical_standard_uncertainty)
        self.assertIn("interval treatment", with_interval.combination_rule)
        self.assertTrue(any("numerical" in item for item in without.excluded))

    def test_uniform_treatment_and_reproducibility(self) -> None:
        """Uniform numerical error has u = U/sqrt(3); one seed reproduces draws."""
        samples = gaussian_samples(5.0, 0.1, 2000, seed=4)
        numerical = numerical_statement(0.3)
        first = predict_with_uncertainty_components(
            "y", {}, samples, numerical=numerical, numerical_treatment="uniform", seed=9
        )
        second = predict_with_uncertainty_components(
            "y", {}, samples, numerical=numerical, numerical_treatment="uniform", seed=9
        )
        self.assertAlmostEqual(
            first.numerical_standard_uncertainty or 0.0, 0.3 / math.sqrt(3.0)
        )
        self.assertEqual(first.combined_interval, second.combined_interval)
        self.assertTrue(all(abs(z - y) <= 0.3 for z, y in zip(
            first.combined_samples, first.parametric_samples, strict=True
        )))
        self.assertTrue(any("model-form" in item for item in first.excluded))

    def test_parametric_only_prediction_reports_exclusions(self) -> None:
        """Without other components the combined interval is the parametric one."""
        samples = gaussian_samples(0.0, 1.0, 1000, seed=5)
        result = predict_with_uncertainty_components("y", {}, samples, seed=1)
        self.assertEqual(result.combined_interval, result.parametric_interval)
        self.assertEqual(len(result.excluded), 2)
        self.assertIn("not supplied", result.summary())
        with self.assertRaises(ValueError):
            predict_with_uncertainty_components("y", {}, [1.0])
        with self.assertRaises(ValueError):
            predict_with_uncertainty_components("y", {}, samples, probability=1.0)
        with self.assertRaises(ValueError):
            predict_with_uncertainty_components(
                "y", {}, samples, numerical_treatment="box"  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            predict_with_uncertainty_components(
                "y", {}, samples, discrepancy=0.1  # type: ignore[arg-type]
            )

    def test_extrapolated_prediction_flows_into_credibility_record(self) -> None:
        """The statement names its components and the record flags extrapolation."""
        samples = gaussian_samples(1.0, 0.05, 2000, seed=6)
        base = AdditiveGaussianDiscrepancy(0.1, "test", "t in [0, 2]")
        rule = DiscrepancyExtrapolation(base, domain_on_time(), growth_rate=0.5)
        result = predict_with_uncertainty_components(
            "y",
            {"t": 4.0},
            samples,
            numerical=numerical_statement(0.02),
            discrepancy=rule,
            seed=11,
        )
        self.assertAlmostEqual(
            result.discrepancy.standard_deviation,  # type: ignore[union-attr]
            0.1 * math.sqrt(1.0 + 0.5 * 1.0),
        )
        statement = result.to_prediction_statement()
        self.assertIn("model form", statement.uncertainty_sources)
        self.assertIn("numerical", statement.uncertainty_sources)
        gaps = CredibilityRecord(predictions=(statement,)).gaps
        self.assertTrue(any("extrapolation" in gap for gap in gaps))
        self.assertIn("model form", result.summary())


@dataclass(frozen=True)
class FakeStage:
    """Minimal stand-in for ``probstat.workflow.StageRecord``."""

    stage: str
    label: str
    seed: int
    evaluations: int


class WorkflowRecordTests(unittest.TestCase):
    """Auditable status of every VV-UQ workflow step."""

    quantity = QuantityOfInterest("y", "response")

    def test_new_record_lists_every_step_as_a_gap(self) -> None:
        """A started record is pending everywhere and not complete."""
        record = UQWorkflowRecord.start("study", (self.quantity,))
        self.assertEqual([e.step for e in record.entries], list(WorkflowStep))
        self.assertEqual(len(record.gaps), 7)
        self.assertFalse(record.complete)
        self.assertEqual(record.directions, ())
        self.assertIn("| Step | Direction |", record.summary())

    def test_entries_require_evidence_or_justification(self) -> None:
        """Complete needs evidence; not applicable needs a note."""
        with self.assertRaises(ValueError):
            WorkflowEntry(WorkflowStep.VERIFICATION, StepStatus.COMPLETE)
        with self.assertRaises(ValueError):
            WorkflowEntry(WorkflowStep.CALIBRATION, StepStatus.NOT_APPLICABLE)
        entry = WorkflowEntry(
            WorkflowStep.CALIBRATION, StepStatus.NOT_APPLICABLE, note="no data"
        )
        self.assertTrue(entry.settled)
        with self.assertRaises(ValueError):
            UQWorkflowRecord("s", (self.quantity,), (entry,))

    def test_prerequisite_and_prediction_gaps(self) -> None:
        """Completing a step before its prerequisites is reported."""
        record = UQWorkflowRecord.start("study", (self.quantity,))
        record = record.update(
            WorkflowStep.VALIDATION, StepStatus.COMPLETE, evidence=("area 0.01",)
        )
        self.assertTrue(
            any("prerequisite verification" in gap for gap in record.gaps)
        )
        record = record.update(
            WorkflowStep.PREDICTION, StepStatus.COMPLETE, evidence=("interval",)
        )
        self.assertTrue(
            any("without a complete propagation" in gap for gap in record.gaps)
        )

    def test_record_stages_maps_uq_study_stages(self) -> None:
        """Forward stages map to propagation, calibration to the inverse step."""
        record = UQWorkflowRecord.start("study", (self.quantity,))
        record = record.record_stages(
            [
                FakeStage("propagation", "sobol, 64", 1, 64),
                FakeStage("reliability", "P(y<0)", 2, 100),
                FakeStage("calibration", "posterior", 3, 500),
            ]
        )
        propagation = record.entry(WorkflowStep.PROPAGATION)
        self.assertEqual(propagation.status, StepStatus.COMPLETE)
        self.assertEqual(len(propagation.evidence), 2)
        self.assertIn("stream seed 1", propagation.evidence[0])
        self.assertEqual(
            record.directions, (ProblemDirection.FORWARD, ProblemDirection.INVERSE)
        )
        with self.assertRaises(ValueError):
            record.record_stages([FakeStage("unknown", "x", 0, 0)])
        with self.assertRaises(TypeError):
            record.record_stages([object()])  # type: ignore[list-item]

    def test_guide_covers_every_step_with_resolvable_calls(self) -> None:
        """Each step has a guide whose library calls exist in the package."""
        self.assertEqual(set(WORKFLOW_GUIDE), set(WorkflowStep))
        self.assertIs(
            WORKFLOW_GUIDE[WorkflowStep.PROPAGATION].direction, ProblemDirection.FORWARD
        )
        self.assertIs(
            WORKFLOW_GUIDE[WorkflowStep.CALIBRATION].direction, ProblemDirection.INVERSE
        )
        for step, guide in WORKFLOW_GUIDE.items():
            self.assertTrue(guide.library_calls, step)
            for dotted in guide.library_calls:
                with self.subTest(call=dotted):
                    parts = dotted.split(".")
                    target: object = None
                    for cut in range(len(parts), 0, -1):
                        try:
                            target = importlib.import_module(".".join(parts[:cut]))
                        except ModuleNotFoundError:
                            continue
                        for attribute in parts[cut:]:
                            target = getattr(target, attribute)
                        break
                    self.assertTrue(callable(target), dotted)


def load_example():
    """Import the worked example module from the examples directory."""
    spec = importlib.util.spec_from_file_location("uq_workflow_lab", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WorkedExampleTests(unittest.TestCase):
    """The forward-inverse-predict example quoted in the foundations guide."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the worked example once with its documented seed."""
        cls.results = load_example().run()

    def test_forward_then_inverse_narrows_parametric_uncertainty(self) -> None:
        """Calibration shrinks the prior spread by more than an order of magnitude."""
        results = self.results
        prior_width = results.forward.quantile(0.95) - results.forward.quantile(0.05)
        low, high = results.components.parametric_interval
        self.assertGreater(prior_width, 10.0 * (high - low))
        means = results.posterior.mean()
        self.assertAlmostEqual(means["A"], 5.0, delta=0.25)
        self.assertAlmostEqual(means["k"], 0.3, delta=0.02)

    def test_verification_and_validation_evidence(self) -> None:
        """The GCI sees second order; validation detects the missing offset."""
        results = self.results
        self.assertEqual(results.verification.convergence.value, "monotone")
        self.assertAlmostEqual(results.verification.observed_order, 2.0, delta=0.4)
        self.assertFalse(results.assessment.consistent)
        self.assertGreater(results.estimate.discrepancy.standard_deviation, 0.03)
        self.assertLess(results.estimate.mean_error, 0.0)

    def test_prediction_needs_the_model_form_component(self) -> None:
        """Parametric spread alone misses the truth; the combined interval covers it."""
        results = self.results
        components = results.components
        low, high = components.parametric_interval
        self.assertFalse(low <= results.truth <= high)
        c_low, c_high = components.combined_interval
        self.assertTrue(c_low <= results.truth <= c_high)
        self.assertGreater(components.variance_shares()["model_form"], 0.5)
        self.assertTrue(components.discrepancy.domain_check.extrapolation)

    def test_workflow_record_is_complete(self) -> None:
        """Every step is settled with evidence and both directions are exercised."""
        record = self.results.record
        self.assertTrue(record.complete, record.gaps)
        self.assertEqual(
            record.directions, (ProblemDirection.FORWARD, ProblemDirection.INVERSE)
        )
        stages = [r.stage for r in self.results.study.records]
        self.assertEqual(stages, ["propagation", "calibration", "prediction"])


if __name__ == "__main__":
    unittest.main()
