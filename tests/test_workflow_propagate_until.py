"""Tests for ``UQStudy.propagate_until``, the error-targeted propagation stage."""

from __future__ import annotations

import math
import unittest
from statistics import NormalDist

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import propagate
from probstat.uq import (
    WORKFLOW_GUIDE,
    QuantityOfInterest,
    StepStatus,
    UQWorkflowRecord,
    WorkflowStep,
)
from probstat.workflow import TargetedPropagation, UQStudy

#: Exact law of the QoI below: ``Y ~ Normal(10, 2)``, coefficient of variation 0.2.
EXACT = NormalDist(10.0, 2.0)


def sum_model() -> ProbabilisticModel:
    """Return ``Y = a + b`` with ``a ~ N(10, 1.6)``, ``b ~ N(0, 1.2)``."""
    inputs = JointInput.independent(
        RandomInput("a", Normal(10.0, 1.6)), RandomInput("b", Normal(0.0, 1.2))
    )

    def total(point):
        """Return the sum of the two inputs."""
        return {"y": point["a"] + point["b"]}

    return ProbabilisticModel(
        inputs, total, (QuantityOfInterest("y", "sum"),), name="sum"
    )


class PropagateUntilMeanTests(unittest.TestCase):
    """Mean-controlled stopping."""

    def test_reproducible_by_seed_and_streams_advance(self) -> None:
        """Same seed gives the same result; a second run uses a new stream."""
        first = UQStudy(sum_model(), seed=5)
        second = UQStudy(sum_model(), seed=5)
        a = first.propagate_until("y", target_relative_error=0.02, batch=64)
        b = second.propagate_until("y", target_relative_error=0.02, batch=64)
        self.assertEqual(a, b)
        again = first.propagate_until("y", target_relative_error=0.02, batch=64)
        self.assertNotEqual(a.mean, again.mean)
        seeds = [r.seed for r in first.records]
        self.assertEqual(len(set(seeds)), 2)

    def test_reaches_target_near_theoretical_sample_size(self) -> None:
        """Samples used track ``n = (cv / target)^2 = 400`` for cv 0.2, target 1%.

        Observed for seeds 1, 2, 3 with batch 50: 500, 400, 400 samples.
        """
        theory = (EXACT.stdev / EXACT.mean / 0.01) ** 2
        self.assertAlmostEqual(theory, 400.0)
        for seed in (1, 2, 3):
            study = UQStudy(sum_model(), seed=seed)
            result = study.propagate_until("y", target_relative_error=0.01, batch=50)
            self.assertIsInstance(result, TargetedPropagation)
            self.assertTrue(result.converged)
            self.assertEqual(result.controlled, "mean")
            self.assertLessEqual(result.relative_error, 0.01)
            self.assertEqual(result.sample_size % 50, 0)
            self.assertGreaterEqual(result.sample_size, 0.7 * theory)
            self.assertLessEqual(result.sample_size, 1.3 * theory)
            # Stopping is at the first batch meeting the target.
            self.assertEqual(result.batches, result.sample_size // 50)
            self.assertEqual(
                study.latest("sequential_propagation").evaluations, result.sample_size
            )
            self.assertLess(abs(result.mean - 10.0), 4.0 * result.standard_error)
            for level, estimate in result.quantiles:
                self.assertLess(abs(estimate - EXACT.inv_cdf(level)), 0.5)
            self.assertTrue(math.isnan(result.probability))
            self.assertEqual(result.quantile(0.5), result.quantiles[1][1])
            with self.assertRaises(KeyError):
                result.quantile(0.25)

    def test_non_convergence_is_reported(self) -> None:
        """A tiny budget stops at ``max_samples`` with ``converged=False``."""
        study = UQStudy(sum_model(), seed=0)
        result = study.propagate_until(
            "y", target_relative_error=1e-4, max_samples=300, batch=128
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.sample_size, 300)
        self.assertEqual(result.batches, 3)  # 128 + 128 + truncated 44
        self.assertGreater(result.relative_error, 1e-4)
        record = study.latest("sequential_propagation")
        self.assertEqual(record.summary["converged"], "no (max_samples reached)")
        self.assertEqual(record.evaluations, 300)

    def test_invalid_arguments_raise(self) -> None:
        """Unknown QoI and out-of-domain arguments are rejected before sampling."""
        study = UQStudy(sum_model(), seed=0)
        with self.assertRaises(ValueError):
            study.propagate_until("z")
        with self.assertRaises(ValueError):
            study.propagate_until("y", target_relative_error=0.0)
        with self.assertRaises(ValueError):
            study.propagate_until("y", batch=1)
        with self.assertRaises(ValueError):
            study.propagate_until("y", quantiles=(0.0,))
        with self.assertRaises(ValueError):
            study.propagate_until("y", threshold=1.0, failure="sideways")  # type: ignore[arg-type]
        self.assertEqual(study.records, ())
        self.assertEqual(study.model.evaluations, 0)


class PropagateUntilProbabilityTests(unittest.TestCase):
    """Exceedance-controlled stopping."""

    def test_exceedance_agrees_with_direct_estimate(self) -> None:
        """``P[Y >= 12]`` stops near ``(1-p)/(p eps^2) = 2121`` and covers truth.

        Observed with seed 2: 2200 samples, p = 0.1559, Wilson 95% interval
        (0.1414, 0.1717); exact p = 0.15866; a direct 20000-run estimate on an
        independent seed gives 0.1631.
        """
        exact = 1.0 - EXACT.cdf(12.0)
        theory = (1.0 - exact) / (exact * 0.05**2)
        study = UQStudy(sum_model(), seed=2)
        result = study.propagate_until(
            "y", target_relative_error=0.05, threshold=12.0, batch=100
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.controlled, "probability")
        self.assertLess(abs(result.sample_size - theory), 0.3 * theory)
        lower, upper = result.probability_interval
        self.assertLess(lower, exact)
        self.assertLess(exact, upper)
        direct_values = propagate(sum_model(), 20_000, design="mc", seed=99)
        values = direct_values.outputs["y"].values
        direct = sum(v >= 12.0 for v in values) / len(values)
        self.assertLess(lower, direct)
        self.assertLess(direct, upper)
        self.assertEqual(result.events, round(result.probability * result.sample_size))

    def test_below_failure_and_interval_coverage(self) -> None:
        """``P[Y <= 8]`` intervals cover the exact value for most seeds."""
        exact = EXACT.cdf(8.0)
        covered = 0
        seeds = range(12)
        for seed in seeds:
            result = UQStudy(sum_model(), seed=seed).propagate_until(
                "y",
                target_relative_error=0.1,
                threshold=8.0,
                failure="below",
                batch=100,
            )
            self.assertTrue(result.converged)
            lower, upper = result.probability_interval
            covered += lower < exact < upper
        # Nominal 95%; data-dependent stopping makes coverage slightly lower.
        self.assertGreaterEqual(covered, 9)


class PropagateUntilWorkflowTests(unittest.TestCase):
    """Stage-kind mapping and report integration."""

    def test_stage_maps_to_propagation_and_appears_in_report(self) -> None:
        """The record maps to the propagation step and the report shows it."""
        guide = WORKFLOW_GUIDE[WorkflowStep.PROPAGATION]
        self.assertIn("sequential_propagation", guide.study_stages)
        self.assertIn("probstat.workflow.UQStudy.propagate_until", guide.library_calls)
        study = UQStudy(sum_model(), seed=3)
        study.propagate_until("y", target_relative_error=0.05, threshold=12.0)
        record = UQWorkflowRecord.start("sum", study.model.quantities)
        record = record.record_stages(study.records)
        entry = record.entry(WorkflowStep.PROPAGATION)
        self.assertEqual(entry.status, StepStatus.COMPLETE)
        self.assertIn("sequential_propagation", entry.evidence[0])
        report = study.report()
        self.assertIn("sequential_propagation: Monte Carlo of y", report)
        self.assertIn("| P[y >= 12] |", report)
        self.assertIn("| converged | yes |", report)


class PropagateUntilMinimumSampleTests(unittest.TestCase):
    """Tiny batches cannot stop before min_samples."""

    def test_batch_of_two_waits_for_min_samples(self) -> None:
        """Seed 24 used to stop at n = 2 with a claimed error 80x too small."""
        study = UQStudy(sum_model(), seed=24)
        result = study.propagate_until("y", target_relative_error=0.01, batch=2)
        self.assertGreaterEqual(result.sample_size, 100)

    def test_all_failures_do_not_claim_zero_error(self) -> None:
        """An event seen in every draw keeps a positive relative error."""
        study = UQStudy(sum_model(), seed=3)
        result = study.propagate_until(
            "y", threshold=-100.0, failure="above", max_samples=512
        )
        self.assertGreater(result.relative_error, 0.0)
