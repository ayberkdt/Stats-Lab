"""Tests for ``UQStudy.propagate_until``, the error-targeted propagation stage.

Since ADR-0058 the sequential pilot only sizes the sample; the reported
estimates come from a fresh independent sample of the chosen size
(``final="fresh"``), which removes the optional-stopping bias of reusing the
pilot (``final="reuse"``). The simulation studies at the end show both.
"""

from __future__ import annotations

import math
import statistics
import unittest
import warnings
from statistics import NormalDist

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
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


def total(point):
    """Return the sum of the two inputs."""
    return {"y": point["a"] + point["b"]}


def sum_model() -> ProbabilisticModel:
    """Return ``Y = a + b`` with ``a ~ N(10, 1.6)``, ``b ~ N(0, 1.2)``."""
    inputs = JointInput.independent(
        RandomInput("a", Normal(10.0, 1.6)), RandomInput("b", Normal(0.0, 1.2))
    )
    return ProbabilisticModel(
        inputs, total, (QuantityOfInterest("y", "sum"),), name="sum"
    )


def affine(point):
    """Return ``y = 1 + 0.5 z``."""
    return {"y": 1.0 + 0.5 * point["z"]}


def affine_model() -> ProbabilisticModel:
    """Return ``Y ~ N(1, 0.5)`` (coefficient of variation 0.5)."""
    inputs = JointInput.independent(RandomInput("z", Normal(0.0, 1.0)))
    return ProbabilisticModel(inputs, affine, (QuantityOfInterest("y", "y"),))


class PropagateUntilMeanTests(unittest.TestCase):
    """Mean-controlled stopping."""

    def test_reproducible_by_seed_and_streams_advance(self) -> None:
        """Same seed gives an equal result (no NaN fields); new runs use new streams.

        This failed on Python 3.13 while unused fields held NaN sentinels,
        because dataclass equality compares fields and ``nan != nan``.
        """
        first = UQStudy(sum_model(), seed=5)
        second = UQStudy(sum_model(), seed=5)
        a = first.propagate_until("y", target_relative_error=0.02, batch=64)
        b = second.propagate_until("y", target_relative_error=0.02, batch=64)
        self.assertEqual(a, b)
        self.assertIsNone(a.probability)
        self.assertIsNone(a.probability_standard_error)
        self.assertIsNone(a.probability_interval)
        again = first.propagate_until("y", target_relative_error=0.02, batch=64)
        self.assertNotEqual(a.mean, again.mean)
        seeds = [r.seed for r in first.records]
        self.assertEqual(len(set(seeds)), 2)

    def test_reaches_target_near_theoretical_sample_size(self) -> None:
        """The pilot stops near ``n = (cv / target)^2 = 400``; a fresh n follows.

        Observed for seeds 1, 2, 3 with batch 50: pilots of 500, 400, 400.
        """
        theory = (EXACT.stdev / EXACT.mean / 0.01) ** 2
        self.assertAlmostEqual(theory, 400.0)
        for seed in (1, 2, 3):
            study = UQStudy(sum_model(), seed=seed)
            result = study.propagate_until("y", target_relative_error=0.01, batch=50)
            self.assertIsInstance(result, TargetedPropagation)
            self.assertTrue(result.converged)
            self.assertEqual(result.controlled, "mean")
            self.assertEqual(result.final, "fresh")
            self.assertLessEqual(result.pilot_relative_error, 0.01)
            self.assertLess(abs(result.relative_error - 0.01), 0.002)
            self.assertEqual(result.pilot_size % 50, 0)
            self.assertGreaterEqual(result.pilot_size, 0.7 * theory)
            self.assertLessEqual(result.pilot_size, 1.3 * theory)
            self.assertEqual(result.sample_size, result.pilot_size)
            # Stopping is at the first batch meeting the target.
            self.assertEqual(result.batches, result.pilot_size // 50)
            record = study.latest("sequential_propagation")
            self.assertEqual(record.evaluations, 2 * result.sample_size)
            self.assertEqual(result.evaluations, record.evaluations)
            self.assertLess(abs(result.mean - 10.0), 4.0 * result.standard_error)
            assert result.mean_interval is not None
            low, high = result.mean_interval
            self.assertAlmostEqual(
                (high - low) / 2.0,
                NormalDist().inv_cdf(0.975) * result.standard_error,
                delta=0.01 * result.standard_error,
            )
            for level, estimate in result.quantiles:
                # Four asymptotic standard errors sqrt(p(1-p)/n) / f(q_p).
                q = EXACT.inv_cdf(level)
                spread = math.sqrt(level * (1 - level) / result.sample_size)
                self.assertLess(abs(estimate - q), 4.0 * spread / EXACT.pdf(q))
            self.assertIsNone(result.probability)
            self.assertEqual(result.quantile(0.5), result.quantiles[1][1])
            with self.assertRaises(KeyError):
                result.quantile(0.25)

    def test_reuse_reports_the_pilot_and_warns(self) -> None:
        """``final='reuse'`` is the legacy behaviour: no second sample."""
        study = UQStudy(sum_model(), seed=1)
        with self.assertWarnsRegex(UserWarning, "optional-stopping bias"):
            legacy = study.propagate_until(
                "y", target_relative_error=0.01, batch=50, final="reuse"
            )
        self.assertEqual(legacy.sample_size, legacy.pilot_size)
        self.assertEqual(legacy.relative_error, legacy.pilot_relative_error)
        record = study.latest("sequential_propagation")
        self.assertEqual(record.evaluations, legacy.sample_size)
        self.assertIn("optional-stopping bias", record.summary["final sample"])
        self.assertTrue(record.warnings)
        fresh = UQStudy(sum_model(), seed=1).propagate_until(
            "y", target_relative_error=0.01, batch=50
        )
        self.assertEqual(fresh.pilot_size, legacy.pilot_size)
        self.assertNotEqual(fresh.mean, legacy.mean)

    def test_non_convergence_is_reported(self) -> None:
        """A tiny budget stops the pilot at ``max_samples`` and says so."""
        study = UQStudy(sum_model(), seed=0)
        with self.assertWarnsRegex(UserWarning, "max_samples"):
            result = study.propagate_until(
                "y", target_relative_error=1e-4, max_samples=300, batch=128
            )
        self.assertFalse(result.converged)
        self.assertEqual(result.pilot_size, 300)
        self.assertEqual(result.sample_size, 300)
        self.assertEqual(result.batches, 3)  # 128 + 128 + truncated 44
        self.assertGreater(result.relative_error, 1e-4)
        record = study.latest("sequential_propagation")
        self.assertEqual(record.summary["converged"], "no (max_samples reached)")
        self.assertEqual(record.evaluations, 600)

    def test_confidence_sets_every_interval(self) -> None:
        """A 99% interval is wider than a 90% one on the same sample."""
        wide = UQStudy(sum_model(), seed=4).propagate_until(
            "y", threshold=12.0, target_relative_error=0.1, confidence=0.99
        )
        narrow = UQStudy(sum_model(), seed=4).propagate_until(
            "y", threshold=12.0, target_relative_error=0.1, confidence=0.90
        )
        assert wide.mean_interval and narrow.mean_interval
        assert wide.probability_interval and narrow.probability_interval
        self.assertEqual(wide.mean, narrow.mean)
        self.assertGreater(
            wide.mean_interval[1] - wide.mean_interval[0],
            narrow.mean_interval[1] - narrow.mean_interval[0],
        )
        self.assertGreater(
            wide.probability_interval[1] - wide.probability_interval[0],
            narrow.probability_interval[1] - narrow.probability_interval[0],
        )
        self.assertEqual(wide.confidence, 0.99)

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
        with self.assertRaises(ValueError):
            study.propagate_until("y", confidence=1.0)
        with self.assertRaises(ValueError):
            study.propagate_until("y", final="pooled")  # type: ignore[arg-type]
        self.assertEqual(study.records, ())
        self.assertEqual(study.model.evaluations, 0)


class PropagateUntilProbabilityTests(unittest.TestCase):
    """Exceedance-controlled stopping."""

    def test_exceedance_event_matches_propagate(self) -> None:
        """``P[Y > 12]`` (strict, as in propagate) stops near 2121 and covers truth.

        Theory: ``(1-p)/(p eps^2) = 2121`` runs for ``p = 0.15866`` and 5%.
        """
        exact = 1.0 - EXACT.cdf(12.0)
        theory = (1.0 - exact) / (exact * 0.05**2)
        study = UQStudy(sum_model(), seed=2)
        result = study.propagate_until(
            "y", target_relative_error=0.05, threshold=12.0, batch=100
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.controlled, "probability")
        self.assertLess(abs(result.pilot_size - theory), 0.3 * theory)
        assert result.probability is not None and result.probability_interval
        lower, upper = result.probability_interval
        self.assertLess(lower, exact)
        self.assertLess(exact, upper)
        self.assertEqual(result.events, round(result.probability * result.sample_size))
        self.assertIn("P[y > 12]", study.latest("sequential_propagation").summary)

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
            assert result.probability_interval is not None
            lower, upper = result.probability_interval
            covered += lower < exact < upper
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
        self.assertIn("| P[y > 12] |", report)
        self.assertIn("| converged | yes |", report)
        self.assertIn("| P[y > 12] Wilson 95% lower |", report)
        self.assertIn("final=fresh", report)


class PropagateUntilMinimumSampleTests(unittest.TestCase):
    """Tiny batches cannot stop before min_samples."""

    def test_batch_of_two_waits_for_min_samples(self) -> None:
        """Seed 24 used to stop at n = 2 with a claimed error 80x too small."""
        study = UQStudy(sum_model(), seed=24)
        result = study.propagate_until("y", target_relative_error=0.01, batch=2)
        self.assertGreaterEqual(result.pilot_size, 100)

    def test_all_failures_do_not_claim_zero_error(self) -> None:
        """An event seen in every draw keeps a positive relative error."""
        study = UQStudy(sum_model(), seed=3)
        result = study.propagate_until(
            "y", threshold=-100.0, failure="above", max_samples=512
        )
        self.assertGreater(result.relative_error, 0.0)


def _replicate(final: str, seeds: range, **options: object) -> tuple[list[float], int]:
    """Return the estimates of many seeded runs and how many intervals cover.

    Mean mode estimates ``E[Y] = 1``; with ``threshold`` the estimate is
    ``P(Y > 1.64)``.
    """
    exact = 1.0 if "threshold" not in options else 1.0 - NormalDist(1.0, 0.5).cdf(1.64)
    estimates: list[float] = []
    covered = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in seeds:
            result = UQStudy(affine_model(), seed=seed).propagate_until(
                "y", final=final, **options  # type: ignore[arg-type]
            )
            if "threshold" in options:
                assert result.probability is not None and result.probability_interval
                estimates.append(result.probability)
                low, high = result.probability_interval
            else:
                assert result.mean_interval is not None
                estimates.append(result.mean)
                low, high = result.mean_interval
            covered += low <= exact <= high
    return estimates, covered


class OptionalStoppingSimulationTests(unittest.TestCase):
    """Seeded simulation studies of the stopping bias (300 runs each).

    ``Y ~ N(1, 0.5)``. Mean target 5% (``batch=5, min_samples=10``): the
    pilot needs about 100 runs. Probability ``P(Y > 1.64) = 0.10027`` with a
    20% target (``batch=10, min_samples=20``): about 230 runs. Observed for
    seeds 0..299: mean bias reuse +1.04%, fresh +0.11%; probability bias
    reuse +4.9%, fresh -1.4%; coverage of fresh intervals 0.927 (mean) and
    0.927 (probability). A bias is called significant when it exceeds three
    empirical standard errors of the averaged estimate.
    """

    SEEDS = range(300)

    def _bias_in_standard_errors(self, estimates: list[float], exact: float) -> float:
        """Return ``(mean - exact) / (sd / sqrt(R))``."""
        spread = statistics.stdev(estimates) / math.sqrt(len(estimates))
        return (statistics.fmean(estimates) - exact) / spread

    def _assert_nominal(self, covered: int) -> None:
        """Coverage within three binomial standard deviations of 0.95."""
        rate = covered / len(self.SEEDS)
        spread = math.sqrt(0.95 * 0.05 / len(self.SEEDS))
        self.assertLess(abs(rate - 0.95), 3.0 * spread)

    def test_mean_fresh_is_unbiased_reuse_is_not(self) -> None:
        """Fresh estimates of ``E[Y] = 1`` are unbiased; the pilot's are not."""
        options = {"target_relative_error": 0.05, "batch": 5, "min_samples": 10}
        fresh, covered = _replicate("fresh", self.SEEDS, **options)
        reused, _ = _replicate("reuse", self.SEEDS, **options)
        self.assertLess(abs(self._bias_in_standard_errors(fresh, 1.0)), 3.0)
        self.assertGreater(self._bias_in_standard_errors(reused, 1.0), 3.0)
        self._assert_nominal(covered)

    def test_probability_fresh_is_unbiased_reuse_is_not(self) -> None:
        """Reusing the pilot over-estimates a small probability by about 5%."""
        exact = 1.0 - NormalDist(1.0, 0.5).cdf(1.64)
        options = {
            "target_relative_error": 0.2,
            "threshold": 1.64,
            "batch": 10,
            "min_samples": 20,
        }
        fresh, covered = _replicate("fresh", self.SEEDS, **options)
        reused, _ = _replicate("reuse", self.SEEDS, **options)
        self.assertLess(abs(self._bias_in_standard_errors(fresh, exact)), 3.0)
        self.assertGreater(self._bias_in_standard_errors(reused, exact), 3.0)
        self._assert_nominal(covered)


if __name__ == "__main__":
    unittest.main()
