"""Tests for ``UQStudy.verify``, the solution-verification stage."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import (
    WORKFLOW_GUIDE,
    QuantityOfInterest,
    StepStatus,
    UQWorkflowRecord,
    WorkflowStep,
    grid_convergence_index,
    verify_solution,
)
from probstat.validation import ValidationExperiment
from probstat.workflow import UQStudy

#: Fine-to-coarse QoI values ``phi(h) = 2 + 0.03 h^2 + 0.004 h^3`` on
#: ``h = 1, 2, 4`` (refinement ratio 2): observed order between 2 and 3.
GRID_VALUES = (2.034, 2.152, 2.736)


def spring_model() -> ProbabilisticModel:
    """Return the extension ``x = k * load`` with ``k ~ N(2, 0.1)``."""
    inputs = JointInput.independent(RandomInput("k", Normal(2.0, 0.1)))

    def spring(point):
        """Return the extension of a spring under a load."""
        return {"x": point["k"] * point["load"]}

    return ProbabilisticModel(
        inputs,
        spring,
        (QuantityOfInterest("x", "extension"),),
        fixed={"load": 1.0},
        name="spring",
    )


def campaign() -> list[ValidationExperiment]:
    """Return three seeded experiments drawn from the model."""
    rng = random.Random(3)
    return [
        ValidationExperiment(
            f"load-{load}",
            [rng.gauss(2.0, 0.1) * load + rng.gauss(0.0, 0.05) for _ in range(8)],
            {"load": load},
            measurement_uncertainty=0.05,
            provenance="synthetic",
        )
        for load in (1.0, 2.0, 3.0)
    ]


class VerifyStageTests(unittest.TestCase):
    """The stage reproduces the library calls and records the evidence."""

    def test_reproduces_direct_gci_call(self) -> None:
        """Order, extrapolation, and GCI equal grid_convergence_index exactly."""
        study = UQStudy(spring_model(), seed=1)
        result = study.verify("x", GRID_VALUES, 2.0)
        direct = grid_convergence_index((1.0, 2.0, 4.0), GRID_VALUES)
        self.assertEqual(result.observed_order, direct.order)
        self.assertEqual(result.extrapolated, direct.extrapolated)
        self.assertEqual(result.expanded_uncertainty, direct.gci_fine)
        self.assertEqual(
            result, verify_solution("x", (1.0, 2.0, 4.0), GRID_VALUES)
        )
        self.assertEqual(
            study.numerical_uncertainty("x"), direct.gci_fine / math.sqrt(3.0)
        )
        # Grid sizes in any order give the same statement as the ratio.
        again = study.verify("x", GRID_VALUES[::-1], grid_sizes=(4.0, 2.0, 1.0))
        self.assertEqual(again, result)

    def test_summary_keys_and_cost(self) -> None:
        """The summary carries order, extrapolation, GCI, u_num, and the check."""
        study = UQStudy(spring_model(), seed=1)
        study.verify("x", GRID_VALUES, 2.0, safety_factor=3.0)
        record = study.latest("verification")
        self.assertEqual(record.evaluations, 0)
        for key in (
            "observed_order",
            "extrapolated",
            "gci_fine",
            "U",
            "u_num",
            "asymptotic_ratio",
            "asymptotic_range",
            "convergence",
            "warnings",
        ):
            self.assertIn(key, record.summary)
        self.assertEqual(record.summary["safety_factor"], 3.0)
        self.assertEqual(record.summary["convergence"], "monotone")
        self.assertEqual(record.summary["asymptotic_range"], "False")
        self.assertIn("verification", study.report())

    def test_four_grids_and_oscillation(self) -> None:
        """Four grids use least squares; oscillation reports no order."""
        study = UQStudy(spring_model(), seed=1)
        values = tuple(2.0 + 0.03 * h * h for h in (1.0, 2.0, 4.0, 8.0))
        result = study.verify("x", values, 2.0)
        self.assertEqual(result.grid_count, 4)
        self.assertAlmostEqual(result.observed_order or 0.0, 2.0, places=6)
        study.verify("x", (2.0, 2.1, 1.95), 2.0)
        summary = study.latest("verification").summary
        self.assertEqual(summary["observed_order"], "n/a")
        self.assertEqual(summary["asymptotic_range"], "n/a (oscillatory)")

    def test_argument_errors(self) -> None:
        """Exactly one grid description; unknown QoIs and bad ratios fail."""
        study = UQStudy(spring_model(), seed=1)
        with self.assertRaises(ValueError):
            study.verify("x", GRID_VALUES)
        with self.assertRaises(ValueError):
            study.verify("x", GRID_VALUES, 2.0, grid_sizes=(1.0, 2.0, 4.0))
        with self.assertRaises(ValueError):
            study.verify("x", GRID_VALUES, 1.0)
        with self.assertRaises(KeyError):
            study.verify("y", GRID_VALUES, 2.0)
        with self.assertRaises(KeyError):
            study.numerical_uncertainty("x")
        with self.assertRaises(KeyError):
            study.validate(
                "x", campaign(), tolerance=0.3, numerical_uncertainty="verified"
            )
        with self.assertRaises(ValueError):
            study.validate(
                "x",
                campaign(),
                tolerance=0.3,
                numerical_uncertainty="stored",  # type: ignore[arg-type]
            )

    def test_stage_kind_maps_to_verification_step(self) -> None:
        """The guide lists the stage and record_stages completes verification."""
        guide = WORKFLOW_GUIDE[WorkflowStep.VERIFICATION]
        self.assertIn("verification", guide.study_stages)
        self.assertIn("probstat.workflow.UQStudy.verify", guide.library_calls)
        study = UQStudy(spring_model(), seed=1)
        study.verify("x", GRID_VALUES, 2.0)
        record = UQWorkflowRecord.start(
            "spring", (QuantityOfInterest("x", "extension"),)
        ).record_stages(study.records)
        self.assertIs(
            record.entry(WorkflowStep.VERIFICATION).status, StepStatus.COMPLETE
        )


class VerifiedValidationTests(unittest.TestCase):
    """``validate(numerical_uncertainty="verified")`` widens only u_val."""

    def test_u_num_enters_asme_only(self) -> None:
        """u_val grows to sqrt(u_val0^2 + u_num^2); the verdict is unchanged."""
        plain = UQStudy(spring_model(), seed=6)
        base = plain.validate("x", campaign(), tolerance=0.3, sample_size=400)
        self.assertNotIn("u_num", plain.latest("validation").summary)

        verified = UQStudy(spring_model(), seed=6)
        verified.verify("x", GRID_VALUES, 2.0)
        u_num = verified.numerical_uncertainty("x")
        result = verified.validate(
            "x",
            campaign(),
            tolerance=0.3,
            sample_size=400,
            numerical_uncertainty="verified",
        )
        self.assertEqual(result.consistent, base.consistent)
        self.assertEqual(result.u_pooling, base.u_pooling)
        for with_num, without in zip(result.experiments, base.experiments, strict=True):
            self.assertEqual(with_num.area, without.area)
            self.assertEqual(with_num.asme.numerical_uncertainty, u_num)
            self.assertAlmostEqual(
                with_num.asme.validation_uncertainty,
                math.hypot(without.asme.validation_uncertainty, u_num),
                places=12,
            )
        summary = verified.latest("validation").summary
        self.assertEqual(summary["u_num"], u_num)
        self.assertEqual(summary["u_num.source"], "verify stage")
        base_summary = dict(plain.latest("validation").summary)
        trimmed = {
            k: v for k, v in summary.items() if k not in ("u_num", "u_num.source")
        }
        self.assertEqual(trimmed, base_summary)


if __name__ == "__main__":
    unittest.main()
