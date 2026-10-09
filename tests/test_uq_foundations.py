"""Tests for uncertainty vocabulary, forward models, and verification tools."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from probstat.uq import (
    NumericalErrorBudget,
    QuantityOfInterest,
    UncertaintyNature,
    UncertaintySource,
    UncertaintySourceKind,
    evaluate_model,
    observed_order_of_accuracy,
    richardson_extrapolation,
)


class UQTaxonomyTests(unittest.TestCase):
    """Verify that uncertainty nature and engineering source remain distinct."""

    def test_source_records_both_classification_axes(self) -> None:
        """One parameter may be epistemic without conflating the two axes."""
        source = UncertaintySource(
            name="drag_coefficient",
            description="Calibration uncertainty in the drag coefficient.",
            nature=UncertaintyNature.EPISTEMIC,
            kind=UncertaintySourceKind.PARAMETRIC,
            unit=None,
        )
        self.assertEqual(source.nature, UncertaintyNature.EPISTEMIC)
        self.assertEqual(source.kind, UncertaintySourceKind.PARAMETRIC)
        with self.assertRaises(FrozenInstanceError):
            source.name = "changed"  # type: ignore[misc]

    def test_invalid_descriptive_fields_are_rejected(self) -> None:
        """Identifiers, descriptions, units, and enums are validated eagerly."""
        with self.assertRaises(ValueError):
            QuantityOfInterest(" ", "Maximum displacement", "m")
        with self.assertRaises(ValueError):
            QuantityOfInterest("u_max", "Maximum displacement", " ")
        with self.assertRaises(TypeError):
            UncertaintySource(
                "load",
                "Applied load variability",
                "aleatory",  # type: ignore[arg-type]
                UncertaintySourceKind.PARAMETRIC,
            )

    def test_forward_model_extracts_finite_named_qois(self) -> None:
        """Validated named inputs produce an immutable selected-output map."""

        def projectile(parameters: dict[str, float]) -> dict[str, float]:
            speed = parameters["speed"]
            angle = parameters["angle"]
            return {
                "range": speed**2 * math.sin(2.0 * angle) / 9.81,
                "flight_time": 2.0 * speed * math.sin(angle) / 9.81,
            }

        qoi = QuantityOfInterest("range", "Horizontal landing distance", "m")
        result = evaluate_model(
            projectile,
            {"speed": 20.0, "angle": math.pi / 4.0},
            [qoi],
        )
        self.assertAlmostEqual(result["range"], 400.0 / 9.81)
        with self.assertRaises(TypeError):
            result["range"] = 0.0  # type: ignore[index]

    def test_forward_model_contract_rejects_invalid_boundaries(self) -> None:
        """Missing, repeated, non-finite, and malformed values fail clearly."""
        qoi = QuantityOfInterest("response", "Scalar response")
        cases = (
            (lambda _: {}, {"x": 1.0}, [qoi], ValueError),
            (lambda _: {"response": math.inf}, {"x": 1.0}, [qoi], ValueError),
            (lambda _: {"response": 1.0}, {"x": math.nan}, [qoi], ValueError),
            (lambda _: {"response": 1.0}, {"x": 1.0}, [qoi, qoi], ValueError),
            (lambda _: 1.0, {"x": 1.0}, [qoi], TypeError),
        )
        for model, parameters, quantities, error_type in cases:
            with self.subTest(error_type=error_type):
                with self.assertRaises(error_type):
                    evaluate_model(model, parameters, quantities)  # type: ignore[arg-type]


class NumericalVerificationTests(unittest.TestCase):
    """Check refinement equations and conservative error accounting."""

    def test_error_budget_is_componentwise_and_additive(self) -> None:
        """The total bound is the stable sum of non-negative components."""
        budget = NumericalErrorBudget(0.1, 0.01, 0.2, 0.03)
        self.assertAlmostEqual(budget.total_bound, 0.34)
        with self.assertRaises(ValueError):
            NumericalErrorBudget(roundoff=-0.01)
        with self.assertRaises(ValueError):
            NumericalErrorBudget(iterative=math.inf)

    def test_second_order_sequence_recovers_order_and_limit(self) -> None:
        """A known quadratic error sequence yields order two and its limit."""
        coarse, medium, fine = 1.16, 1.04, 1.01
        order = observed_order_of_accuracy(coarse, medium, fine)
        self.assertAlmostEqual(order, 2.0)
        extrapolated = richardson_extrapolation(
            medium,
            fine,
            refinement_ratio=2.0,
            order=order,
        )
        self.assertAlmostEqual(extrapolated, 1.0)

    def test_refinement_validation_rejects_undefined_cases(self) -> None:
        """Invalid ratios, orders, flat sequences, and overflow are explicit."""
        with self.assertRaises(ValueError):
            observed_order_of_accuracy(3.0, 2.0, 1.0, refinement_ratio=1.0)
        with self.assertRaises(ValueError):
            observed_order_of_accuracy(1.0, 1.0, 0.5)
        with self.assertRaises(ValueError):
            richardson_extrapolation(1.1, 1.0, refinement_ratio=2.0, order=0.0)
        with self.assertRaises(OverflowError):
            richardson_extrapolation(
                1.1,
                1.0,
                refinement_ratio=1e300,
                order=2.0,
            )


if __name__ == "__main__":
    unittest.main()
