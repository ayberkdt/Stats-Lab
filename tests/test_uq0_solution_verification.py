"""Tests for GCI, least-squares order, and solution-verification statements."""

from __future__ import annotations

import math
import unittest

from probstat.uq import (
    ComputationalExperiment,
    ConvergenceType,
    QuantityOfInterest,
    SolutionVerification,
    classify_convergence,
    grid_convergence_index,
    least_squares_convergence,
    refinement_study,
    representative_spacing,
    run_computational_experiment,
    verify_solution,
)
from probstat.validation import CredibilityRecord, VerificationEvidence


def manufactured(order: float, coefficient: float = 0.5, exact: float = 1.0):
    """Return ``phi(h) = exact + coefficient h^order`` as a callable."""

    def phi(h: float) -> float:
        """Evaluate the manufactured sequence at one spacing."""
        return exact + coefficient * h**order

    return phi


def trapezoid_exp(cells: int) -> float:
    """Return the composite trapezoid integral of ``exp`` on ``[0, 1]``."""
    h = 1.0 / cells
    interior = math.fsum(math.exp(i * h) for i in range(1, cells))
    return h * (0.5 * (1.0 + math.e) + interior)


def trapezoid_model(parameters):
    """Forward model integrating ``exp(a x)`` on ``[0, 1]`` with spacing ``h``."""
    a = parameters["a"]
    cells = round(1.0 / parameters["h"])
    h = 1.0 / cells
    interior = math.fsum(math.exp(a * i * h) for i in range(1, cells))
    return {"integral": h * (0.5 * (1.0 + math.exp(a)) + interior)}


class GridConvergenceIndexTests(unittest.TestCase):
    """Three-grid GCI on manufactured sequences of known order."""

    def test_uniform_refinement_recovers_order_and_limit(self) -> None:
        """A pure h^2 sequence gives p=2, phi_ext=1, and GCI = 1.25 |error|."""
        phi = manufactured(2.0)
        spacings = (0.1, 0.05, 0.025)
        result = grid_convergence_index(spacings, [phi(h) for h in spacings])
        self.assertAlmostEqual(result.order, 2.0, places=9)
        self.assertEqual(result.order_solver, "fixed-point")
        self.assertEqual(result.order_iterations, 1)
        self.assertAlmostEqual(result.extrapolated, 1.0, places=12)
        true_error = 0.5 * 0.025**2
        self.assertAlmostEqual(result.gci_fine, 1.25 * true_error, places=14)
        self.assertEqual(result.convergence, ConvergenceType.MONOTONE)
        self.assertAlmostEqual(result.asymptotic_ratio_at(2.0), 1.0, places=9)
        self.assertAlmostEqual(
            result.asymptotic_ratio, result.values[0] / result.values[1], places=12
        )
        self.assertTrue(result.in_asymptotic_range(0.01, order=2.0))
        self.assertAlmostEqual(
            result.extrapolated_relative_error, true_error / 1.0, places=12
        )

    def test_non_uniform_ratios_use_fixed_point_iteration(self) -> None:
        """Different r21 and r32 still recover the manufactured order."""
        phi = manufactured(2.0)
        spacings = (0.07, 0.1, 0.04)
        result = grid_convergence_index(spacings, [phi(h) for h in spacings])
        self.assertEqual(result.spacings, (0.04, 0.07, 0.1))
        self.assertAlmostEqual(result.refinement_ratios[0], 1.75)
        self.assertAlmostEqual(result.refinement_ratios[1], 1.0 / 0.7)
        self.assertEqual(result.order_solver, "fixed-point")
        self.assertGreater(result.order_iterations, 1)
        self.assertAlmostEqual(result.order, 2.0, places=8)
        self.assertAlmostEqual(result.extrapolated, 1.0, places=10)

    def test_fractional_order_with_non_uniform_ratios(self) -> None:
        """A p=1.5 sequence on non-uniform grids is recovered."""
        phi = manufactured(1.5, coefficient=-0.8, exact=2.0)
        spacings = (0.2, 0.13, 0.05)
        result = grid_convergence_index(spacings, [phi(h) for h in spacings])
        self.assertAlmostEqual(result.order, 1.5, places=8)
        self.assertAlmostEqual(result.extrapolated, 2.0, places=10)
        self.assertAlmostEqual(result.asymptotic_ratio_at(1.5), 1.0, places=8)
        self.assertGreater(abs(result.asymptotic_ratio_at(2.0) - 1.0), 0.1)

    def test_brent_fallback_when_fixed_point_is_not_contractive(self) -> None:
        """A large r21 with tiny r32 still yields the correct order."""
        phi = manufactured(2.0)
        spacings = (0.025, 0.03, 0.1)
        result = grid_convergence_index(spacings, [phi(h) for h in spacings])
        self.assertEqual(result.order_solver, "brent")
        self.assertAlmostEqual(result.order, 2.0, places=8)

    def test_real_trapezoid_rule_on_non_uniform_grids(self) -> None:
        """The second-order trapezoid rule is verified with a covering GCI."""
        cells = (8, 12, 20)
        spacings = [1.0 / n for n in cells]
        values = [trapezoid_exp(n) for n in cells]
        result = grid_convergence_index(spacings, values)
        exact = math.e - 1.0
        self.assertAlmostEqual(result.order, 2.0, delta=0.01)
        self.assertGreater(result.gci_fine, abs(values[2] - exact))
        self.assertLess(abs(result.extrapolated - exact), 1e-6)
        self.assertIsNotNone(result.gci_fine_relative)

    def test_oscillatory_sequence_is_detected(self) -> None:
        """Alternating differences are classified and flagged as oscillatory."""
        spacings = (0.025, 0.05, 0.1)
        values = (0.99, 1.02, 1.0)
        self.assertEqual(
            classify_convergence(spacings, values), ConvergenceType.OSCILLATORY
        )
        result = grid_convergence_index(spacings, values)
        self.assertTrue(result.oscillatory)
        self.assertLess(result.difference_ratio, 0.0)
        non_uniform = grid_convergence_index((0.04, 0.07, 0.1), values)
        self.assertTrue(non_uniform.oscillatory)
        self.assertGreater(non_uniform.order, 0.0)

    def test_divergent_sequence_is_rejected(self) -> None:
        """Differences that grow under refinement give no GCI."""
        spacings = (0.025, 0.05, 0.1)
        values = (1.0, 1.01, 1.015)
        self.assertEqual(
            classify_convergence(spacings, values), ConvergenceType.DIVERGENT
        )
        with self.assertRaises(ValueError):
            grid_convergence_index(spacings, values)
        with self.assertRaises(ValueError):
            verify_solution("q", spacings, values)

    def test_invalid_studies_raise(self) -> None:
        """Malformed refinement data are rejected explicitly."""
        with self.assertRaises(ValueError):
            grid_convergence_index((0.1, 0.05), (1.0, 1.1))
        with self.assertRaises(ValueError):
            grid_convergence_index((0.1, 0.1, 0.05), (1.0, 1.1, 1.2))
        with self.assertRaises(ValueError):
            grid_convergence_index((0.1, -0.05, 0.02), (1.0, 1.1, 1.2))
        with self.assertRaises(ValueError):
            grid_convergence_index((0.1, 0.05, 0.025), (1.0, 1.0, 1.2))
        with self.assertRaises(ValueError):
            grid_convergence_index(
                (0.1, 0.05, 0.025), (1.3, 1.1, 1.05), safety_factor=0.9
            )
        with self.assertRaises(TypeError):
            grid_convergence_index("abc", (1.0, 1.1, 1.2))
        with self.assertRaises(ValueError):
            classify_convergence((0.1, 0.05, 0.025, 0.01), (1.0, 1.1, 1.2, 1.3))

    def test_representative_spacing(self) -> None:
        """``h = (V/N)^(1/d)`` for a cubic mesh."""
        self.assertAlmostEqual(representative_spacing(1000, dimension=3), 0.1)
        self.assertAlmostEqual(
            representative_spacing(400, dimension=2, volume=4.0), 0.1
        )
        with self.assertRaises(ValueError):
            representative_spacing(10, dimension=4)
        with self.assertRaises(ValueError):
            representative_spacing(0, dimension=1)


class LeastSquaresConvergenceTests(unittest.TestCase):
    """Least-squares observed order from four or more grids."""

    spacings = (0.4, 0.2, 0.1, 0.05, 0.025)

    def test_exact_fractional_order(self) -> None:
        """Noise-free p=1.5 data are fitted exactly with a 1.25 factor."""
        phi = manufactured(1.5, coefficient=0.3)
        fit = least_squares_convergence(self.spacings, [phi(h) for h in self.spacings])
        self.assertAlmostEqual(fit.order, 1.5, places=6)
        self.assertAlmostEqual(fit.extrapolated, 1.0, places=9)
        self.assertLess(fit.residual_standard_deviation, 1e-10)
        self.assertTrue(fit.credible_order)
        self.assertEqual(fit.safety_factor, 1.25)
        self.assertAlmostEqual(fit.fine_error, 0.3 * 0.025**1.5, places=10)
        self.assertAlmostEqual(fit.uncertainty, 1.25 * abs(fit.fine_error), places=9)
        self.assertEqual(fit.grid_count, 5)

    def test_scattered_data_include_the_scatter_term(self) -> None:
        """Noisy second-order data keep U >= the true finest-grid error."""
        noise = (0.0004, -0.0003, 0.0002, -0.0001, 0.00005)
        values = [
            1.0 + 0.3 * h**2 + e for h, e in zip(self.spacings, noise, strict=True)
        ]
        fit = least_squares_convergence(self.spacings, values)
        self.assertGreater(fit.residual_standard_deviation, 0.0)
        self.assertGreater(fit.uncertainty, abs(values[-1] - 1.0))
        self.assertGreaterEqual(
            fit.uncertainty, fit.residual_standard_deviation
        )

    def test_order_above_credible_range_uses_conservative_rule(self) -> None:
        """A p=3 fit with p_th=2 switches to F_s=3 and the fixed-order refit."""
        phi = manufactured(3.0, coefficient=2.0)
        fit = least_squares_convergence(
            self.spacings, [phi(h) for h in self.spacings], theoretical_order=2.0
        )
        self.assertAlmostEqual(fit.order, 3.0, places=5)
        self.assertFalse(fit.credible_order)
        self.assertEqual(fit.safety_factor, 3.0)
        expected = 3.0 * max(abs(fit.fine_error), abs(fit.fixed_order_error)) + max(
            fit.residual_standard_deviation,
            fit.fixed_order_residual_standard_deviation,
        )
        self.assertAlmostEqual(fit.uncertainty, expected, places=12)

    def test_invalid_least_squares_inputs(self) -> None:
        """Too few grids, constant data, and bad bounds are rejected."""
        with self.assertRaises(ValueError):
            least_squares_convergence((0.1, 0.05, 0.025), (1.0, 1.1, 1.2))
        with self.assertRaises(ValueError):
            least_squares_convergence(self.spacings, [1.0] * 5)
        with self.assertRaises(ValueError):
            least_squares_convergence(
                self.spacings, [1.0, 1.1, 1.2, 1.3, 1.4], order_bounds=(2.0, 1.0)
            )


class SolutionVerificationTests(unittest.TestCase):
    """Numerical-uncertainty statements consumed by validation."""

    def test_three_grid_statement_and_conversions(self) -> None:
        """A monotone study yields phi_1 +/- GCI and explicit u_num rules."""
        phi = manufactured(2.0)
        spacings = (0.1, 0.05, 0.025)
        statement = verify_solution("q", spacings, [phi(h) for h in spacings])
        self.assertIn("GCI", statement.method)
        self.assertTrue(statement.reliable)
        low, high = statement.interval
        self.assertLess(low, 1.0)
        self.assertGreater(high, 1.0)
        u = statement.expanded_uncertainty
        self.assertAlmostEqual(statement.standard_uncertainty(), u / math.sqrt(3.0))
        self.assertAlmostEqual(
            statement.standard_uncertainty("normal95"), u / 1.959963984540054
        )
        with self.assertRaises(ValueError):
            statement.standard_uncertainty("triangular")  # type: ignore[arg-type]
        self.assertEqual(statement.error_budget().discretization, u)
        self.assertIn("monotone convergence", statement.summary())

    def test_oscillatory_statement_uses_half_range_bound(self) -> None:
        """Oscillation gives U = 3 (max - min)/2 and no Richardson estimate."""
        statement = verify_solution("q", (0.025, 0.05, 0.1), (0.99, 1.02, 1.0))
        self.assertEqual(statement.convergence, ConvergenceType.OSCILLATORY)
        self.assertAlmostEqual(statement.expanded_uncertainty, 3.0 * 0.5 * 0.03)
        self.assertIsNone(statement.observed_order)
        self.assertIsNone(statement.extrapolated)
        self.assertFalse(statement.reliable)
        self.assertTrue(statement.warnings)

    def test_pre_asymptotic_grids_are_warned(self) -> None:
        """A strong higher-order term triggers the asymptotic-range warning."""
        spacings = (0.4, 0.2, 0.1)
        values = [1.0 + h**2 + 5.0 * h**3 for h in spacings]
        statement = verify_solution("q", spacings, values)
        self.assertFalse(statement.reliable)
        self.assertIn("asymptotic", statement.warnings[0])

    def test_four_grids_use_least_squares(self) -> None:
        """Four or more grids are analysed by the least-squares fit."""
        spacings = (0.4, 0.2, 0.1, 0.05)
        phi = manufactured(2.0)
        statement = verify_solution("q", spacings, [phi(h) for h in spacings])
        self.assertIn("least-squares", statement.method)
        self.assertEqual(statement.grid_count, 4)
        self.assertAlmostEqual(statement.observed_order or 0.0, 2.0, places=5)
        wobbly = verify_solution("q", spacings, (1.08, 1.02, 1.025, 1.004))
        self.assertEqual(wobbly.convergence, ConvergenceType.OSCILLATORY)
        self.assertIn("non-monotone", wobbly.warnings[0])

    def test_statement_feeds_credibility_record(self) -> None:
        """Verification evidence built from the statement keeps U and the method."""
        phi = manufactured(2.0)
        spacings = (0.1, 0.05, 0.025)
        statement = verify_solution("q", spacings, [phi(h) for h in spacings])
        evidence = VerificationEvidence.from_solution_verification(statement)
        self.assertAlmostEqual(
            evidence.standard_uncertainty, statement.standard_uncertainty("uniform")
        )
        self.assertEqual(evidence.observed_order, statement.observed_order)
        summary = CredibilityRecord(verification=(evidence,)).summary()
        self.assertIn("three-grid GCI", summary)
        with self.assertRaises(TypeError):
            VerificationEvidence.from_solution_verification("q")  # type: ignore[arg-type]

    def test_statement_validation(self) -> None:
        """Invalid statement fields are rejected."""
        with self.assertRaises(ValueError):
            SolutionVerification(
                "q", "m", 0.1, 1.0, -0.1, ConvergenceType.MONOTONE, 2.0, 1.0, 1.25, 3
            )
        with self.assertRaises(TypeError):
            SolutionVerification(
                "q", "m", 0.1, 1.0, 0.1, "monotone", 2.0, 1.0, 1.25, 3  # type: ignore[arg-type]
            )


class ComputationalExperimentTests(unittest.TestCase):
    """Simulations recorded as computational experiments."""

    quantity = QuantityOfInterest("integral", "integral of exp(a x) on [0, 1]")

    def test_run_records_parameters_settings_and_outputs(self) -> None:
        """Settings stay separate from physical parameters in the record."""
        run = run_computational_experiment(
            trapezoid_model,
            (self.quantity,),
            {"a": 1.0},
            settings={"h": 0.1},
            name="coarse",
            model_name="trapezoid",
            purpose="verification",
        )
        self.assertEqual(dict(run.parameters), {"a": 1.0})
        self.assertEqual(dict(run.settings), {"h": 0.1})
        self.assertAlmostEqual(run.outputs["integral"], trapezoid_exp(10), places=12)
        self.assertIn("settings: h=0.1", run.describe())
        with self.assertRaises(ValueError):
            run_computational_experiment(
                trapezoid_model,
                (self.quantity,),
                {"a": 1.0, "h": 0.1},
                settings={"h": 0.1},
            )
        with self.assertRaises(ValueError):
            ComputationalExperiment("x", "m", {"a": 1.0}, {"a": 2.0}, {"y": 1.0})

    def test_refinement_study_verifies_the_simulation(self) -> None:
        """A refinement experiment gives a GCI that covers the true error."""
        study = refinement_study(
            trapezoid_model,
            self.quantity,
            {"a": 1.0},
            (0.125, 1.0 / 12.0, 0.05),
            model_name="trapezoid",
        )
        self.assertEqual(len(study.experiments), 3)
        self.assertEqual(study.spacings, (0.125, 1.0 / 12.0, 0.05))
        verification = study.verification
        self.assertEqual(verification.convergence, ConvergenceType.MONOTONE)
        self.assertAlmostEqual(verification.observed_order or 0.0, 2.0, delta=0.01)
        exact = math.e - 1.0
        low, high = verification.interval
        self.assertLess(low, exact)
        self.assertGreater(high, exact)
        self.assertTrue(
            all("grid refinement" in run.purpose for run in study.experiments)
        )
        with self.assertRaises(ValueError):
            refinement_study(
                trapezoid_model,
                self.quantity,
                {"a": 1.0},
                (0.1, 0.05, 0.025),
                settings={"h": 0.2},
            )


if __name__ == "__main__":
    unittest.main()
