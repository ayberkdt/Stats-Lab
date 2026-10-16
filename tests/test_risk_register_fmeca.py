"""Tests for FMECA, the risk register, the MIL-STD-882E matrix, and F-N curves."""

from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction

from probstat.distributions import Normal
from probstat.risk import (
    MIL_STD_882E_TABLE,
    Branch,
    DiscreteImpact,
    EventTree,
    FailureMode,
    Fmeca,
    FNCriterion,
    Heading,
    Outcome,
    Risk,
    RiskRegister,
    fn_curve,
    fn_curve_from_event_trees,
    hazard_risk_level,
    mil_std_882e_risk_matrix,
    register_from_rows,
    risk_assessment_code,
    risk_priority_number,
)


def _worksheet() -> Fmeca:
    """Return a two-item FMECA worksheet."""
    return Fmeca(
        (
            FailureMode(
                "pump",
                "seal leak",
                3,
                2e-5,
                0.6,
                0.5,
                1000.0,
                severity_rank=5,
                occurrence_rank=6,
                detection_rank=3,
            ),
            FailureMode(
                "pump",
                "seizure",
                2,
                2e-5,
                0.4,
                1.0,
                1000.0,
                severity_rank=8,
                occurrence_rank=3,
                detection_rank=4,
            ),
            FailureMode(
                "valve",
                "fails closed",
                2,
                5e-6,
                0.7,
                1.0,
                1000.0,
                severity_rank=8,
                occurrence_rank=2,
                detection_rank=6,
            ),
            FailureMode("valve", "external leak", 3, 5e-6, 0.3, 0.1, 1000.0),
        )
    )


class FmecaTests(unittest.TestCase):
    """MIL-STD-1629A criticality numbers and RPN."""

    def test_mode_and_item_criticality(self) -> None:
        """C_m = beta alpha lambda t and C_r sums within a severity class."""
        sheet = _worksheet()
        modes = dict(((m.item, m.mode), c) for m, c in sheet.mode_criticality())
        self.assertAlmostEqual(modes[("pump", "seal leak")], 0.5 * 0.6 * 2e-5 * 1000)
        self.assertAlmostEqual(modes[("pump", "seizure")], 0.008)
        rows = sheet.item_criticality()
        self.assertEqual(
            [(r.item, r.severity_class) for r in rows][:2], [("pump", 2), ("valve", 2)]
        )
        self.assertAlmostEqual(rows[0].criticality, 0.008)
        self.assertAlmostEqual(rows[1].criticality, 0.0035)
        self.assertEqual(sheet.mode_ratio_sums()["pump"], 1.0)
        self.assertEqual(sheet.inconsistent_items(), ())

    def test_criticality_matrix_and_rpn(self) -> None:
        """Matrix binning uses the analyst's edges; RPN ranking is stable."""
        matrix = _worksheet().criticality_matrix((1e-3, 5e-3))
        self.assertEqual(matrix.cells[(2, 2)], ("pump",))
        self.assertEqual(matrix.cells[(2, 1)], ("valve",))
        self.assertEqual(matrix.count(3, 2), 1)  # pump seal leak 0.006
        self.assertEqual(matrix.count(1, 0), 0)
        ranking = _worksheet().rpn_ranking()
        self.assertEqual([r for _, r in ranking], [96, 96, 90])
        self.assertEqual(ranking[0][0].mode, "seizure")
        with self.assertRaises(ValueError):
            FailureMode("x", "y", 5, 1.0, 0.5, 0.5, 1.0)
        with self.assertRaises(TypeError):
            risk_priority_number(True, 2, 3)  # type: ignore[arg-type]

    def test_rpn_takes_only_120_distinct_values(self) -> None:
        """Enumerating all 1000 rank triples reproduces the Bowles observation."""
        values = {
            risk_priority_number(s, o, d)
            for s, o, d in itertools.product(range(1, 11), repeat=3)
        }
        self.assertEqual(len(values), 120)
        self.assertEqual(max(values), 1000)


class HazardMatrixTests(unittest.TestCase):
    """MIL-STD-882E Table III encoding."""

    def test_every_cell_matches_the_table(self) -> None:
        """All 24 RACs classify to the documented risk level."""
        matrix = mil_std_882e_risk_matrix()
        for level, row in MIL_STD_882E_TABLE.items():
            for severity, expected in zip((1, 2, 3, 4), row, strict=True):
                self.assertEqual(hazard_risk_level(matrix, severity, level), expected)
        self.assertEqual(hazard_risk_level(matrix, 1, "C"), "High")
        self.assertEqual(hazard_risk_level(matrix, 4, "E"), "Low")
        self.assertEqual(risk_assessment_code(1, "C"), "1C")
        with self.assertRaises(ValueError):
            risk_assessment_code(5, "A")

    def test_custom_table(self) -> None:
        """A tailored table replaces the default."""
        table = {k: ("X", "Y", "Y", "Z") for k in "ABCDEF"}
        matrix = mil_std_882e_risk_matrix(table)
        self.assertEqual(hazard_risk_level(matrix, 1, "A"), "X")
        self.assertEqual(hazard_risk_level(matrix, 4, "F"), "Z")
        with self.assertRaises(ValueError):
            mil_std_882e_risk_matrix({"A": ("X", "Y", "Y", "Z")})


def _register() -> RiskRegister:
    """Return three independent risks, two with mitigation."""
    return RiskRegister(
        (
            Risk(
                "supplier", 0.3, 100.0, mitigation_cost=5.0, mitigated_probability=0.1
            ),
            Risk(
                "test",
                0.2,
                DiscreteImpact((50.0, 200.0), (0.5, 0.5)),
                mitigation_cost=10.0,
                mitigated_impact=60.0,
            ),
            Risk("late", 0.5, 40.0, category="schedule"),
        )
    )


def _brute_force(register: RiskRegister) -> dict[Fraction, Fraction]:
    """Enumerate every occurrence and impact combination in exact arithmetic."""
    laws = []
    for risk in register.risks:
        p = Fraction(risk.probability)
        impact = risk.impact
        atoms = (
            [(Fraction(impact), Fraction(1))]
            if isinstance(impact, float)
            else [
                (Fraction(v), Fraction(q))
                for v, q in zip(impact.values, impact.probabilities, strict=True)
            ]  # type: ignore[union-attr]
        )
        laws.append([(Fraction(0), 1 - p)] + [(v, p * q) for v, q in atoms])
    result: dict[Fraction, Fraction] = {}
    for combination in itertools.product(*laws):
        total = sum((v for v, _ in combination), Fraction(0))
        weight = math.prod(w for _, w in combination)
        result[total] = result.get(total, Fraction(0)) + weight
    return result


class RiskRegisterTests(unittest.TestCase):
    """Exact moments, convolution, Monte Carlo, ranking, and waterfall."""

    def test_moments_and_exact_distribution(self) -> None:
        """Convolution equals brute-force enumeration; moments are exact."""
        register = _register()
        # E = 0.3*100 + 0.2*125 + 0.5*40 = 75.
        self.assertAlmostEqual(register.expected_exposure(), 75.0)
        # Var = sum p E[I^2] - p^2 E[I]^2.
        var = (
            (0.3 * 1e4 - 0.09 * 1e4)
            + (0.2 * 21250 - 0.04 * 125**2)
            + (0.5 * 1600 - 0.25 * 1600)
        )
        self.assertAlmostEqual(register.exposure_variance(), var)
        exact = register.exposure_distribution()
        oracle = _brute_force(register)
        self.assertEqual(len(exact.values), len(oracle))
        for v, p in zip(exact.values, exact.probabilities, strict=True):
            self.assertAlmostEqual(p, float(oracle[Fraction(v)]), places=15)
        self.assertAlmostEqual(exact.mean, 75.0, places=12)
        self.assertAlmostEqual(exact.variance, var, places=9)
        self.assertAlmostEqual(
            exact.exceedance(200.0), float(sum(p for v, p in oracle.items() if v > 200))
        )
        # P(0) = P(40) = 0.28, so the median is 40.
        self.assertEqual(exact.quantile(0.5), 40.0)

    def test_monte_carlo_for_continuous_impacts(self) -> None:
        """Simulated mean and variance agree with the exact moments."""
        register = RiskRegister(
            (Risk("a", 0.4, Normal(100.0, 20.0)), Risk("b", 0.1, 500.0))
        )
        samples = register.simulate(40_000, seed=11)
        mean = math.fsum(samples) / len(samples)
        var = math.fsum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
        se = math.sqrt(register.exposure_variance() / len(samples))
        self.assertLess(abs(mean - register.expected_exposure()), 4 * se)
        self.assertLess(abs(var / register.exposure_variance() - 1.0), 0.05)
        with self.assertRaises(TypeError):
            register.exposure_distribution()

    def test_ranking_roi_and_waterfall(self) -> None:
        """Reductions add exactly and are ordered by ROI."""
        register = _register()
        self.assertEqual(
            [r.name for r in register.ranking()], ["supplier", "test", "late"]
        )
        supplier, test, late = register.risks
        self.assertAlmostEqual(supplier.risk_reduction, 20.0)
        self.assertAlmostEqual(supplier.mitigation_roi, 4.0)
        self.assertAlmostEqual(test.mitigation_roi, 1.3)
        self.assertTrue(math.isnan(late.mitigation_roi))
        steps = register.mitigation_waterfall()
        self.assertEqual([s.risk for s in steps], ["supplier", "test"])
        self.assertAlmostEqual(
            steps[-1].exposure_after, register.expected_exposure(mitigated=True)
        )
        self.assertAlmostEqual(steps[-1].cumulative_cost, 15.0)
        self.assertAlmostEqual(register.totals_by("category")["schedule"], 20.0)
        rebuilt = register_from_rows([{"name": "x", "probability": 0.5, "impact": 2.0}])
        self.assertEqual(rebuilt.expected_exposure(), 1.0)

    def test_support_cap(self) -> None:
        """A tiny cap forces the documented error."""
        with self.assertRaises(ValueError):
            _register().exposure_distribution(max_support=3)


class FNCurveTests(unittest.TestCase):
    """F-N curves and criterion comparison."""

    def test_hand_computed_curve(self) -> None:
        """F(N) sums frequencies of scenarios with at least N fatalities."""
        curve = fn_curve(
            [(1e-3, 1), (1e-4, 10), (1e-5, 100), (2e-4, 10), (5e-2, 0)],
            FNCriterion(1e-3, 2.0),
        )
        self.assertEqual([p.consequence for p in curve.points], [1.0, 10.0, 100.0])
        self.assertAlmostEqual(curve.points[0].frequency, 1.31e-3)
        self.assertAlmostEqual(curve.points[1].frequency, 3.1e-4)
        self.assertAlmostEqual(curve.frequency(5.0), 3.1e-4)
        self.assertEqual(curve.frequency(101.0), 0.0)
        self.assertAlmostEqual(curve.expected_consequence, 1e-3 + 3e-3 + 1e-3)
        self.assertFalse(curve.compliant)
        self.assertAlmostEqual(curve.max_ratio, 100.0)
        lenient = fn_curve([(1e-4, 1), (1e-6, 10)], FNCriterion(1e-3, 1.0))
        self.assertTrue(lenient.compliant)

    def test_from_event_tree(self) -> None:
        """Event-tree sequences feed the curve with their frequencies."""
        tree = EventTree(
            "fire",
            0.01,
            Heading(
                "sprinkler",
                (
                    Branch("works", 0.9, Outcome("minor", 0.0)),
                    Branch("fails", 0.1, Outcome("major", 3.0)),
                ),
            ),
        )
        curve = fn_curve_from_event_trees([tree])
        self.assertEqual(len(curve.points), 1)
        self.assertAlmostEqual(curve.points[0].frequency, 0.001)
        self.assertIsNone(curve.compliant)


if __name__ == "__main__":
    unittest.main()
