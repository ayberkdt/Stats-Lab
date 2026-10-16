"""End-to-end mission assessment against independent closed forms."""

from __future__ import annotations

import math
import unittest

from probstat.cost import CostElement, LifeCycleCostModel
from probstat.distributions import Beta, Gamma, Triangular
from probstat.modeling import RandomInput
from probstat.reliability import demand_failure_probability, failure_rate_estimate
from probstat.risk import Risk, RiskRegister
from probstat.systems import (
    DemandComponent,
    ExponentialComponent,
    FixedProbability,
    MarkovComponent,
    SystemModel,
    parallel,
    series,
    standby_system,
)
from probstat.workflow import MissionAssessment, MissionAssessmentResult

LAM_A, LAM_B, LAM_C = 2e-4, 3e-4, 1e-4
T = 500.0


def _exact_reliability(t: float) -> float:
    """Return R(t) of series(parallel(A, B), C) with exponential lives."""
    ra, rb, rc = (math.exp(-lam * t) for lam in (LAM_A, LAM_B, LAM_C))
    return (1.0 - (1.0 - ra) * (1.0 - rb)) * rc


def _system() -> SystemModel:
    """Return the reference series-parallel mission system."""
    return SystemModel(
        series(parallel("A", "B"), "C"),
        {
            "A": ExponentialComponent(LAM_A),
            "B": ExponentialComponent(LAM_B),
            "C": ExponentialComponent(LAM_C),
        },
    )


class MissionReliabilityTests(unittest.TestCase):
    """Exact reliability, MTTF, cut sets and importance."""

    def test_reliability_matches_closed_form(self) -> None:
        """R(T), Q(T) and the MTTF equal the closed forms."""
        assessment = MissionAssessment(_system(), T, name="ref", seed=7)
        result = assessment.reliability()
        self.assertAlmostEqual(result.reliability, _exact_reliability(T), 14)
        self.assertAlmostEqual(result.unreliability, 1.0 - _exact_reliability(T), 14)
        # MTTF = int R = 1/(a+c) + 1/(b+c) - 1/(a+b+c).
        mttf = (
            1.0 / (LAM_A + LAM_C)
            + 1.0 / (LAM_B + LAM_C)
            - 1.0 / (LAM_A + LAM_B + LAM_C)
        )
        assert result.mttf is not None
        self.assertAlmostEqual(result.mttf.value / mttf, 1.0, 8)
        self.assertEqual(result.single_points_of_failure, ("C",))
        self.assertEqual(
            set(result.minimal_cut_sets), {frozenset({"C"}), frozenset({"A", "B"})}
        )

    def test_importance_ranks_single_point_of_failure_first(self) -> None:
        """Birnbaum importance of C is the reliability of the parallel block."""
        ranked = MissionAssessment(_system(), T).importance()
        self.assertEqual(ranked[0].component, "C")
        ra, rb = math.exp(-LAM_A * T), math.exp(-LAM_B * T)
        self.assertAlmostEqual(ranked[0].birnbaum, 1.0 - (1.0 - ra) * (1.0 - rb), 13)


class MissionUncertaintyTests(unittest.TestCase):
    """Epistemic propagation through exact system evaluations."""

    def test_beta_epistemic_mean_matches_exact_moment(self) -> None:
        """For independent Beta reliabilities, E[R] equals R at the means."""
        system = SystemModel(
            series(parallel("A", "B"), "C"),
            {name: FixedProbability(0.9) for name in "ABC"},
        )
        uncertain = {
            "A": RandomInput("pA", Beta(18.0, 2.0)),
            "B": RandomInput("pB", Beta(9.0, 1.0)),
            "C": RandomInput("pC", Beta(95.0, 5.0)),
        }
        assessment = MissionAssessment(system, 1.0, uncertain=uncertain, seed=11)
        unc = assessment.uncertainty(4000, sensitivity_samples=256)
        # Multilinearity and independence: E[R] = (1-(1-mA)(1-mB)) mC.
        exact = (1.0 - (1.0 - 0.9) * (1.0 - 0.9)) * 0.95
        standard_error = unc.standard_deviation / math.sqrt(4000)
        self.assertLess(abs(unc.mean - exact), 4.0 * standard_error)
        self.assertLessEqual(unc.mean_interval[0], exact)
        self.assertGreaterEqual(unc.mean_interval[1], exact)
        # The exact epistemic variance from the BDD moment recursion.
        moments = system.structure.reliability_moments(
            {"A": 0.9, "B": 0.9, "C": 0.95},
            {
                "A": 18.0 * 2.0 / (20.0**2 * 21.0),
                "B": 9.0 * 1.0 / (10.0**2 * 11.0),
                "C": 95.0 * 5.0 / (100.0**2 * 101.0),
            },
        )
        self.assertAlmostEqual(moments.mean, exact, 14)
        self.assertLess(
            abs(unc.standard_deviation - math.sqrt(moments.variance)),
            0.06 * math.sqrt(moments.variance),
        )
        self.assertEqual(unc.ranking[0], "pC")

    def test_rate_posteriors_from_field_data(self) -> None:
        """Gamma posteriors from field data drive the rate uncertainty."""
        estimate = failure_rate_estimate(2, 10_000.0)
        assessment = MissionAssessment(
            _system(),
            T,
            uncertain={
                "A": estimate.random_input("lamA"),
                "C": RandomInput("lamC", Gamma(2.5, 25_000.0)),
            },
            seed=3,
        )
        unc = assessment.uncertainty(1500, sensitivity_samples=256)
        self.assertTrue(0.0 < unc.percentiles[0][1] <= unc.percentiles[-1][1] <= 1.0)
        self.assertEqual(unc.evaluations, 1500 + 256 * 4)

    def test_requires_uncertain_inputs(self) -> None:
        """Without uncertain inputs the epistemic step refuses to run."""
        with self.assertRaises(ValueError):
            MissionAssessment(_system(), T).uncertainty(10)


class MissionCostAndRiskTests(unittest.TestCase):
    """Cost of unreliability, life-cycle cost and the register."""

    def test_loss_cost_and_campaign_probability(self) -> None:
        """Loss frequency, campaign probability and undiscounted loss cost."""
        assessment = MissionAssessment(
            _system(),
            T,
            missions_per_year=4.0,
            horizon=10.0,
            mission_loss_cost=2.0e6,
        )
        cost = assessment.cost()
        q = 1.0 - _exact_reliability(T)
        self.assertAlmostEqual(cost.loss_frequency, 4.0 * q, 14)
        self.assertAlmostEqual(
            cost.probability_of_any_loss, 1.0 - (1.0 - q) ** 40, 12
        )
        self.assertAlmostEqual(
            cost.unreliability.expected_cost / (4.0 * q * 10.0 * 2.0e6), 1.0, 12
        )
        self.assertIsNone(cost.cost_risk)

    def test_life_cycle_and_register_integrate(self) -> None:
        """Life-cycle PV, S-curve and register combine in one result."""
        lcc = LifeCycleCostModel(
            (
                CostElement.one_time(
                    "build", Triangular(8e6, 9e6, 12e6), category="acquisition"
                ),
                CostElement.recurring(
                    "ops", 4e5, 1.0, 10.0, category="operation"
                ),
            )
        )
        register = RiskRegister(
            (Risk("supplier delay", 0.2, 5e5, category="schedule"),)
        )
        assessment = MissionAssessment(
            _system(),
            T,
            missions_per_year=2.0,
            horizon=10.0,
            mission_loss_cost=1e7,
            life_cycle=lcc,
            risks=register,
            seed=5,
        )
        result = assessment.run()
        self.assertIsInstance(result, MissionAssessmentResult)
        expected_lcc = (8e6 + 9e6 + 12e6) / 3.0 + 10 * 4e5
        self.assertAlmostEqual(
            result.cost.life_cycle_present_value / expected_lcc, 1.0, 12
        )
        self.assertIsNotNone(result.cost.cost_risk)
        names = [risk.name for risk in result.register.risks]
        self.assertEqual(names, ["supplier delay", "mission loss"])
        q = 1.0 - _exact_reliability(T)
        self.assertAlmostEqual(
            result.register.expected_exposure(), 0.2 * 5e5 + q * 1e7, 6
        )
        text = assessment.report(result)
        for heading in ("Mission reliability", "Importance", "Cost", "Risk register"):
            self.assertIn(heading, text)


class HierarchicalMissionTests(unittest.TestCase):
    """Markov subsystems and demand components inside one mission."""

    def test_markov_standby_and_demand_block(self) -> None:
        """A cold-standby CTMC block in series with a demand component."""
        lam = 1e-3
        p_demand = demand_failure_probability(1, 200).posterior.mean
        system = SystemModel(
            series("pumps", "igniter"),
            {
                "pumps": MarkovComponent(
                    standby_system(spares=1, failure_rate=lam)
                ),
                "igniter": DemandComponent(p_demand),
            },
        )
        result = MissionAssessment(system, T).reliability()
        expected = math.exp(-lam * T) * (1.0 + lam * T) * (1.0 - p_demand)
        self.assertAlmostEqual(result.reliability, expected, 12)


if __name__ == "__main__":
    unittest.main()
