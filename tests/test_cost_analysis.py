"""Tests for probstat.cost against closed forms and SciPy-derived oracles.

Literal reference values marked "scipy" were computed once with SciPy 1.18 in
a separate verification environment (scipy.integrate.quad,
scipy.optimize.minimize_scalar / brentq, scipy.stats); they are pasted here
so the test suite itself needs only the standard library.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from probstat.cost import (
    CostElement,
    DesignAlternative,
    FailureCostItem,
    LearningCurve,
    LifeCycleCostModel,
    age_replacement_cost_rate,
    annuity_factor,
    capital_recovery_factor,
    compare_alternatives,
    continuous_rate,
    cost_effectiveness_ratio,
    cost_for_joint_confidence,
    cost_of_unreliability,
    cost_risk_analysis,
    discount_factor,
    equivalent_annual_cost,
    escalate,
    expected_backorders,
    expected_corrective_cost,
    expected_failure_cost,
    fit_total_distribution,
    incremental_cost_effectiveness,
    joint_confidence_level,
    learning_exponent,
    life_cycle_cost_risk,
    method_of_moments_total,
    minimal_repair_overhaul,
    minimum_spares,
    net_present_value,
    nominal_rate,
    optimal_spares,
    optimize_age_replacement,
    optimize_block_replacement,
    optimize_periodic_overhaul,
    present_value,
    protection_level,
    real_rate,
    reliability_decision_problem,
    renewal_function,
    spare_demand_mean,
    spare_plan_cost,
)
from probstat.distributions import Exponential, Gamma, LogNormal, Normal, Weibull
from probstat.modeling import RandomInput


class DiscountingTests(unittest.TestCase):
    """Time-value-of-money identities."""

    def test_discount_factors_and_npv(self) -> None:
        """Discrete and continuous factors agree through log(1 + r)."""
        self.assertAlmostEqual(discount_factor(0.05, 2.0), 1.0 / 1.05**2, places=15)
        self.assertAlmostEqual(
            discount_factor(continuous_rate(0.05), 2.0, compounding="continuous"),
            discount_factor(0.05, 2.0),
            places=15,
        )
        npv = net_present_value([-1000.0, 300.0, 400.0, 500.0], 0.1)
        exact = -1000 + 300 / 1.1 + 400 / 1.1**2 + 500 / 1.1**3
        self.assertAlmostEqual(npv, exact, places=10)
        self.assertAlmostEqual(present_value(100.0, 0.1, 1.0), 100 / 1.1)
        with self.assertRaises(ValueError):
            discount_factor(-1.0, 1.0)

    def test_annuity_and_equivalent_annual_cost(self) -> None:
        """Annuity factor equals the sum of discount factors."""
        direct = math.fsum(1.07**-k for k in range(1, 11))
        self.assertAlmostEqual(annuity_factor(0.07, 10), direct, places=13)
        self.assertEqual(annuity_factor(0.0, 10), 10.0)
        self.assertAlmostEqual(
            capital_recovery_factor(0.07, 10) * direct, 1.0, places=13
        )
        self.assertAlmostEqual(
            equivalent_annual_cost(1000.0, 0.07, 10), 1000.0 / direct, places=10
        )
        self.assertAlmostEqual(annuity_factor(1e-12, 5), 5.0, places=9)

    def test_fisher_relation_round_trip(self) -> None:
        """(1 + n) = (1 + r)(1 + i) exactly."""
        r = real_rate(0.08, 0.03)
        self.assertAlmostEqual(r, 1.08 / 1.03 - 1.0, places=15)
        self.assertAlmostEqual(nominal_rate(r, 0.03), 0.08, places=15)
        self.assertAlmostEqual(escalate(100.0, 0.03, 2.0), 100.0 * 1.03**2)


def _model() -> LifeCycleCostModel:
    """Return a small CBS with one fixed and three random elements."""
    return LifeCycleCostModel(
        (
            CostElement.spread(
                "development",
                Normal(1000.0, 100.0),
                {0.0: 0.4, 1.0: 0.6},
                category="research_development",
            ),
            CostElement.one_time(
                "production", LogNormal(6.0, 0.2), 2.0, category="acquisition"
            ),
            CostElement.recurring(
                "operations",
                Normal(50.0, 5.0),
                3.0,
                7.0,
                escalation=0.02,
                category="operation",
            ),
            CostElement.one_time("disposal", 80.0, 8.0, category="disposal"),
        ),
        discount_rate=0.05,
    )


class LifeCycleCostTests(unittest.TestCase):
    """Breakdowns, coefficients, moments, and the probabilistic model."""

    def test_breakdown_matches_hand_calculation(self) -> None:
        """Totals at the means reproduce a direct cash-flow sum."""
        model = _model()
        breakdown = model.deterministic()
        production = math.exp(6.0 + 0.02)
        flows = [(0.0, 400.0), (1.0, 600.0), (2.0, production), (8.0, 80.0)] + [
            (float(t), 50.0 * 1.02**t) for t in range(3, 8)
        ]
        pv = math.fsum(a / 1.05**t for t, a in flows)
        self.assertAlmostEqual(breakdown.total_present_value, pv, places=9)
        self.assertAlmostEqual(
            breakdown.total_undiscounted, math.fsum(a for _, a in flows), places=9
        )
        self.assertAlmostEqual(breakdown.by_category["disposal"], 80.0 / 1.05**8)
        self.assertEqual(
            tuple(breakdown.by_category),
            ("research_development", "acquisition", "operation", "disposal"),
        )
        self.assertAlmostEqual(math.fsum(breakdown.by_time_present_value.values()), pv)

    def test_moments_are_exact_for_linear_totals(self) -> None:
        """Mean equals the deterministic total; variance is a' Sigma a."""
        model = _model()
        a = model.coefficients()
        rho = ((1.0, 0.5, 0.0), (0.5, 1.0, 0.3), (0.0, 0.3, 1.0))
        mean, variance = model.moments(rho)
        self.assertAlmostEqual(
            mean, model.deterministic().total_present_value, places=9
        )
        sd = [100.0, math.sqrt(LogNormal(6.0, 0.2).variance), 5.0]
        weights = [a["development"], a["production"], a["operations"]]
        expected = math.fsum(
            weights[i] * weights[j] * rho[i][j] * sd[i] * sd[j]
            for i in range(3)
            for j in range(3)
        )
        self.assertAlmostEqual(variance, expected, places=6)

    def test_probabilistic_model_evaluates_breakdown(self) -> None:
        """The model's QoIs equal the breakdown at the same realization."""
        model = _model()
        probabilistic = model.as_probabilistic_model()
        point = (900.0, 450.0, 55.0)
        out = probabilistic.evaluate(point)
        direct = model.evaluate(
            dict(zip(("development", "production", "operations"), point, strict=True))
        )
        self.assertAlmostEqual(out["lcc_present_value"], direct.total_present_value)
        self.assertAlmostEqual(out["pv_disposal"], 80.0 / 1.05**8)
        self.assertEqual(probabilistic.fixed["disposal"], 80.0)

    def test_uq_study_propagates_and_ranks_costs(self) -> None:
        """UQStudy on the CBS: mean matches; Sobol indices match a_i^2 s_i^2 / Var."""
        from probstat.workflow import UQStudy

        model = LifeCycleCostModel(
            (
                CostElement.one_time("a", Normal(100.0, 30.0), 0.0),
                CostElement.recurring("b", Normal(20.0, 2.0), 1.0, 5.0),
                CostElement.one_time("c", Normal(50.0, 10.0), 3.0),
            ),
            discount_rate=0.05,
        )
        study = UQStudy(model.as_probabilistic_model(), seed=4)
        propagation = study.propagate(2048, design="sobol")
        mean, variance = model.moments()
        self.assertLess(abs(propagation["lcc_present_value"].mean / mean - 1.0), 1e-3)
        sobol = study.sensitivity("lcc_present_value", 1024, bootstrap=20)
        a = model.coefficients()
        shares = {
            "a": (a["a"] * 30.0) ** 2 / variance,
            "b": (a["b"] * 2.0) ** 2 / variance,
            "c": (a["c"] * 10.0) ** 2 / variance,
        }
        self.assertEqual(sobol.ranking()[0], "a")
        for name, first in zip(sobol.input_names, sobol.first_order, strict=True):
            self.assertLess(abs(first.estimate - shares[name]), 0.05)

    def test_validation(self) -> None:
        """Bad categories, repeated names, and bad fractions are rejected."""
        with self.assertRaises(ValueError):
            CostElement("x", 1.0, category="nonsense")
        with self.assertRaises(ValueError):
            CostElement.spread("x", 1.0, {0.0: 0.5, 1.0: 0.4})
        with self.assertRaises(ValueError):
            LifeCycleCostModel((CostElement("x", 1.0), CostElement("x", 2.0)))
        with self.assertRaises(TypeError):
            CostElement("x", True)  # type: ignore[arg-type]


class CostRiskTests(unittest.TestCase):
    """Method of moments, Monte Carlo S-curve, contingency, and JCL."""

    def test_method_of_moments_matches_exact_formula(self) -> None:
        """Var = sum_ij rho_ij s_i s_j; the independent part is smaller."""
        total = method_of_moments_total(
            [100.0, 200.0, 50.0],
            [10.0, 30.0, 5.0],
            [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]],
        )
        self.assertEqual(total.mean, 350.0)
        # 100 + 900 + 25 + 2 (0.5*300 + 0.3*50 + 0.2*150) = 1415.
        self.assertAlmostEqual(total.variance, 1415.0, places=10)
        self.assertAlmostEqual(total.independent_variance, 1025.0, places=10)
        self.assertGreater(total.spread_understatement, 0.14)
        with self.assertRaises(ValueError):
            method_of_moments_total([1.0, 1.0], [1.0, 1.0], [[1, 2], [2, 1]])

    def test_normal_elements_monte_carlo_matches_analytic(self) -> None:
        """For normal elements the total is exactly normal (scipy quantiles)."""
        elements = [
            RandomInput("a", Normal(100.0, 10.0)),
            RandomInput("b", Normal(200.0, 30.0)),
            RandomInput("c", Normal(50.0, 5.0)),
        ]
        rho = [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]]
        result = cost_risk_analysis(
            elements, rho, family="normal", samples=20_000, seed=3, point_estimate=340.0
        )
        # scipy.stats.norm(350, sqrt(1415)).ppf and norm(350, sqrt(1025)).ppf.
        oracle = {
            0.7: (369.726104412, 366.789008165),
            0.8: (381.658833138, 376.945026597),
            0.9: (398.2074662, 391.029669454),
        }
        sd = math.sqrt(1415.0)
        for row in result.rows:
            if row.level in oracle:
                self.assertAlmostEqual(row.analytic, oracle[row.level][0], places=6)
                self.assertAlmostEqual(
                    row.analytic_independent, oracle[row.level][1], places=6
                )
            # Quantile standard error ~ sqrt(q(1-q)/n)/phi; allow 4 such errors.
            z = Normal().quantile(row.level)
            se = math.sqrt(row.level * (1 - row.level) / 20_000) / Normal().pdf(z) * sd
            self.assertLess(abs(row.monte_carlo - row.analytic), 4.0 * se)
            self.assertAlmostEqual(row.contingency_analytic, row.analytic - 340.0)
        self.assertLess(
            abs(result.monte_carlo_mean - 350.0),
            4.0 * result.monte_carlo_standard_error,
        )
        self.assertLess(abs(result.monte_carlo_standard_deviation / sd - 1.0), 0.03)
        analytic_p, mc_p = result.confidence_of_point_estimate()
        self.assertAlmostEqual(analytic_p, Normal(350.0, sd).cdf(340.0), places=12)
        self.assertLess(abs(mc_p - analytic_p), 0.02)
        self.assertEqual(len(result.s_curve(11)), 11)

    def test_omitting_correlation_understates_spread(self) -> None:
        """Independent Monte Carlo has a visibly smaller P90 than correlated."""
        elements = [RandomInput(f"e{k}", LogNormal(3.0, 0.4)) for k in range(5)]
        rho = [[1.0 if i == j else 0.6 for j in range(5)] for i in range(5)]
        correlated = cost_risk_analysis(elements, rho, samples=8000, seed=5)
        independent = cost_risk_analysis(elements, None, samples=8000, seed=5)
        self.assertGreater(
            correlated.rows[-1].monte_carlo, independent.rows[-1].monte_carlo
        )
        self.assertGreater(
            correlated.monte_carlo_standard_deviation,
            1.3 * independent.monte_carlo_standard_deviation,
        )
        self.assertTrue(any("understates" in n for n in independent.notes))

    def test_lognormal_fit_and_life_cycle_wrapper(self) -> None:
        """Moment-matched lognormal reproduces mean and sd; scipy P80 oracle."""
        law = fit_total_distribution(350.0, math.sqrt(1415.0))
        self.assertAlmostEqual(law.mean, 350.0, places=9)
        self.assertAlmostEqual(law.variance, 1415.0, places=6)
        # scipy.stats.lognorm(...).ppf(0.8) for the moment-matched law.
        self.assertAlmostEqual(law.quantile(0.8), 380.842143741, places=6)
        result = life_cycle_cost_risk(_model(), samples=2000, seed=1)
        mean, variance = _model().moments()
        self.assertAlmostEqual(result.moments.mean, mean, places=9)
        self.assertAlmostEqual(result.moments.variance, variance, places=6)

    def test_joint_confidence_level(self) -> None:
        """Hand-counted JCL on a tiny joint sample."""
        costs = [1.0, 2.0, 3.0, 4.0, 5.0]
        schedules = [5.0, 1.0, 4.0, 2.0, 3.0]
        jcl = joint_confidence_level(costs, schedules, 3.0, 4.0)
        self.assertEqual(jcl.joint, 0.4)  # (2,1), (3,4)
        self.assertEqual(jcl.cost_only, 0.6)
        self.assertEqual(jcl.schedule_only, 0.8)
        self.assertEqual(cost_for_joint_confidence(costs, schedules, 4.0, 0.6), 4.0)
        self.assertIsNone(cost_for_joint_confidence(costs, schedules, 2.0, 0.6))


class LearningCurveTests(unittest.TestCase):
    """Wright and Crawford theories."""

    def test_exponent_and_exact_sums(self) -> None:
        """Exponent log2(0.85); numpy sums give the oracle totals."""
        self.assertAlmostEqual(learning_exponent(0.85), -0.234465253637023, places=14)
        unit = LearningCurve(1000.0, 0.85)
        self.assertAlmostEqual(unit.cumulative_cost(100), 43753.8671392, places=5)
        self.assertAlmostEqual(unit.lot_cost(11, 30), 9974.59574268, places=5)
        self.assertAlmostEqual(unit.unit_cost(2), 850.0, places=10)
        wright = LearningCurve(1000.0, 0.85, "cumulative_average")
        self.assertAlmostEqual(wright.cumulative_cost(100), 33967.9621368, places=5)
        self.assertAlmostEqual(wright.unit_cost(10), 451.627920172, places=6)
        self.assertAlmostEqual(wright.cumulative_average_cost(2), 850.0, places=10)
        self.assertAlmostEqual(
            wright.lot_cost(11, 30),
            wright.cumulative_cost(30) - wright.cumulative_cost(10),
        )

    def test_midpoint_approximation_is_flagged_and_close(self) -> None:
        """Approximation error is reported and small away from the first unit."""
        unit = LearningCurve(1000.0, 0.85)
        later = unit.lot_midpoint_approximation(11, 30)
        self.assertEqual(later.exact_cost, unit.lot_cost(11, 30))
        self.assertLess(abs(later.relative_error), 1e-3)
        self.assertAlmostEqual(
            later.approximate_cost,
            20 * 1000.0 * later.lot_midpoint ** learning_exponent(0.85),
        )
        first = unit.lot_midpoint_approximation(1, 10)
        self.assertGreater(abs(first.relative_error), abs(later.relative_error))
        with self.assertRaises(ValueError):
            LearningCurve(
                1000.0, 0.85, "cumulative_average"
            ).lot_midpoint_approximation(1, 2)


class MaintenanceTests(unittest.TestCase):
    """Replacement policies against closed forms and SciPy oracles."""

    def test_exponential_age_replacement_has_no_finite_optimum(self) -> None:
        """Constant hazard: C(T) decreases to c_f lambda."""
        result = optimize_age_replacement(Exponential(2.0), 1.0, 10.0)
        self.assertFalse(result.finite_optimum)
        self.assertEqual(result.interval, math.inf)
        self.assertAlmostEqual(result.cost_rate, 20.0)
        # Closed form: lambda c_f + lambda c_p e^{-lambda T}/(1 - e^{-lambda T}).
        t = 0.3
        closed = 2.0 * 10.0 + 2.0 * math.exp(-0.6) / (1.0 - math.exp(-0.6))
        self.assertAlmostEqual(
            age_replacement_cost_rate(Exponential(2.0), t, 1.0, 10.0), closed, places=9
        )

    def test_weibull_age_replacement_matches_scipy(self) -> None:
        """Weibull(2, 1), c_p = 1, c_f = 10 (scipy minimize_scalar and brentq)."""
        result = optimize_age_replacement(Weibull(2.0, 1.0), 1.0, 10.0)
        self.assertTrue(result.finite_optimum)
        self.assertAlmostEqual(result.interval, 0.336451191255, places=7)
        self.assertAlmostEqual(result.cost_rate, 6.0561214426, places=9)
        self.assertAlmostEqual(
            result.run_to_failure_cost_rate, 10.0 / (math.sqrt(math.pi) / 2)
        )
        self.assertAlmostEqual(
            age_replacement_cost_rate(Weibull(2.0, 1.0), 0.5, 1.0, 10.0),
            6.48366811288334,
            places=9,
        )
        # Optimality condition h(T) int_0^T R - F(T) = c_p / (c_f - c_p).
        t = result.interval
        integral = math.sqrt(math.pi) / 2 * math.erf(t)
        self.assertAlmostEqual(
            2 * t * integral - (1 - math.exp(-t * t)), 1.0 / 9.0, places=7
        )
        scaled = optimize_age_replacement(Weibull(3.0, 1000.0), 500.0, 5000.0)
        self.assertAlmostEqual(scaled.interval, 382.45553275, delta=1e-4)
        self.assertAlmostEqual(scaled.cost_rate, 1.97467514958, places=8)

    def test_renewal_function_closed_forms(self) -> None:
        """Erlang-2: M(t) = t/2 - 1/4 + e^{-2t}/4; exponential: M(t) = t."""
        erlang = renewal_function(Gamma(2.0, 1.0), 5.0, steps=200)
        worst = max(
            abs(v - (t / 2 - 0.25 + math.exp(-2 * t) / 4))
            for t, v in zip(erlang.times, erlang.values, strict=True)
        )
        self.assertLess(worst, 1e-5)
        self.assertLess(worst, erlang.error_estimate)
        self.assertAlmostEqual(erlang(1.0), 0.2838338208091532, places=5)
        exponential = renewal_function(Exponential(1.0), 4.0, steps=200)
        self.assertLess(
            max(
                abs(v - t)
                for t, v in zip(exponential.times, exponential.values, strict=True)
            ),
            1e-4,
        )

    def test_block_replacement(self) -> None:
        """Erlang-2 optimum (scipy, closed-form M); exponential has none."""
        result = optimize_block_replacement(Gamma(2.0, 1.0), 1.0, 10.0)
        self.assertTrue(result.finite_optimum)
        self.assertAlmostEqual(result.interval, 0.688210679771, delta=result.resolution)
        self.assertAlmostEqual(result.cost_rate, 3.73759760237, delta=1e-4)
        self.assertFalse(
            optimize_block_replacement(Exponential(1.0), 1.0, 10.0).finite_optimum
        )

    def test_minimal_repair_closed_form_and_numeric(self) -> None:
        """T* = eta (c_p/((beta-1) c_m))^(1/beta) agrees with numerics."""
        closed = minimal_repair_overhaul(2.5, 100.0, 5000.0, 800.0)
        self.assertAlmostEqual(closed.interval, 176.976459799, places=6)
        self.assertAlmostEqual(closed.cost_rate, 47.0872416749, places=8)
        numeric = optimize_periodic_overhaul(
            lambda t: (t / 100.0) ** 2.5, 5000.0, 800.0, upper=1000.0
        )
        self.assertAlmostEqual(numeric.interval, closed.interval, places=4)
        self.assertAlmostEqual(numeric.cost_rate, closed.cost_rate, places=9)
        self.assertFalse(
            minimal_repair_overhaul(1.0, 100.0, 5000.0, 800.0).finite_optimum
        )

    def test_expected_corrective_cost(self) -> None:
        """Power-law intensity: Lambda(H) = (H/eta)^beta; PV from scipy quad."""
        result = expected_corrective_cost(
            lambda t: 2.5 / 100 * (t / 100) ** 1.5, 300.0, 800.0, discount_rate=0.05
        )
        self.assertAlmostEqual(result.expected_failures, 3.0**2.5, places=9)
        self.assertAlmostEqual(result.expected_cost, 800 * 3.0**2.5, places=6)
        self.assertAlmostEqual(result.present_value, 47.5592261298, places=8)


class SparesTests(unittest.TestCase):
    """Poisson spares against scipy.stats.poisson and brute-force sums."""

    def test_protection_and_backorders(self) -> None:
        """CDF and EBO match scipy and an explicit truncated sum."""
        cases = [
            (3.2, 5, 0.894591894530823, 0.174999437238739),
            (3.2, 0, 0.0407622039783662, 3.2),
            (50.0, 60, 0.927839820186743, 0.283641809151241),
            (0.4, 2, 0.992073668132746, 0.00876811048553432),
        ]
        for mu, s, cdf, ebo in cases:
            self.assertAlmostEqual(protection_level(mu, s), cdf, places=12)
            self.assertAlmostEqual(expected_backorders(mu, s), ebo, places=12)
        brute = math.fsum(
            (k - 5) * math.exp(-3.2) * 3.2**k / math.factorial(k) for k in range(6, 120)
        )
        self.assertAlmostEqual(expected_backorders(3.2, 5), brute, places=13)

    def test_minimum_and_optimal_spares(self) -> None:
        """Minimum stock equals scipy poisson.ppf; optimum is the fractile."""
        for mu, q, s in [
            (3.2, 0.95, 6),
            (50.0, 0.99, 67),
            (0.4, 0.9, 1),
            (1000.0, 0.999, 1099),
        ]:
            self.assertEqual(minimum_spares(mu, q), s)
        self.assertEqual(spare_demand_mean(1e-4, 8, 4000.0), 3.2)
        plan = optimal_spares(3.2, 100.0, 2000.0)
        self.assertEqual(plan.spares, 6)
        costs = [spare_plan_cost(3.2, s, 100.0, 2000.0).total_cost for s in range(15)]
        self.assertEqual(costs.index(min(costs)), 6)
        self.assertEqual(optimal_spares(3.2, 100.0, 50.0).spares, 0)


class UnreliabilityTests(unittest.TestCase):
    """Expected failure cost and trade-offs."""

    def test_cost_of_unreliability(self) -> None:
        """Campbell: PV = lambda C (1 - e^{-rH}) / r."""
        result = cost_of_unreliability(
            [
                FailureCostItem("pump", 0.2, 5000.0),
                FailureCostItem("valve", 0.05, 20000.0),
            ],
            10.0,
            discount_rate=0.04,
        )
        self.assertAlmostEqual(result.cost_rate, 2000.0)
        self.assertAlmostEqual(result.expected_cost, 20000.0)
        self.assertAlmostEqual(
            result.present_value, 2000.0 * (1 - math.exp(-0.4)) / 0.04
        )
        self.assertAlmostEqual(result.shares["pump"], 0.5)
        self.assertAlmostEqual(expected_failure_cost([0.1, 0.01], [100.0, 1e4]), 110.0)

    def test_tradeoff_and_evpi_hook(self) -> None:
        """Ranking by LCC + risk cost; EVPI equals a hand computation."""
        cheap = DesignAlternative("cheap", 100.0, 20.0, 0.5, 40.0)
        robust = DesignAlternative("robust", 160.0, 20.0, 0.05, 40.0)
        ranked = compare_alternatives([cheap, robust], 5.0)
        # cheap: 120 + 0.5*40*5 = 220; robust: 180 + 0.05*40*5 = 190.
        self.assertEqual([r.name for r in ranked], ["robust", "cheap"])
        self.assertAlmostEqual(ranked[0].total_cost, 190.0)
        self.assertAlmostEqual(ranked[1].risk_cost, 100.0)
        problem = reliability_decision_problem(
            [cheap, robust],
            ("benign", "harsh"),
            (0.7, 0.3),
            {"cheap": (0.1, 1.5), "robust": (0.05, 0.1)},
            5.0,
        )
        # Losses: cheap (140, 420), robust (190, 200); expected 224 vs 193.
        self.assertEqual(problem.optimal_decision(), "robust")
        self.assertAlmostEqual(
            problem.expected_value_of_perfect_information(),
            193.0 - (0.7 * 140 + 0.3 * 200),
        )
        self.assertAlmostEqual(cost_effectiveness_ratio(100.0, 0.8), 125.0)
        self.assertAlmostEqual(
            incremental_cost_effectiveness(100.0, 0.8, 130.0, 0.9), 300.0
        )

    def test_fraction_exactness_of_npv(self) -> None:
        """NPV at r = 0 equals the exact rational sum of flows."""
        flows = [0.1, 0.2, 0.3]
        self.assertEqual(
            Fraction(net_present_value(flows, 0.0)), Fraction(math.fsum(flows))
        )


if __name__ == "__main__":
    unittest.main()
