"""Tests for quantitative risk analysis: measures, trees, fragility, decisions."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.risk import (
    BasicEvent,
    Branch,
    DecisionProblem,
    EventTree,
    FaultTree,
    Gate,
    Heading,
    LognormalFragility,
    LossFunction,
    Outcome,
    RiskMatrix,
    and_gate,
    annual_failure_frequency,
    assess_risk,
    binary_heading,
    bootstrap_risk_interval,
    conditional_value_at_risk,
    custom_loss,
    exceedance_probability,
    expected_loss,
    expected_shortfall,
    fit_lognormal_fragility,
    k_of_n_gate,
    linear_loss,
    lognormal_conditional_value_at_risk,
    lognormal_value_at_risk,
    loss_exceedance_curve,
    normal_conditional_value_at_risk,
    normal_value_at_risk,
    or_gate,
    piecewise_linear_loss,
    poisson_exceedance_probability,
    power_law_failure_frequency,
    return_period,
    step_loss,
    value_at_risk,
)
from probstat.uq import QuantityOfInterest


def _textbook_tree() -> FaultTree:
    """Return TOP = (A and B) or (C and (A or D)) with repeated event A."""
    a, b, c, d = (
        BasicEvent(name, p)
        for name, p in zip("ABCD", (0.1, 0.2, 0.3, 0.4), strict=True)
    )
    return FaultTree(
        or_gate("TOP", and_gate("G1", a, b), and_gate("G2", c, or_gate("G3", a, d)))
    )


class SampleRiskMeasureTests(unittest.TestCase):
    """Check VaR, CVaR, expected loss, and exceedance on samples."""

    def test_hand_computed_var_and_cvar(self) -> None:
        """Small samples reproduce the lower-quantile and RU definitions."""
        losses = [5.0, 1.0, 4.0, 2.0, 3.0]
        self.assertEqual(value_at_risk(losses, 0.8), 4.0)
        self.assertAlmostEqual(conditional_value_at_risk(losses, 0.8), 5.0)
        self.assertEqual(value_at_risk(losses, 0.7), 4.0)
        # Worst 30%: 0.1 of mass at 4 plus 0.2 at 5 -> (0.4 + 1.0) / 0.3.
        self.assertAlmostEqual(conditional_value_at_risk(losses, 0.7), 14.0 / 3.0)
        self.assertAlmostEqual(expected_loss(losses), 3.0)
        self.assertAlmostEqual(expected_shortfall(losses, 0.7), 14.0 / 3.0)

    def test_cvar_splits_tied_atoms(self) -> None:
        """Ties at VaR contribute only the needed probability fraction."""
        losses = [0.0, 0.0, 0.0, 10.0, 10.0]
        self.assertEqual(value_at_risk(losses, 0.5), 0.0)
        # Worst 50%: 0.4 mass at 10 plus 0.1 mass at 0 -> 4 / 0.5 = 8.
        self.assertAlmostEqual(conditional_value_at_risk(losses, 0.5), 8.0)
        self.assertAlmostEqual(conditional_value_at_risk(losses, 0.9), 10.0)
        # Floating-point level 0.7 * 10 must not skip an order statistic.
        ten = [float(value) for value in range(1, 11)]
        self.assertEqual(value_at_risk(ten, 0.7), 7.0)
        self.assertAlmostEqual(conditional_value_at_risk(ten, 0.7), 9.0)

    def test_weighted_scenarios(self) -> None:
        """Weighted scenarios use cumulative probability mass."""
        losses, weights = [0.0, 100.0, 1000.0], [0.9, 0.08, 0.02]
        self.assertEqual(value_at_risk(losses, 0.95, weights=weights), 100.0)
        self.assertAlmostEqual(
            conditional_value_at_risk(losses, 0.95, weights=weights), 460.0
        )
        self.assertAlmostEqual(expected_loss(losses, weights=weights), 28.0)
        self.assertAlmostEqual(
            exceedance_probability(losses, 100.0, weights=weights), 0.02
        )
        unnormalized = [9.0, 0.8, 0.2]
        self.assertAlmostEqual(
            conditional_value_at_risk(losses, 0.95, weights=unnormalized), 460.0
        )

    def test_exceedance_curve_is_non_increasing(self) -> None:
        """The empirical exceedance curve steps down at observed losses."""
        curve = loss_exceedance_curve([1.0, 2.0, 2.0, 3.0])
        self.assertEqual(curve, ((1.0, 0.75), (2.0, 0.25), (3.0, 0.0)))
        custom = loss_exceedance_curve([1.0, 2.0, 3.0, 4.0], [0.0, 2.5, 10.0])
        self.assertEqual(custom, ((0.0, 1.0), (2.5, 0.5), (10.0, 0.0)))

    def test_normal_and_lognormal_closed_forms(self) -> None:
        """Large seeded samples agree with analytic VaR and CVaR."""
        generator = random.Random(20261011)
        normal = [generator.gauss(1.0, 2.0) for _ in range(100_000)]
        self.assertAlmostEqual(normal_value_at_risk(0.0, 1.0, 0.95), 1.6448536, 6)
        self.assertAlmostEqual(
            normal_conditional_value_at_risk(0.0, 1.0, 0.95), 2.0627128, 6
        )
        self.assertAlmostEqual(
            value_at_risk(normal, 0.95),
            normal_value_at_risk(1.0, 2.0, 0.95),
            delta=0.05,
        )
        self.assertAlmostEqual(
            conditional_value_at_risk(normal, 0.95),
            normal_conditional_value_at_risk(1.0, 2.0, 0.95),
            delta=0.06,
        )
        skewed = [math.exp(generator.gauss(0.0, 0.5)) for _ in range(100_000)]
        for alpha in (0.9, 0.99):
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(
                    value_at_risk(skewed, alpha),
                    lognormal_value_at_risk(0.0, 0.5, alpha),
                    delta=0.05,
                )
                self.assertAlmostEqual(
                    conditional_value_at_risk(skewed, alpha),
                    lognormal_conditional_value_at_risk(0.0, 0.5, alpha),
                    delta=0.08,
                )

    def test_closed_form_cvar_limits(self) -> None:
        """Analytic CVaR exceeds VaR and tends to the mean as alpha -> 0."""
        self.assertGreater(
            lognormal_conditional_value_at_risk(0.0, 1.0, 0.5),
            lognormal_value_at_risk(0.0, 1.0, 0.5),
        )
        self.assertAlmostEqual(
            lognormal_conditional_value_at_risk(0.0, 1.0, 1e-9), math.exp(0.5), 6
        )
        self.assertAlmostEqual(normal_conditional_value_at_risk(3.0, 1.0, 1e-9), 3.0, 6)

    def test_bootstrap_is_seeded_and_brackets_estimate(self) -> None:
        """The bootstrap is reproducible and contains the point estimate."""
        generator = random.Random(5)
        losses = [generator.gauss(0.0, 1.0) for _ in range(1000)]
        first = bootstrap_risk_interval(losses, 0.9, replicates=200, seed=11)
        second = bootstrap_risk_interval(losses, 0.9, replicates=200, seed=11)
        self.assertEqual(first, second)
        self.assertLess(first.lower, first.estimate)
        self.assertLess(first.estimate, first.upper)
        self.assertAlmostEqual(first.estimate, conditional_value_at_risk(losses, 0.9))
        var = bootstrap_risk_interval(
            losses, 0.9, measure="var", replicates=100, seed=1
        )
        self.assertEqual(var.estimate, value_at_risk(losses, 0.9))
        self.assertGreater(var.standard_error, 0.0)

    def test_return_period_helpers(self) -> None:
        """Return period inverts the rate; Poisson gives 1 - exp(-rate t)."""
        self.assertAlmostEqual(return_period(1.0 / 475.0), 475.0)
        self.assertAlmostEqual(
            poisson_exceedance_probability(1.0 / 475.0, 50.0),
            1.0 - math.exp(-50.0 / 475.0),
        )
        self.assertEqual(poisson_exceedance_probability(0.0, 10.0), 0.0)

    def test_invalid_measure_inputs(self) -> None:
        """Invalid levels, weights, and samples raise explicit errors."""
        with self.assertRaises(ValueError):
            value_at_risk([1.0, 2.0], 1.0)
        with self.assertRaises(ValueError):
            conditional_value_at_risk([1.0, 2.0], 0.0)
        with self.assertRaises(ValueError):
            value_at_risk([], 0.5)
        with self.assertRaises(ValueError):
            value_at_risk([1.0, 2.0], 0.5, weights=[1.0])
        with self.assertRaises(ValueError):
            value_at_risk([1.0, 2.0], 0.5, weights=[1.0, -1.0])
        with self.assertRaises(ValueError):
            value_at_risk([1.0, 2.0], 0.5, weights=[0.0, 0.0])
        with self.assertRaises(ValueError):
            value_at_risk([1.0, math.nan], 0.5)
        with self.assertRaises(ValueError):
            bootstrap_risk_interval([1.0, 2.0, 3.0], measure="mean", seed=1)
        with self.assertRaises(ValueError):
            bootstrap_risk_interval([1.0, 2.0, 3.0], replicates=5, seed=1)
        with self.assertRaises(ValueError):
            return_period(0.0)
        with self.assertRaises(ValueError):
            normal_value_at_risk(0.0, 0.0, 0.5)


class ConsequenceModelTests(unittest.TestCase):
    """Check loss functions and model-based risk assessment."""

    def test_loss_function_shapes(self) -> None:
        """Step, hinge, piecewise, and custom losses evaluate as documented."""
        step = step_loss(1.0, 50.0)
        self.assertEqual((step(1.0), step(1.5)), (0.0, 50.0))
        self.assertEqual(step_loss(1.0, 50.0, inclusive=True)(1.0), 50.0)
        hinge = linear_loss(2.0, 10.0, cap=15.0)
        self.assertEqual(hinge.apply([1.0, 2.5, 9.0]), (0.0, 5.0, 15.0))
        piecewise = piecewise_linear_loss([(0.0, 0.0), (1.0, 10.0), (3.0, 30.0)])
        self.assertEqual(piecewise.apply([-1.0, 0.5, 2.0, 5.0]), (0.0, 5.0, 20.0, 30.0))
        self.assertEqual(custom_loss(lambda y: y * y, "square")(3.0), 9.0)
        self.assertEqual(piecewise.kind, "piecewise_linear")

    def test_invalid_loss_functions(self) -> None:
        """Bad knots, negative rates, and non-finite losses are rejected."""
        with self.assertRaises(ValueError):
            piecewise_linear_loss([(0.0, 0.0)])
        with self.assertRaises(ValueError):
            piecewise_linear_loss([(1.0, 0.0), (1.0, 1.0)])
        with self.assertRaises(ValueError):
            linear_loss(0.0, -1.0)
        with self.assertRaises(ValueError):
            custom_loss(lambda y: math.inf)(1.0)
        with self.assertRaises(TypeError):
            LossFunction(3.0)  # type: ignore[arg-type]

    def test_assess_risk_linear_model(self) -> None:
        """Hinge loss of a standard normal QoI matches closed forms."""

        def forward(parameters):  # type: ignore[no-untyped-def]
            """Return the input unchanged as the QoI."""
            return {"y": parameters["x"]}

        model = ProbabilisticModel(
            JointInput.independent(RandomInput("x", Normal(0.0, 1.0))),
            forward,
            (QuantityOfInterest("y", "identity output"),),
        )
        assessment = assess_risk(
            model, "y", linear_loss(0.0, 1.0), 4000, design="lhs", seed=7
        )
        # E[max(X, 0)] = 1 / sqrt(2 pi); tail measures equal the normal ones.
        self.assertAlmostEqual(
            assessment.expected_loss.estimate, 1.0 / math.sqrt(2.0 * math.pi), 2
        )
        self.assertAlmostEqual(
            assessment.value_at_risk[0.95],
            normal_value_at_risk(0.0, 1.0, 0.95),
            delta=0.03,
        )
        self.assertAlmostEqual(
            assessment.conditional_value_at_risk[0.95],
            normal_conditional_value_at_risk(0.0, 1.0, 0.95),
            delta=0.05,
        )
        self.assertAlmostEqual(assessment.probability_of_loss, 0.5, delta=0.01)
        self.assertEqual(assessment.model_evaluations, 4000)
        self.assertEqual(len(assessment.exceedance_curve), 50)
        self.assertAlmostEqual(assessment.exceedance_probability(0.0), 0.5, delta=0.01)
        with self.assertRaises(KeyError):
            assess_risk(model, "z", linear_loss(0.0, 1.0), 10, seed=1)
        with self.assertRaises(ValueError):
            assess_risk(model, "y", linear_loss(0.0, 1.0), 10, levels=(), seed=1)


class FaultTreeTests(unittest.TestCase):
    """Check MOCUS, exact quantification, bounds, and importance."""

    def test_mocus_cut_sets_with_repeated_event(self) -> None:
        """MOCUS absorbs non-minimal rows and reports the repeated event."""
        tree = _textbook_tree()
        self.assertEqual(
            tree.minimal_cut_sets,
            (frozenset("AB"), frozenset("AC"), frozenset("CD")),
        )
        self.assertEqual(tree.repeated_events, ("A",))

    def test_absorption_removes_supersets(self) -> None:
        """A or (A and B) has the single minimal cut set {A}."""
        a, b = BasicEvent("A", 0.1), BasicEvent("B", 0.5)
        tree = FaultTree(or_gate("TOP", a, and_gate("G", a, b)))
        self.assertEqual(tree.minimal_cut_sets, (frozenset("A"),))
        self.assertAlmostEqual(tree.top_event_probability(), 0.1)

    def test_exact_probability_and_bounds(self) -> None:
        """Shannon and inclusion-exclusion agree with the hand result 0.152."""
        tree = _textbook_tree()
        # Condition on A: 0.1 * (1 - 0.8 * 0.7) + 0.9 * (0.3 * 0.4) = 0.152.
        self.assertAlmostEqual(tree.top_event_probability(), 0.152, 12)
        self.assertAlmostEqual(
            tree.top_event_probability(method="inclusion_exclusion"), 0.152, 12
        )
        self.assertAlmostEqual(tree.rare_event_approximation(), 0.17, 12)
        self.assertAlmostEqual(tree.min_cut_upper_bound(), 1.0 - 0.98 * 0.97 * 0.88, 12)
        self.assertLessEqual(tree.top_event_probability(), tree.min_cut_upper_bound())

    def test_k_out_of_n_gate(self) -> None:
        """A 2-out-of-3 gate gives 3 p^2 (1 - p) + p^3."""
        events = [BasicEvent(name, 0.1) for name in "XYZ"]
        tree = FaultTree(k_of_n_gate("TOP", 2, *events))
        self.assertAlmostEqual(tree.top_event_probability(), 0.028, 12)
        self.assertEqual(len(tree.minimal_cut_sets), 3)
        self.assertAlmostEqual(
            tree.top_event_probability(method="inclusion_exclusion"), 0.028, 12
        )

    def test_importance_series_and_parallel(self) -> None:
        """Two-component series/parallel systems match hand formulas."""
        a, b = 0.1, 0.2
        series = FaultTree(or_gate("S", BasicEvent("A", a), BasicEvent("B", b)))
        q = a + b - a * b
        measures = series.importance()["A"]
        self.assertAlmostEqual(measures.birnbaum, 1.0 - b)
        self.assertAlmostEqual(measures.fussell_vesely, a / q)
        self.assertAlmostEqual(measures.risk_achievement_worth, 1.0 / q)
        self.assertAlmostEqual(measures.risk_reduction_worth, q / b)
        parallel = FaultTree(and_gate("P", BasicEvent("A", a), BasicEvent("B", b)))
        measures = parallel.importance()["A"]
        self.assertAlmostEqual(measures.birnbaum, b)
        self.assertAlmostEqual(measures.fussell_vesely, 1.0)
        self.assertAlmostEqual(measures.risk_achievement_worth, 1.0 / a)
        self.assertEqual(measures.risk_reduction_worth, math.inf)

    def test_importance_with_repeated_event(self) -> None:
        """Importance on the textbook tree follows the exact definitions."""
        measures = _textbook_tree().importance()["A"]
        self.assertAlmostEqual(measures.birnbaum, 0.44 - 0.12)
        # FV(A) = P(A) P(B or C) / Q = 0.1 * 0.44 / 0.152.
        self.assertAlmostEqual(measures.fussell_vesely, 0.044 / 0.152)
        self.assertAlmostEqual(measures.risk_achievement_worth, 0.44 / 0.152)
        self.assertAlmostEqual(measures.risk_reduction_worth, 0.152 / 0.12)

    def test_invalid_fault_trees(self) -> None:
        """Conflicting definitions and guards raise explicit errors."""
        with self.assertRaises(ValueError):
            FaultTree(or_gate("T", BasicEvent("A", 0.1), BasicEvent("A", 0.2)))
        with self.assertRaises(ValueError):
            Gate("T", "xor", (BasicEvent("A", 0.1),))
        with self.assertRaises(ValueError):
            k_of_n_gate("T", 3, BasicEvent("A", 0.1), BasicEvent("B", 0.1))
        with self.assertRaises(ValueError):
            or_gate("T")
        with self.assertRaises(ValueError):
            BasicEvent("A", 1.5)
        with self.assertRaises(ValueError):
            FaultTree(
                _textbook_tree().top, max_repeated_events=0
            ).top_event_probability()
        with self.assertRaises(KeyError):
            _textbook_tree().top_event_probability({"Z": 0.1})
        with self.assertRaises(ValueError):
            _textbook_tree().top_event_probability(method="bdd")


class EventTreeTests(unittest.TestCase):
    """Check sequence frequencies, total risk, and ranking."""

    def _fire_tree(self) -> EventTree:
        """Return a fire initiating event with detection and suppression."""
        suppression = binary_heading(
            "suppression", 0.05, Outcome("ok", 0.0), Outcome("minor", 1e4)
        )
        return EventTree(
            "fire",
            0.1,
            binary_heading("detection", 0.01, suppression, Outcome("major", 1e6)),
        )

    def test_sequence_frequencies_and_total_risk(self) -> None:
        """Frequencies multiply along paths and risk sums f * c."""
        tree = self._fire_tree()
        frequencies = {s.outcome: s.frequency for s in tree.sequences()}
        self.assertAlmostEqual(frequencies["ok"], 0.1 * 0.99 * 0.95)
        self.assertAlmostEqual(frequencies["minor"], 0.1 * 0.99 * 0.05)
        self.assertAlmostEqual(frequencies["major"], 0.1 * 0.01)
        self.assertAlmostEqual(tree.total_frequency(), 0.1)
        self.assertAlmostEqual(tree.total_risk(), 1049.5)
        ranked = tree.ranked_sequences()
        self.assertEqual([s.outcome for s in ranked], ["major", "minor", "ok"])
        self.assertAlmostEqual(tree.consequence_exceedance_frequency(1e4), 0.001)
        self.assertEqual(
            [s.outcome for s in tree.ranked_sequences(by="frequency")],
            ["ok", "minor", "major"],
        )

    def test_shared_outcome_labels_aggregate(self) -> None:
        """Paths reaching the same end state add their frequencies."""
        tree = EventTree(
            "ie",
            2.0,
            Heading(
                "h",
                (
                    Branch("a", 0.5, Outcome("safe", 0.0)),
                    Branch("b", 0.3, Outcome("safe", 0.0)),
                    Branch("c", 0.2, Outcome("damage", 5.0)),
                ),
            ),
        )
        self.assertAlmostEqual(tree.outcome_frequencies()["safe"], 1.6)
        self.assertAlmostEqual(tree.total_risk(), 2.0)

    def test_invalid_event_trees(self) -> None:
        """Branch probabilities must sum to one and frequency be non-negative."""
        with self.assertRaises(ValueError):
            Heading(
                "h",
                (
                    Branch("a", 0.5, Outcome("x", 0.0)),
                    Branch("b", 0.4, Outcome("y", 1.0)),
                ),
            )
        with self.assertRaises(ValueError):
            EventTree("ie", -1.0, Outcome("x", 0.0))
        with self.assertRaises(ValueError):
            self._fire_tree().ranked_sequences(by="cost")


class FragilityTests(unittest.TestCase):
    """Check fragility evaluation, MLE fitting, and hazard convolution."""

    def test_fragility_evaluation(self) -> None:
        """The median gives one half and quantiles invert the curve."""
        fragility = LognormalFragility(0.5, 0.4)
        self.assertAlmostEqual(fragility.probability(0.5), 0.5)
        self.assertEqual(fragility.probability(0.0), 0.0)
        self.assertAlmostEqual(
            fragility.probability(fragility.intensity_at(0.84)), 0.84, 12
        )

    def test_mle_recovers_known_parameters(self) -> None:
        """Seeded binary data recover theta = 0.5 and beta = 0.4."""
        truth = LognormalFragility(0.5, 0.4)
        generator = random.Random(7)
        intensities: list[float] = []
        failures: list[bool] = []
        for _ in range(3000):
            level = math.exp(generator.uniform(math.log(0.1), math.log(2.0)))
            intensities.append(level)
            failures.append(generator.random() < truth.probability(level))
        fit = fit_lognormal_fragility(intensities, failures)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.fragility.median, 0.5, delta=0.03)
        self.assertAlmostEqual(fit.fragility.dispersion, 0.4, delta=0.05)
        self.assertEqual(fit.trials, 3000)

    def test_mle_with_multiple_stripes(self) -> None:
        """Exact-expectation stripe counts return the generating parameters."""
        truth = LognormalFragility(1.0, 0.5)
        levels = [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5]
        trials = [10_000] * len(levels)
        failures = [round(10_000 * truth.probability(a)) for a in levels]
        fit = fit_lognormal_fragility(levels, failures, trials=trials)
        self.assertAlmostEqual(fit.fragility.median, 1.0, delta=0.01)
        self.assertAlmostEqual(fit.fragility.dispersion, 0.5, delta=0.01)

    def test_mle_rejects_degenerate_data(self) -> None:
        """No failures, separation, or one intensity have no finite MLE."""
        with self.assertRaises(ValueError):
            fit_lognormal_fragility([0.1, 0.2, 0.3], [0, 0, 0])
        with self.assertRaises(ValueError):
            fit_lognormal_fragility([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
        with self.assertRaises(ValueError):
            fit_lognormal_fragility([0.2, 0.2], [0, 1])
        with self.assertRaises(ValueError):
            fit_lognormal_fragility([0.1, 0.2], [0, 2])
        with self.assertRaises(ValueError):
            fit_lognormal_fragility([0.1, 0.2], [3, 1], trials=[2, 2])

    def test_convolution_matches_power_law_closed_form(self) -> None:
        """Numerical convolution reproduces k0 theta^-k exp(k^2 beta^2 / 2)."""
        fragility = LognormalFragility(0.5, 0.4)
        levels = [10.0 ** (-3.0 + 6.0 * i / 2000) for i in range(2001)]
        hazard = [(a, 1e-4 * a**-3.0) for a in levels]
        result = annual_failure_frequency(fragility, hazard)
        exact = power_law_failure_frequency(1e-4, 3.0, fragility)
        self.assertAlmostEqual(exact, 8e-4 * math.exp(0.72), 15)
        self.assertAlmostEqual(result.annual_frequency / exact, 1.0, delta=1e-3)
        self.assertAlmostEqual(result.return_period, 1.0 / result.annual_frequency)

    def test_invalid_hazard_curves(self) -> None:
        """Hazard levels must increase and rates must not increase."""
        fragility = LognormalFragility(0.5, 0.4)
        with self.assertRaises(ValueError):
            annual_failure_frequency(fragility, [(0.1, 1.0)])
        with self.assertRaises(ValueError):
            annual_failure_frequency(fragility, [(0.2, 1.0), (0.1, 0.5)])
        with self.assertRaises(ValueError):
            annual_failure_frequency(fragility, [(0.1, 0.5), (0.2, 1.0)])
        with self.assertRaises(ValueError):
            annual_failure_frequency(lambda a: 2.0, [(0.1, 1.0), (0.2, 0.5)])


class DecisionAnalysisTests(unittest.TestCase):
    """Check expected values, EVPI, EVSI, and the risk matrix."""

    def _umbrella(self) -> DecisionProblem:
        """Return the classic take-or-leave-the-umbrella problem."""
        return DecisionProblem(
            ("take", "leave"),
            ("rain", "dry"),
            (0.3, 0.7),
            ((70.0, 80.0), (0.0, 100.0)),
        )

    def test_expected_values_and_evpi(self) -> None:
        """EU(take) = 77, EU(leave) = 70, EVPI = 91 - 77 = 14."""
        problem = self._umbrella()
        self.assertEqual(dict(problem.expected_values()), {"take": 77.0, "leave": 70.0})
        self.assertEqual(problem.optimal_decision(), "take")
        self.assertAlmostEqual(problem.value_with_perfect_information(), 91.0)
        self.assertAlmostEqual(problem.expected_value_of_perfect_information(), 14.0)

    def test_loss_orientation(self) -> None:
        """Minimizing losses mirrors maximizing utilities."""
        problem = DecisionProblem(
            ("take", "leave"),
            ("rain", "dry"),
            (0.3, 0.7),
            ((30.0, 20.0), (100.0, 0.0)),
            maximize=False,
        )
        self.assertEqual(problem.optimal_decision(), "take")
        self.assertAlmostEqual(problem.expected_value_of_perfect_information(), 14.0)

    def test_evsi_by_preposterior_analysis(self) -> None:
        """A forecast with 80%/10% hit/false-alarm rates is worth 8.4."""
        analysis = self._umbrella().preposterior(
            ("say_rain", "say_dry"), ((0.8, 0.2), (0.1, 0.9))
        )
        self.assertAlmostEqual(analysis.outcome_probabilities["say_rain"], 0.31)
        self.assertAlmostEqual(analysis.posteriors["say_rain"][0], 0.24 / 0.31)
        self.assertEqual(
            dict(analysis.decision_rule), {"say_rain": "take", "say_dry": "leave"}
        )
        self.assertAlmostEqual(analysis.value_with_sample_information, 85.4)
        self.assertAlmostEqual(analysis.evsi, 8.4)
        self.assertLessEqual(analysis.evsi, analysis.evpi)
        useless = self._umbrella().preposterior(("o1", "o2"), ((0.5, 0.5), (0.5, 0.5)))
        self.assertAlmostEqual(useless.evsi, 0.0)

    def test_invalid_decision_problems(self) -> None:
        """Shape and probability errors are rejected."""
        with self.assertRaises(ValueError):
            DecisionProblem(("a",), ("s1", "s2"), (0.5, 0.6), ((1.0, 2.0),))
        with self.assertRaises(ValueError):
            DecisionProblem(("a", "a"), ("s",), (1.0,), ((1.0,), (2.0,)))
        with self.assertRaises(ValueError):
            DecisionProblem(("a",), ("s1", "s2"), (0.5, 0.5), ((1.0,),))
        with self.assertRaises(ValueError):
            self._umbrella().preposterior(("o",), ((1.0,),))

    def test_risk_matrix_classification(self) -> None:
        """Values on an edge fall into the higher bin."""
        matrix = RiskMatrix(
            (1e-3, 1e-1),
            (10.0, 1000.0),
            (
                ("low", "low", "medium"),
                ("low", "medium", "high"),
                ("medium", "high", "high"),
            ),
        )
        self.assertEqual(matrix.classify(1e-4, 5.0), "low")
        self.assertEqual(matrix.cell(1e-3, 10.0), (1, 1))
        self.assertEqual(matrix.classify(0.5, 1e6), "high")
        with self.assertRaises(ValueError):
            RiskMatrix((1.0, 0.5), (1.0,), (("a", "b"),) * 3)
        with self.assertRaises(ValueError):
            RiskMatrix((1.0,), (1.0,), (("a", "b"),))


if __name__ == "__main__":
    unittest.main()
