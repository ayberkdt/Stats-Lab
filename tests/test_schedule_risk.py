"""Schedule risk analysis: CPM, PERT, simulation, and cost-schedule confidence.

Oracles are hand-computed CPM dates, the exact law of a sum of independent
normals, the closed-form maximum of two independent exponentials or uniforms
(mean, distribution function, criticality one half), the mean of the maximum
of two correlated normals ``mu + sigma sqrt((1 - rho) / pi)``, and the product
rule for independent events. Seeded Monte Carlo results are judged on the
scale of their own binomial or sampling standard error, never by pinned
digits.

References:
    [REF-KELLEY-WALKER-1959], [REF-MALCOLM-1959], [REF-VAN-SLYKE-1963],
    [REF-MACCRIMMON-RYAVEC-1964], [REF-WILLIAMS-1992], [REF-GAO-16-89G],
    [REF-CLOPPER-PEARSON-1934], [REF-NASA-CEH-2015]. Checks
    [EQ-COST-CPM-001], [EQ-COST-PERT-001], [EQ-COST-SRA-001],
    [EQ-COST-SRA-002], [EQ-COST-SRA-003], [EQ-COST-LOAD-001].

"""

from __future__ import annotations

import math
import unittest
from statistics import NormalDist

from probstat.cost import (
    Activity,
    ActivityNetwork,
    Precedence,
    ScheduleRiskEvent,
    cost_schedule_frontier,
    cost_schedule_samples,
    critical_path,
    joint_cost_schedule_confidence,
    pert_estimate,
    pert_three_point,
    simulate_schedule,
)
from probstat.distributions import Exponential, Normal, Uniform
from probstat.inference import clopper_pearson_interval
from probstat.propagation import order_statistic_interval
from probstat.risk import DiscreteImpact, Risk

N = 8000
Z = 4.5


def frequency_ok(test: unittest.TestCase, count: int, n: int, truth: float) -> None:
    """Assert a frequency is within ``Z`` binomial standard errors of ``truth``.

    References:
        [REF-CLOPPER-PEARSON-1934]. No mathematical equation is evaluated.

    """
    se = math.sqrt(truth * (1.0 - truth) / n)
    test.assertLess(abs(count / n - truth), Z * se, f"{count / n} vs {truth}")


def classic_network() -> ActivityNetwork:
    """Return the six-activity network whose dates are computed by hand.

    References:
        [REF-KELLEY-WALKER-1959].

    """
    return ActivityNetwork(
        (
            Activity("A", 3),
            Activity("B", 2, ("A",)),
            Activity("C", 4, ("A",)),
            Activity("D", 3, ("B",)),
            Activity("E", 2, ("C",)),
            Activity("F", 2, ("D", "E")),
        )
    )


class CriticalPathTests(unittest.TestCase):
    """Deterministic CPM against dates worked out by hand.

    References:
        [REF-KELLEY-WALKER-1959]. Checks [EQ-COST-CPM-001].

    """

    def test_classic_network_dates_and_floats(self) -> None:
        """Every early/late date, total float, free float, and the path.

        References:
            [REF-KELLEY-WALKER-1959]. Checks [EQ-COST-CPM-001].

        """
        result = critical_path(classic_network())
        # Hand computation: ES/EF A 0/3, B 3/5, C 3/7, D 5/8, E 7/9, F 9/11.
        # Backward: F LF 11 LS 9; D LF 9 LS 6; E LF 9 LS 7; B LF 6 LS 4;
        # C LF 7 LS 3; A LS min(4, 3) = 3 - 3 = 0.
        expected = {
            "A": (0, 3, 0, 3, 0, 0),
            "B": (3, 5, 4, 6, 1, 0),
            "C": (3, 7, 3, 7, 0, 0),
            "D": (5, 8, 6, 9, 1, 1),
            "E": (7, 9, 7, 9, 0, 0),
            "F": (9, 11, 9, 11, 0, 0),
        }
        self.assertEqual(result.duration, 11)
        for name, (es, ef, ls, lf, tf, ff) in expected.items():
            row = result.timing(name)
            self.assertEqual(
                (
                    row.early_start,
                    row.early_finish,
                    row.late_start,
                    row.late_finish,
                    row.total_float,
                    row.free_float,
                ),
                (es, ef, ls, lf, tf, ff),
                name,
            )
            self.assertEqual(row.critical, tf == 0)
        self.assertEqual(result.critical_paths, (("A", "C", "E", "F"),))
        self.assertFalse(result.paths_truncated)

    def test_start_to_start_finish_to_finish_and_lag(self) -> None:
        """SS with lag 1 and FF with lag 1 give a critical chain A, B, C.

        References:
            [REF-KELLEY-WALKER-1959]. Checks [EQ-COST-CPM-001].

        """
        network = ActivityNetwork(
            (
                Activity("A", 4),
                Activity("B", 3, (Precedence("A", "SS", 1.0),)),
                Activity("C", 2, (Precedence("B", "FF", 1.0),)),
                Activity("D", 0.5, ("A",)),
            )
        )
        result = critical_path(network)
        # Hand: B ES 1 EF 4; C needs EF >= 5 so ES 3 EF 5; D ES 4 EF 4.5.
        self.assertEqual(result.duration, 5.0)
        self.assertEqual(
            (result.timing("B").early_start, result.timing("C").early_start),
            (1.0, 3.0),
        )
        for name in ("A", "B", "C"):
            self.assertTrue(result.timing(name).critical, name)
        self.assertEqual(result.timing("D").total_float, 0.5)
        self.assertEqual(result.timing("D").free_float, 0.5)
        self.assertEqual(result.critical_paths, (("A", "B", "C"),))

    def test_finish_to_start_lag(self) -> None:
        """A lag of 2 on a finish-to-start relation delays the successor.

        References:
            [REF-KELLEY-WALKER-1959]. Checks [EQ-COST-CPM-001].

        """
        network = ActivityNetwork(
            (Activity("A", 1), Activity("B", 1, (Precedence("A", lag=2.0),)))
        )
        self.assertEqual(critical_path(network).duration, 4.0)

    def test_two_equal_paths_are_both_reported(self) -> None:
        """A diamond with equal branches has two critical paths.

        References:
            [REF-KELLEY-WALKER-1959]. Checks [EQ-COST-CPM-001].

        """
        network = ActivityNetwork(
            (
                Activity("S", 1),
                Activity("L", 2, ("S",)),
                Activity("R", 2, ("S",)),
                Activity("T", 1, ("L", "R")),
            )
        )
        result = critical_path(network)
        self.assertEqual(set(result.critical_paths), {("S", "L", "T"), ("S", "R", "T")})
        self.assertTrue(critical_path(network, max_paths=1).paths_truncated)

    def test_duration_override_and_table(self) -> None:
        """Overrides change the dates; the table lists activities and the path.

        References:
            [REF-KELLEY-WALKER-1959]. Checks [EQ-COST-CPM-001].

        """
        result = critical_path(classic_network(), {"B": 10})
        self.assertEqual(result.duration, 18)
        self.assertEqual(result.critical_paths, (("A", "B", "D", "F"),))
        text = result.table()
        self.assertIn("critical path: A -> B -> D -> F", text)
        self.assertIn("project duration: 18", text)

    def test_validation(self) -> None:
        """Bad networks and arguments raise.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaisesRegex(ValueError, "cycle: a -> b -> c -> a"):
            ActivityNetwork(
                (
                    Activity("a", 1, ("c",)),
                    Activity("b", 1, ("a",)),
                    Activity("c", 1, ("b",)),
                )
            )
        with self.assertRaisesRegex(ValueError, "cycle"):
            ActivityNetwork(
                (
                    Activity("s", 1),
                    Activity("a", 1, ("s", "b")),
                    Activity("b", 1, ("a",)),
                )
            )
        with self.assertRaisesRegex(ValueError, "unknown predecessor"):
            ActivityNetwork((Activity("a", 1, ("zzz",)),))
        with self.assertRaisesRegex(ValueError, "precedes itself"):
            ActivityNetwork((Activity("a", 1, ("a",)),))
        with self.assertRaisesRegex(ValueError, "distinct"):
            ActivityNetwork((Activity("a", 1), Activity("a", 2)))
        with self.assertRaises(ValueError):
            ActivityNetwork(())
        with self.assertRaises(TypeError):
            ActivityNetwork(("a",))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Activity("", 1)
        with self.assertRaises(ValueError):
            Activity("a", -1.0)
        with self.assertRaises(TypeError):
            Activity("a", object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Activity("a", 1, (3,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Precedence("a", "SF")
        with self.assertRaises(ValueError):
            Precedence("a", lag=math.inf)
        with self.assertRaises(ValueError):
            critical_path(classic_network(), {"nope": 1.0})
        with self.assertRaises(ValueError):
            critical_path(classic_network(), {"A": -1.0})
        with self.assertRaises(ValueError):
            critical_path(classic_network(), max_paths=0)
        with self.assertRaises(TypeError):
            critical_path("network")  # type: ignore[arg-type]

    def test_topological_order_respects_precedence(self) -> None:
        """Every activity appears after all its predecessors.

        References:
            [REF-KELLEY-WALKER-1959].

        """
        network = classic_network()
        position = {n: i for i, n in enumerate(network.topological_order)}
        for activity in network.activities:
            for relation in activity.predecessors:
                self.assertLess(position[relation.activity], position[activity.name])


class PertTests(unittest.TestCase):
    """The PERT normal approximation and its merge bias.

    References:
        [REF-MALCOLM-1959], [REF-MACCRIMMON-RYAVEC-1964]. Checks
        [EQ-COST-PERT-001].

    """

    def test_three_point_formulas(self) -> None:
        """``(a + 4m + b)/6`` and ``((b - a)/6)^2`` for (2, 5, 14).

        References:
            [REF-MALCOLM-1959]. Checks [EQ-COST-PERT-001].

        """
        mean, variance = pert_three_point(2.0, 5.0, 14.0)
        self.assertAlmostEqual(mean, 6.0, 12)
        self.assertAlmostEqual(variance, 4.0, 12)
        with self.assertRaises(ValueError):
            pert_three_point(5.0, 2.0, 14.0)

    def test_series_network_is_exact(self) -> None:
        """For a series of independent normals PERT equals the exact law.

        References:
            [REF-MALCOLM-1959]. Checks [EQ-COST-PERT-001].

        """
        network = ActivityNetwork(
            (
                Activity("a", Normal(10, 1)),
                Activity("b", Normal(20, 2), ("a",)),
                Activity("c", Normal(15, 2), ("b",)),
            )
        )
        pert = pert_estimate(network)
        self.assertEqual(pert.mean, 45.0)
        self.assertAlmostEqual(pert.variance, 9.0, 12)
        exact = NormalDist(45.0, 3.0)
        self.assertAlmostEqual(pert.probability_by(48.0), exact.cdf(48.0), 12)
        self.assertAlmostEqual(pert.quantile(0.9), exact.inv_cdf(0.9), 9)
        result = simulate_schedule(network, samples=N, seed=3)
        self.assertLess(abs(result.mean - pert.mean), Z * 3.0 / math.sqrt(N))
        self.assertLess(
            abs(result.standard_deviation - 3.0), Z * 3.0 / math.sqrt(2 * N)
        )
        self.assertEqual(pert.path, ("a", "b", "c"))

    def test_parallel_network_underestimates_the_mean(self) -> None:
        """Two parallel Exp(1): PERT mean 1, true mean 1.5 (merge bias).

        References:
            [REF-MACCRIMMON-RYAVEC-1964]. Checks [EQ-COST-PERT-001].

        """
        network = ActivityNetwork(
            (Activity("a", Exponential(1.0)), Activity("b", Exponential(1.0)))
        )
        result = simulate_schedule(network, samples=N, seed=5)
        assert result.pert is not None
        self.assertAlmostEqual(result.pert.mean, 1.0, 12)
        # E max = 1 + 1/2; Var max = 1 + 1/4 for two i.i.d. Exp(1).
        se = math.sqrt(1.25 / N)
        self.assertLess(abs(result.mean - 1.5), Z * se)
        self.assertGreater(result.mean - result.pert.mean, 0.4)
        self.assertIn("biased low", result.pert.notes[0])

    def test_pert_rejects_non_finish_to_start(self) -> None:
        """SS and FF relations are outside the classical approximation.

        References:
            [REF-MALCOLM-1959].

        """
        network = ActivityNetwork(
            (Activity("a", 1), Activity("b", 1, (Precedence("a", "SS"),)))
        )
        with self.assertRaises(ValueError):
            pert_estimate(network)
        with self.assertRaises(TypeError):
            pert_estimate(None)  # type: ignore[arg-type]
        self.assertIsNone(simulate_schedule(network, samples=100, seed=1).pert)

    def test_deterministic_pert_probability_is_a_step(self) -> None:
        """Zero variance makes the completion probability a step function.

        References:
            [REF-MALCOLM-1959].

        """
        pert = pert_estimate(classic_network())
        self.assertEqual(pert.variance, 0.0)
        self.assertEqual(pert.probability_by(10.9), 0.0)
        self.assertEqual(pert.probability_by(11.0), 1.0)
        self.assertEqual(pert.quantile(0.5), 11.0)
        with self.assertRaises(ValueError):
            pert.quantile(1.0)


class SimulationTests(unittest.TestCase):
    """Monte Carlo schedule risk against closed forms.

    References:
        [REF-VAN-SLYKE-1963], [REF-WILLIAMS-1992]. Checks [EQ-COST-SRA-001]
        and [EQ-COST-SRA-002].

    """

    def parallel(self, law: object) -> ActivityNetwork:
        """Return two unconnected activities with the same duration law.

        References:
            [REF-VAN-SLYKE-1963].

        """
        return ActivityNetwork(
            (Activity("a", law), Activity("b", law))  # type: ignore[arg-type]
        )

    def test_parallel_uniform_law_criticality_and_deadline(self) -> None:
        """Max of two U(0,1): P(T <= t) = t^2, mean 2/3, criticality 1/2.

        References:
            [REF-VAN-SLYKE-1963]. Checks [EQ-COST-SRA-001], [EQ-COST-SRA-002].

        """
        result = simulate_schedule(
            self.parallel(Uniform(0.0, 1.0)), samples=N, seed=11, confidence=0.999
        )
        for name in ("a", "b"):
            row = result.activity(name)
            frequency_ok(self, round(row.criticality * N), N, 0.5)
            self.assertLess(row.criticality_interval.lower, 0.5)
            self.assertGreater(row.criticality_interval.upper, 0.5)
        # Var max = 2/4 - (2/3)^2 = 1/18.
        self.assertLess(abs(result.mean - 2.0 / 3.0), Z * math.sqrt(1 / 18 / N))
        for deadline in (0.3, 0.7, 0.9):
            estimate = result.probability_by(deadline)
            frequency_ok(self, estimate.successes, N, deadline**2)
            self.assertLessEqual(estimate.interval.lower, deadline**2)
            self.assertGreaterEqual(estimate.interval.upper, deadline**2)
        # Median of the maximum: sqrt(1/2).
        median = result.quantile(0.5, 0.999)
        self.assertLessEqual(median.lower, math.sqrt(0.5))
        self.assertGreaterEqual(median.upper, math.sqrt(0.5))

    def test_parallel_exponential_law(self) -> None:
        """Max of two Exp(1): ``F(t) = (1 - e^-t)^2``.

        References:
            [REF-VAN-SLYKE-1963]. Checks [EQ-COST-SRA-002].

        """
        result = simulate_schedule(self.parallel(Exponential(1.0)), samples=N, seed=2)
        for deadline in (0.5, 1.5, 3.0):
            truth = (1.0 - math.exp(-deadline)) ** 2
            frequency_ok(self, result.probability_by(deadline).successes, N, truth)
        for name in ("a", "b"):
            frequency_ok(self, round(result.activity(name).criticality * N), N, 0.5)

    def test_quantile_interval_matches_the_library_ranks(self) -> None:
        """The bisection ranks equal those of ``order_statistic_interval``.

        References:
            [REF-DAVID-NAGARAJA-2003, Section 7.1]. Checks [EQ-COST-SRA-002].

        """
        result = simulate_schedule(
            self.parallel(Exponential(1.0)), samples=400, seed=12
        )
        for p, level in ((0.5, 0.95), (0.9, 0.9), (0.1, 0.99), (0.75, 0.8)):
            ours = result.quantile(p, level)
            theirs = order_statistic_interval(result.ordered, p, confidence=level)
            self.assertEqual(
                (ours.lower_rank, ours.upper_rank),
                (theirs.lower_rank, theirs.upper_rank),
            )
            self.assertEqual((ours.lower, ours.upper), (theirs.lower, theirs.upper))
            self.assertAlmostEqual(ours.coverage, theirs.coverage, 9)
            self.assertAlmostEqual(ours.estimate, theirs.estimate, 12)
        with self.assertRaises(ValueError):
            result.quantile(0.999999, 0.999)
        with self.assertRaises(ValueError):
            result.quantile(1.0)

    def test_deadline_interval_is_clopper_pearson(self) -> None:
        """The reported interval is the library's exact interval of the count.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-COST-SRA-002].

        """
        result = simulate_schedule(
            self.parallel(Uniform(0.0, 1.0)), samples=1000, seed=4
        )
        estimate = result.probability_by(0.8, confidence=0.9)
        direct = clopper_pearson_interval(estimate.successes, 1000, confidence=0.9)
        self.assertEqual(estimate.interval, direct)
        self.assertEqual(estimate.trials, 1000)
        self.assertAlmostEqual(estimate.probability, estimate.successes / 1000, 15)
        self.assertEqual(result.probability_by(10.0).probability, 1.0)
        self.assertEqual(result.probability_by(-1.0).probability, 0.0)
        with self.assertRaises(ValueError):
            result.probability_by(0.5, confidence=1.0)

    def test_series_cruciality_and_sensitivity(self) -> None:
        """Series of N(0,1) and N(0,2) shifted: corr(d1, T) = s1/sqrt(s1^2+s2^2).

        References:
            [REF-WILLIAMS-1992], [REF-ELMAGHRABY-2000]. Checks [EQ-COST-SRA-001].

        """
        network = ActivityNetwork(
            (
                Activity("a", Normal(30, 1)),
                Activity("b", Normal(40, 2), ("a",)),
                Activity("fixed", 5.0, ("b",)),
            )
        )
        result = simulate_schedule(network, samples=N, seed=9)
        total_sd = math.sqrt(5.0)
        for name, sd in (("a", 1.0), ("b", 2.0)):
            row = result.activity(name)
            self.assertEqual(row.criticality, 1.0)
            self.assertLess(abs(row.cruciality - sd / total_sd), 0.02)
            self.assertLess(abs(row.rank_cruciality - sd / total_sd), 0.02)
            self.assertLess(abs(row.sensitivity - sd / total_sd), 0.02)
        fixed = result.activity("fixed")
        self.assertEqual(
            (fixed.criticality, fixed.cruciality, fixed.sensitivity), (1.0, 0.0, 0.0)
        )
        self.assertLess(abs(result.mean - 75.0), Z * total_sd / math.sqrt(N))

    def test_copula_correlation_widens_the_series_and_narrows_the_max(self) -> None:
        """``Var(sum) = 2 s^2 (1 + rho)`` and ``E max = mu + s sqrt((1-rho)/pi)``.

        References:
            [REF-WILLIAMS-1992], [REF-LIU-DERKIUREGHIAN-1986]. Checks
            [EQ-COST-SRA-001].

        """
        rho = 0.6
        law = Normal(50.0, 2.0)
        series = ActivityNetwork((Activity("a", law), Activity("b", law, ("a",))))
        correlated = simulate_schedule(
            series, samples=N, correlation={("a", "b"): rho}, seed=7
        )
        variance = 2 * 4.0 * (1 + rho)
        self.assertLess(
            abs(correlated.standard_deviation - math.sqrt(variance)),
            Z * math.sqrt(variance) / math.sqrt(2 * N),
        )
        independent = simulate_schedule(series, samples=N, seed=7)
        self.assertLess(independent.standard_deviation, correlated.standard_deviation)
        parallel = ActivityNetwork((Activity("a", law), Activity("b", law)))
        result = simulate_schedule(parallel, samples=N, correlation=rho, seed=8)
        truth = 50.0 + 2.0 * math.sqrt((1.0 - rho) / math.pi)
        self.assertLess(abs(result.mean - truth), Z * 2.0 / math.sqrt(N))
        frequency_ok(self, round(result.activity("a").criticality * N), N, 0.5)
        # Pearson-matched dependence of normals coincides with the copula.
        matched = simulate_schedule(
            parallel,
            samples=N,
            correlation=rho,
            correlation_kind="pearson",
            seed=8,
        )
        self.assertLess(abs(matched.mean - truth), Z * 2.0 / math.sqrt(N))

    def test_reproducible_and_seed_sensitive(self) -> None:
        """The same integer seed reproduces every sample; another does not.

        References:
            [REF-VAN-SLYKE-1963].

        """
        network = self.parallel(Uniform(0.0, 1.0))
        first = simulate_schedule(network, samples=200, seed=21)
        again = simulate_schedule(network, samples=200, seed=21)
        other = simulate_schedule(network, samples=200, seed=22)
        self.assertEqual(first.completion, again.completion)
        self.assertNotEqual(first.completion, other.completion)
        self.assertEqual(first.ordered, tuple(sorted(first.completion)))

    def test_risk_event_adds_duration_with_its_probability(self) -> None:
        """A 30 % event of 5 on a 10-day activity: P(T = 15) = 0.3.

        References:
            [REF-GAO-16-89G]. Checks [EQ-COST-SRA-003].

        """
        network = ActivityNetwork((Activity("work", 10.0),))
        event = ScheduleRiskEvent(Risk("late parts", 0.3, 5.0), "work")
        result = simulate_schedule(network, samples=N, risk_events=[event], seed=1)
        hits = sum(1 for c in result.completion if c == 15.0)
        frequency_ok(self, hits, N, 0.3)
        self.assertEqual(set(result.completion), {10.0, 15.0})
        self.assertIn("risk event", result.notes[-1])
        law_event = ScheduleRiskEvent(Risk("rework", 0.5, Uniform(2.0, 4.0)), "work")
        discrete = ScheduleRiskEvent(
            Risk("choice", 1.0, DiscreteImpact((1.0, 3.0), (0.5, 0.5))), "work"
        )
        more = simulate_schedule(
            network, samples=N, risk_events=[law_event, discrete], seed=2
        )
        self.assertLess(abs(more.mean - (10.0 + 1.5 + 2.0)), 0.05)
        mitigated = ScheduleRiskEvent(
            Risk("m", 0.9, 5.0, mitigated_probability=0.0), "work", mitigated=True
        )
        quiet = simulate_schedule(network, samples=200, risk_events=[mitigated], seed=3)
        self.assertEqual(set(quiet.completion), {10.0})
        with self.assertRaises(TypeError):
            ScheduleRiskEvent("risk", "work")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_schedule(
                network,
                samples=100,
                risk_events=[ScheduleRiskEvent(Risk("r", 0.1, 1.0), "missing")],
            )

    def test_negative_draws_are_rejected(self) -> None:
        """A law that can draw below zero is reported with the activity name.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        network = ActivityNetwork((Activity("shaky", Normal(0.1, 1.0)),))
        with self.assertRaisesRegex(ValueError, "shaky"):
            simulate_schedule(network, samples=200, seed=1)

    def test_argument_validation(self) -> None:
        """Invalid simulation arguments raise.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        network = self.parallel(Uniform(0.0, 1.0))
        fixed = ActivityNetwork((Activity("a", 1.0), Activity("b", Uniform(0, 1))))
        for kwargs in (
            {"samples": 50},
            {"samples": 100.0},
            {"confidence": 1.0},
            {"correlation_kind": "kendall"},
            {"correlation": {("a", "zz"): 0.5}},
            {"correlation": {("a", "a"): 0.5}},
            {"correlation": {("a", "b"): 1.5}},
            {"correlation": 1.0},
            {
                "correlation": {("a", "b"): 0.5},
                "correlation_kind": "pearson",
                "samples": 100.5,
            },
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                simulate_schedule(network, **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_schedule(fixed, samples=100, correlation={("a", "b"): 0.5})
        with self.assertRaises(TypeError):
            simulate_schedule("network")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            simulate_schedule(network, samples=100, risk_events=["x"])  # type: ignore[list-item]
        result = simulate_schedule(network, samples=100, seed=1)
        with self.assertRaises(ValueError):
            result.activity("zz")
        with self.assertRaises(ValueError):
            result.s_curve(1)

    def test_s_curve_table_and_deterministic_network(self) -> None:
        """The S-curve is monotone; the table names every activity.

        References:
            [REF-GAO-16-89G].

        """
        result = simulate_schedule(classic_network(), samples=100, seed=1)
        self.assertEqual(set(result.completion), {11.0})
        self.assertEqual(result.standard_deviation, 0.0)
        for name in ("A", "C", "E", "F"):
            self.assertEqual(result.activity(name).criticality, 1.0)
        for name in ("B", "D"):
            self.assertEqual(result.activity(name).criticality, 0.0)
        curve = result.s_curve(5)
        self.assertEqual(curve[-1][1], 1.0)
        self.assertEqual(curve[0][1], 1.0)
        random_result = simulate_schedule(
            self.parallel(Exponential(1.0)), samples=500, seed=1
        )
        values = [p for _, p in random_result.s_curve(11)]
        self.assertEqual(values, sorted(values))
        text = random_result.table()
        for token in ("completion:", "PERT:", "crit", "a ", "b ", "P50"):
            self.assertIn(token, text)


class CostScheduleTests(unittest.TestCase):
    """Cost loading, joint confidence, and the iso-confidence frontier.

    References:
        [REF-NASA-CEH-2015]. Checks [EQ-COST-LOAD-001] and [EQ-COST-JCL-001].

    """

    def network(self) -> ActivityNetwork:
        """Return two parallel uniform activities.

        References:
            [REF-VAN-SLYKE-1963].

        """
        return ActivityNetwork(
            (Activity("a", Uniform(8.0, 12.0)), Activity("b", Uniform(8.0, 12.0)))
        )

    def test_deterministic_loading_arithmetic(self) -> None:
        """Cost = constant + fixed + burn * duration + project rate * T.

        References:
            [REF-NASA-CEH-2015]. Checks [EQ-COST-LOAD-001].

        """
        network = ActivityNetwork((Activity("a", 10.0), Activity("b", 4.0, ("a",))))
        result = simulate_schedule(network, samples=100, seed=1)
        pairs = cost_schedule_samples(
            result,
            fixed_costs={"a": 100.0},
            burn_rates={"a": 3.0, "b": 5.0},
            project_rate=2.0,
            constant=7.0,
            seed=1,
        )
        # 7 + 100 + 3*10 + 5*4 + 2*14 = 185.
        self.assertEqual(set(pairs.costs), {185.0})
        self.assertEqual(pairs.size, 100)
        self.assertEqual(pairs.correlation, 0.0)

    def test_independent_cost_gives_the_product_of_marginals(self) -> None:
        """With cost independent of schedule, JCL = P(cost) P(schedule).

        References:
            [REF-NASA-CEH-2015]. Checks [EQ-COST-JCL-001].

        """
        result = simulate_schedule(self.network(), samples=N, seed=6)
        pairs = cost_schedule_samples(
            result, fixed_costs={"a": Normal(100.0, 10.0)}, seed=2
        )
        self.assertLess(abs(pairs.correlation), 0.03)
        deadline = result.quantile(0.7).estimate
        cost_target = 100.0 + 10.0 * NormalDist().inv_cdf(0.7)
        joint = joint_cost_schedule_confidence(pairs, cost_target, deadline)
        self.assertLess(
            abs(joint.joint - joint.cost_only * joint.schedule_only),
            Z * joint.standard_error,
        )
        self.assertLess(abs(joint.joint - 0.49), Z * 0.0036)
        self.assertLess(joint.joint, min(joint.cost_only, joint.schedule_only))

    def test_burn_rate_makes_cost_and_schedule_positively_dependent(self) -> None:
        """Burn-rate cost raises the joint confidence above the product.

        References:
            [REF-NASA-CEH-2015]. Checks [EQ-COST-LOAD-001].

        """
        result = simulate_schedule(self.network(), samples=N, seed=6)
        pairs = cost_schedule_samples(
            result,
            burn_rates={"a": 1.0, "b": 1.0},
            project_rate=Uniform(1.0, 3.0),
            seed=3,
        )
        self.assertGreater(pairs.correlation, 0.3)
        ordered = sorted(pairs.costs)
        deadline = result.quantile(0.7).estimate
        joint = joint_cost_schedule_confidence(pairs, ordered[int(0.7 * N)], deadline)
        self.assertGreater(
            joint.joint,
            joint.cost_only * joint.schedule_only + 5 * joint.standard_error,
        )
        self.assertLessEqual(joint.joint, min(joint.cost_only, joint.schedule_only))

    def test_frontier_points_reach_the_level(self) -> None:
        """Each frontier cost has joint confidence >= level, marginals higher.

        References:
            [REF-NASA-CEH-2015]. Checks [EQ-COST-JCL-001].

        """
        result = simulate_schedule(self.network(), samples=5000, seed=6)
        pairs = cost_schedule_samples(
            result,
            burn_rates={"a": 2.0, "b": 1.0},
            fixed_costs={"a": Normal(50, 5)},
            seed=3,
        )
        frontier = cost_schedule_frontier(pairs, 0.7)
        self.assertGreaterEqual(len(frontier.points), 3)
        for point in frontier.points:
            joint = joint_cost_schedule_confidence(pairs, point.cost, point.deadline)
            self.assertGreaterEqual(joint.joint, 0.7 - 1e-12)
            self.assertGreaterEqual(point.schedule_confidence, 0.7)
            self.assertGreaterEqual(point.cost_confidence, 0.7)
        deadlines = [p.deadline for p in frontier.points]
        costs = [p.cost for p in frontier.points]
        self.assertEqual(deadlines, sorted(deadlines))
        self.assertEqual(costs, sorted(costs, reverse=True))
        self.assertIn("joint confidence 0.7", frontier.table())
        explicit = cost_schedule_frontier(pairs, 0.7, [0.0, result.ordered[-1]])
        self.assertEqual([p.deadline for p in explicit.points], [result.ordered[-1]])

    def test_validation(self) -> None:
        """Bad arguments to the cost functions raise.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        result = simulate_schedule(self.network(), samples=100, seed=1)
        with self.assertRaises(TypeError):
            cost_schedule_samples("result")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cost_schedule_samples(result, burn_rates={"zz": 1.0})
        with self.assertRaises(ValueError):
            cost_schedule_samples(result, fixed_costs={"zz": 1.0})
        pairs = cost_schedule_samples(result, project_rate=1.0)
        with self.assertRaises(TypeError):
            joint_cost_schedule_confidence("x", 1.0, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cost_schedule_frontier("x", 0.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cost_schedule_frontier(pairs, 1.0)
        with self.assertRaises(ValueError):
            cost_schedule_frontier(pairs, 0.5, points=1)


if __name__ == "__main__":
    unittest.main()
