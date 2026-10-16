"""Markov reliability, availability, standby and phased-mission models.

Every builder is checked against an independent oracle: textbook closed
forms, binomial/product formulas, exact rational arithmetic
(``fractions.Fraction``), or literals computed once with mpmath 1.4 (``expm``
and ``lu_solve`` at 60 digits) and scipy 1.18 (``expm``) in a separate
verification environment.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from probstat.processes import ContinuousTimeMarkovChain, MarkovChain
from probstat.systems import (
    MarkovPhasedMission,
    MarkovReliabilityModel,
    MissionPhase,
    MissionStage,
    MissionStageModel,
    discrete_mission_outcome,
    from_states,
    load_sharing_pair,
    parallel_redundant_pair,
    repairable_k_out_of_n,
    single_repairable_unit,
    standby_system,
)


def binomial_at_least(n: int, k: int, p: float) -> float:
    """Return ``Pr(Binomial(n, p) >= k)``."""
    return math.fsum(
        math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)
    )


class SingleUnitTests(unittest.TestCase):
    """The two-state model against its closed forms."""

    def test_point_and_interval_measures(self) -> None:
        """A(t), R(t), interval availability, downtime and failure count."""
        lam, mu = 0.02, 0.5
        model = single_repairable_unit(lam, mu)
        total = lam + mu
        for t in (0.0, 0.3, 4.0, 50.0, 1000.0):
            availability = mu / total + lam / total * math.exp(-total * t)
            got = model.availability(t)
            # 1e-15 allows for rounding in evaluating the closed form itself.
            self.assertLessEqual(abs(got.value - availability), got.error_bound + 1e-15)
            self.assertLess(abs(got.value - availability), 1e-14)
            down = model.unavailability(t)
            self.assertLess(abs(down.value - (1 - availability)), 1e-14)
            self.assertLess(abs(model.reliability(t).value - math.exp(-lam * t)), 1e-14)
            omega = model.failure_frequency(t)
            self.assertLess(abs(omega.value - lam * availability), 1e-15)
        t = 30.0
        uptime = mu * t / total + lam / total**2 * (1 - math.exp(-total * t))
        interval = model.interval_availability(t)
        self.assertLess(abs(interval.value - uptime / t), 1e-14)
        self.assertLessEqual(abs(interval.value - uptime / t), interval.error_bound)
        downtime = model.expected_downtime(t)
        self.assertLess(abs(downtime.value - (t - uptime)), 1e-12)
        self.assertLessEqual(abs(downtime.value - (t - uptime)), downtime.error_bound)
        failures = model.expected_failures(t)
        self.assertLess(abs(failures.value - lam * uptime), 1e-14)
        curve = model.availability_curve((1.0, 2.0))
        self.assertLess(
            abs(curve[1].value - (mu / total + lam / total * math.exp(-2 * total))),
            1e-14,
        )

    def test_steady_state_measures(self) -> None:
        """MTTF = MUT = 1/lambda, MDT = MTTR = 1/mu, w = lambda mu/(lambda+mu)."""
        lam, mu = 0.02, 0.5
        model = single_repairable_unit(lam, mu)
        self.assertAlmostEqual(model.steady_state_availability(), mu / (lam + mu), 15)
        self.assertAlmostEqual(model.mean_time_to_failure(), 1 / lam, 12)
        self.assertAlmostEqual(model.mean_up_time(), 1 / lam, 11)
        self.assertAlmostEqual(model.mean_down_time(), 1 / mu, 13)
        self.assertAlmostEqual(model.mean_time_to_repair(), 1 / mu, 13)
        self.assertAlmostEqual(model.mean_time_between_failures(), 1 / lam + 1 / mu, 11)
        self.assertAlmostEqual(
            model.steady_state_failure_frequency(), lam * mu / (lam + mu), 15
        )

    def test_stiff_unit(self) -> None:
        """Rates 1e-6 and 1e3 at q t = 1e5: unavailability keeps 1e-12 relative."""
        lam, mu = 1e-6, 1e3
        model = single_repairable_unit(lam, mu)
        t = 100.0
        exact = lam / (lam + mu) * (1 - math.exp(-(lam + mu) * t))
        got = model.unavailability(t, tolerance=1e-20)
        self.assertLess(abs(got.value - exact) / exact, 1e-12)
        self.assertLessEqual(abs(got.value - exact), got.error_bound)
        self.assertLess(got.error_bound, 1e-17)

    def test_non_repairable_unit(self) -> None:
        """Without repair A = R, A_inf = 0 and MUT is undefined."""
        model = single_repairable_unit(0.1, 0.0)
        self.assertAlmostEqual(model.availability(3.0).value, math.exp(-0.3), 15)
        self.assertEqual(model.steady_state_availability(), 0.0)
        self.assertAlmostEqual(
            model.expected_failures(4.0).value, 1 - math.exp(-0.4), 14
        )
        with self.assertRaises(ValueError):
            model.mean_up_time()


class RedundancyTests(unittest.TestCase):
    """k-out-of-n, parallel pairs and load sharing."""

    def test_one_out_of_two_mttf(self) -> None:
        """``(3 lambda + mu) / (2 lambda^2)``, including the stiff case."""
        for lam, mu in ((0.1, 1.0), (1e-6, 1e3)):
            exact = (3 * lam + mu) / (2 * lam**2)
            for model in (
                repairable_k_out_of_n(2, 1, lam, mu),
                parallel_redundant_pair(lam, lam, mu, mu, shared_repair=True),
                parallel_redundant_pair(lam, lam, mu, mu, shared_repair=False),
            ):
                self.assertLess(
                    abs(model.mean_time_to_failure() - exact) / exact, 1e-14
                )

    def test_pair_availabilities(self) -> None:
        """Independent repair: product formula. Shared crew: birth-death."""
        independent = parallel_redundant_pair(0.1, 0.2, 1.0, 2.0, shared_repair=False)
        self.assertAlmostEqual(
            independent.steady_state_unavailability(), (0.1 / 1.1) * (0.2 / 2.2), 16
        )
        lam, mu = 0.1, 1.0
        shared = parallel_redundant_pair(lam, lam, mu, mu)
        exact = (mu * mu + 2 * lam * mu) / (mu * mu + 2 * lam * mu + 2 * lam * lam)
        self.assertAlmostEqual(shared.steady_state_availability(), exact, 15)
        self.assertEqual(len(shared.down_states), 2)

    def test_k_out_of_n_product_forms(self) -> None:
        """Binomial with independent repair; product formula with one crew."""
        lam, mu = 0.2, 1.5
        independent = repairable_k_out_of_n(4, 2, lam, mu, repair_crews=None)
        self.assertAlmostEqual(
            independent.steady_state_availability(),
            binomial_at_least(4, 2, mu / (lam + mu)),
            15,
        )
        crew = repairable_k_out_of_n(4, 2, lam, mu, repair_crews=1)
        weights = [1.0]
        for j in range(4):
            weights.append(weights[-1] * (4 - j) * lam / mu)
        self.assertAlmostEqual(
            crew.steady_state_availability(),
            math.fsum(weights[:3]) / math.fsum(weights),
            15,
        )
        # Independent repair: the system availability at t is binomial in the
        # unit availability at t.
        t = 0.7
        unit = mu / (lam + mu) + lam / (lam + mu) * math.exp(-(lam + mu) * t)
        self.assertLess(
            abs(independent.availability(t).value - binomial_at_least(4, 2, unit)),
            1e-14,
        )

    def test_non_repairable_k_out_of_n(self) -> None:
        """R(t) is binomial in exp(-lambda t); MTTF = sum 1/(j lambda)."""
        lam = 0.2
        model = repairable_k_out_of_n(4, 2, lam, 0.0)
        for t in (0.5, 1.0, 5.0, 20.0):
            exact = binomial_at_least(4, 2, math.exp(-lam * t))
            self.assertLess(abs(model.reliability(t).value - exact), 1e-14)
            self.assertLess(abs(model.availability(t).value - exact), 1e-14)
        self.assertAlmostEqual(
            model.mean_time_to_failure(), sum(1 / (j * lam) for j in range(2, 5)), 13
        )
        stopped = repairable_k_out_of_n(3, 2, lam, 1.0, failures_when_down=False)
        self.assertEqual(stopped.chain.exit_rate(2), 1.0)

    def test_load_sharing_pair(self) -> None:
        """Closed-form reliability and MTTF of a load-sharing pair."""
        lam, loaded = 0.1, 0.35
        model = load_sharing_pair(lam, loaded)
        for t in (1.0, 4.0, 12.0):
            exact = math.exp(-2 * lam * t) + 2 * lam / (loaded - 2 * lam) * (
                math.exp(-2 * lam * t) - math.exp(-loaded * t)
            )
            self.assertLess(abs(model.reliability(t).value - exact), 1e-14)
        self.assertAlmostEqual(
            model.mean_time_to_failure(), 1 / (2 * lam) + 1 / loaded, 13
        )
        repaired = load_sharing_pair(lam, loaded, repair_rate=1.0)
        self.assertGreater(
            repaired.mean_time_to_failure(), model.mean_time_to_failure()
        )

    def test_stiff_unreliability_against_high_precision(self) -> None:
        """1-out-of-2, lambda = 1e-6, mu = 1e3, up to t = 1e6 (q t = 1e9)."""
        model = repairable_k_out_of_n(2, 1, 1e-6, 1e3)
        # Oracle: mpmath.expm at 60 digits of the absorbing generator.
        oracle = {
            100.0: 1.99997999399992e-13,
            1e4: 1.9999997939800012e-11,
            1e6: 1.99999999e-09,
        }
        for t, exact in oracle.items():
            got = model.unreliability(t)
            self.assertLess(abs(got.value - exact) / exact, 1e-9)
            self.assertLessEqual(abs(got.value - exact), got.error_bound)


class StandbyTests(unittest.TestCase):
    """Cold, warm and hot standby with imperfect switching."""

    def test_cold_standby_erlang(self) -> None:
        """``R(t) = exp(-lambda t) sum_(k < n) (lambda t)^k / k!``."""
        lam = 0.3
        for spares in (1, 2, 3):
            model = standby_system(spares, lam)
            for t in (0.5, 3.0, 15.0):
                exact = math.exp(-lam * t) * math.fsum(
                    (lam * t) ** k / math.factorial(k) for k in range(spares + 1)
                )
                self.assertLess(abs(model.reliability(t).value - exact), 1e-14)
            self.assertAlmostEqual(model.mean_time_to_failure(), (spares + 1) / lam, 12)

    def test_imperfect_switching(self) -> None:
        """Cold: ``exp(-lt)(1 + p l t)``; warm: the one-spare closed form."""
        lam, dormant, p = 0.3, 0.1, 0.8
        cold = standby_system(1, lam, 0.0, p)
        warm = standby_system(1, lam, dormant, p)
        for t in (0.5, 2.0, 10.0):
            self.assertLess(
                abs(cold.reliability(t).value - math.exp(-lam * t) * (1 + p * lam * t)),
                1e-14,
            )
            exact = math.exp(-lam * t) * (
                1 + p * lam / dormant * (1 - math.exp(-dormant * t))
            )
            self.assertLess(abs(warm.reliability(t).value - exact), 1e-14)
        self.assertAlmostEqual(
            warm.mean_time_to_failure(), 1 / lam + p / (lam + dormant), 13
        )
        self.assertIn("switch failure (1 failed)", warm.down_states)

    def test_hot_standby_is_active_parallel(self) -> None:
        """``lambda_s = lambda`` and perfect switching: ``2e^-lt - e^-2lt``."""
        lam = 0.4
        hot = standby_system(1, lam, lam)
        for t in (0.5, 2.0):
            exact = 2 * math.exp(-lam * t) - math.exp(-2 * lam * t)
            self.assertLess(abs(hot.reliability(t).value - exact), 1e-14)

    def test_warm_standby_several_spares(self) -> None:
        """Rising-factorial closed form and the hypoexponential MTTF."""
        lam, dormant = 0.3, 0.1
        ratio = lam / dormant
        for spares in (1, 2, 3):
            model = standby_system(spares, lam, dormant)
            t = 2.5
            x = 1 - math.exp(-dormant * t)
            exact = math.exp(-lam * t) * math.fsum(
                math.prod(ratio + i for i in range(k)) / math.factorial(k) * x**k
                for k in range(spares + 1)
            )
            self.assertLess(abs(model.reliability(t).value - exact), 1e-14)
            self.assertAlmostEqual(
                model.mean_time_to_failure(),
                math.fsum(1 / (lam + j * dormant) for j in range(spares + 1)),
                13,
            )

    def test_repairable_standby_against_high_precision(self) -> None:
        """Two warm spares, p = 0.95, one crew: every measure versus mpmath."""
        model = standby_system(2, 0.5, 0.05, 0.95, repair_rate=2.0, repair_crews=1)
        # Oracle: mpmath.expm / Van Loan block exponential / lu_solve, 60 digits.
        cases = (
            (model.availability(1.0), 0.984914730443824),
            (model.availability(7.0), 0.9750051166605518),
            (model.reliability(1.0), 0.9681502697786586),
            (model.reliability(7.0), 0.7369911603905505),
            (model.expected_downtime(5.0), 0.0994409837777629),
            (model.expected_failures(5.0), 0.22376391257531933),
        )
        for got, exact in cases:
            self.assertLess(abs(got.value - exact), 2e-15)
            self.assertLessEqual(abs(got.value - exact), got.error_bound)
        self.assertAlmostEqual(model.mean_time_to_failure(), 22.076171875, 12)
        self.assertAlmostEqual(
            model.steady_state_availability(), 0.9749909184899545, 15
        )
        self.assertAlmostEqual(
            model.steady_state_failure_frequency(), 0.050018163020091114, 16
        )
        self.assertAlmostEqual(model.mean_up_time(), 19.49273743016759, 12)
        self.assertAlmostEqual(model.mean_down_time(), 0.5, 14)


class ModelValidationTests(unittest.TestCase):
    """Invalid models and infinite MTTF."""

    def test_invalid_models(self) -> None:
        """Initial mass on down states, no down states, bad parameters."""
        chain = ContinuousTimeMarkovChain(("up", "down"), {("up", "down"): 1.0})
        with self.assertRaises(ValueError):
            MarkovReliabilityModel(chain, ("up",), "down")
        with self.assertRaises(ValueError):
            MarkovReliabilityModel(chain, ("up", "down"), "up")
        with self.assertRaises(ValueError):
            single_repairable_unit(0.0, 1.0)
        with self.assertRaises(ValueError):
            repairable_k_out_of_n(3, 4, 1.0, 1.0)
        with self.assertRaises(ValueError):
            standby_system(1, 1.0, switch_success_probability=1.5)
        with self.assertRaises(TypeError):
            single_repairable_unit(True, 1.0)

    def test_infinite_mttf(self) -> None:
        """A reachable set of up states that never fails gives ``inf``."""
        model = from_states(
            ("new", "safe", "down"),
            {("new", "safe"): 1.0, ("new", "down"): 1.0},
            ("new", "safe"),
            "new",
        )
        self.assertEqual(model.mean_time_to_failure(), math.inf)
        self.assertAlmostEqual(model.reliability(50.0).value, 0.5, 14)
        unreachable = from_states(
            ("new", "safe", "down"),
            {("new", "down"): 2.0},
            ("new", "safe"),
            "new",
        )
        self.assertAlmostEqual(unreachable.mean_time_to_failure(), 0.5, 15)


class PhasedMissionTests(unittest.TestCase):
    """Phased missions against expm products and closed forms."""

    def setUp(self) -> None:
        """Build a three-phase two-unit mission."""
        states = ("0", "1", "2")
        self.takeoff = ContinuousTimeMarkovChain(
            states,
            {("0", "1"): 0.02, ("1", "2"): 0.01, ("1", "0"): 0.5, ("2", "1"): 0.5},
        )
        self.cruise = ContinuousTimeMarkovChain(
            states, {("0", "1"): 0.1, ("1", "2"): 0.05}
        )
        self.landing = ContinuousTimeMarkovChain(
            states, {("0", "1"): 0.04, ("1", "2"): 0.02, ("1", "0"): 1.0}
        )

    def test_against_expm_product(self) -> None:
        """Cumulative success equals the hand-chained matrix exponentials."""
        mission = MarkovPhasedMission(
            (
                MissionPhase("takeoff", self.takeoff, 2.0, ("0", "1")),
                MissionPhase("cruise", self.cruise, 10.0, ("0", "1")),
                MissionPhase("landing", self.landing, 1.0, ("0",), continuous=False),
            ),
            "0",
        )
        result = mission.solve()
        # Oracle: scipy expm of each phase generator (failure rows zeroed for
        # continuous phases), success states kept between phases.
        oracle = (0.9997106224590848, 0.8390524449411421, 0.6458746979727551)
        for got, exact in zip(result.cumulative_success, oracle, strict=True):
            self.assertLess(abs(got - exact), 1e-14)
            self.assertLessEqual(abs(got - exact), result.error_bound + 1e-15)
        self.assertAlmostEqual(result.phase_reliabilities[1], oracle[1] / oracle[0], 13)
        self.assertEqual(result.mission_success, result.cumulative_success[-1])
        self.assertAlmostEqual(math.fsum(result.final_distribution), oracle[2], 14)
        self.assertLess(result.error_bound, 1e-11)

    def test_identical_phases_equal_one_chain(self) -> None:
        """Splitting one CTMC into two phases changes nothing."""
        split = MarkovPhasedMission(
            (
                MissionPhase("a", self.cruise, 3.0, ("0", "1")),
                MissionPhase("b", self.cruise, 4.5, ("0", "1")),
            ),
            "0",
        )
        whole = MarkovReliabilityModel(self.cruise, ("0", "1"), "0")
        self.assertAlmostEqual(
            split.success_probability().value, whole.reliability(7.5).value, 14
        )

    def test_mapping_and_end_of_phase_checks(self) -> None:
        """A switching loss at the phase change; an end-only requirement."""
        lam1, lam2, p = 0.05, 0.2, 0.9
        unit1 = ContinuousTimeMarkovChain(("ok", "bad"), {("ok", "bad"): lam1})
        unit2 = ContinuousTimeMarkovChain(("on", "off"), {("on", "off"): lam2})
        mission = MarkovPhasedMission(
            (
                MissionPhase("one", unit1, 3.0, ("ok",)),
                MissionPhase(
                    "two", unit2, 2.0, ("on",), entry_mapping={("ok", "on"): p}
                ),
            ),
            "ok",
        )
        exact = math.exp(-3.0 * lam1) * p * math.exp(-2.0 * lam2)
        self.assertAlmostEqual(mission.solve().mission_success, exact, 15)
        lam, mu, duration = 0.3, 2.0, 1.5
        repairable = single_repairable_unit(lam, mu).chain
        end_only = MarkovPhasedMission(
            (MissionPhase("p", repairable, duration, ("up",), continuous=False),), "up"
        )
        availability = mu / (lam + mu) + lam / (lam + mu) * math.exp(
            -(lam + mu) * duration
        )
        self.assertAlmostEqual(end_only.solve().mission_success, availability, 15)
        continuous = MarkovPhasedMission(
            (MissionPhase("p", repairable, duration, ("up",)),), "up"
        )
        self.assertAlmostEqual(
            continuous.solve().mission_success, math.exp(-lam * duration), 15
        )

    def test_invalid_missions(self) -> None:
        """Mappings on the first phase, missing labels, bad rows."""
        with self.assertRaises(ValueError):
            MarkovPhasedMission(
                (MissionPhase("a", self.cruise, 1.0, ("0",), entry_mapping={}),), "0"
            )
        other = ContinuousTimeMarkovChain(("x", "y"), {("x", "y"): 1.0})
        with self.assertRaises(ValueError):
            MarkovPhasedMission(
                (
                    MissionPhase("a", self.cruise, 1.0, ("0",)),
                    MissionPhase("b", other, 1.0, ("x",)),
                ),
                "0",
            )
        with self.assertRaises(ValueError):
            MarkovPhasedMission(
                (
                    MissionPhase("a", self.cruise, 1.0, ("0",)),
                    MissionPhase(
                        "b", other, 1.0, ("x",), entry_mapping={("0", "x"): 1.5}
                    ),
                ),
                "0",
            )
        with self.assertRaises(ValueError):
            MissionPhase("a", self.cruise, -1.0, ("0",))


class DiscreteMissionTests(unittest.TestCase):
    """Stage models and discrete absorbing missions."""

    def test_stage_model_against_exact_fractions(self) -> None:
        """Success, abort, loss and expected attempts in exact arithmetic."""
        spec = (
            ("launch", Fraction(95, 100), Fraction(3, 100), Fraction(1, 100)),
            ("orbit", Fraction(9, 10), Fraction(5, 100), Fraction(4, 100)),
            ("return", Fraction(98, 100), Fraction(0), Fraction(1, 100)),
        )
        model = MissionStageModel(
            tuple(
                MissionStage(name, float(s), retry=float(r), abort=float(a))
                for name, s, r, a in spec
            )
        )
        success, abort, prefix = Fraction(1), Fraction(0), Fraction(1)
        for _, s, r, a in spec:
            abort += prefix * a / (1 - r)
            prefix *= s / (1 - r)
        success = prefix
        attempts = Fraction(0)
        for _, s, r, _ in reversed(spec):
            attempts = (1 + s * attempts) / (1 - r)
        outcome = model.outcome()
        self.assertAlmostEqual(outcome.success_probability, float(success), 15)
        self.assertAlmostEqual(outcome.other_probability, float(abort), 15)
        self.assertAlmostEqual(
            outcome.failure_probability, float(1 - success - abort), 15
        )
        self.assertAlmostEqual(outcome.expected_steps, float(attempts), 14)
        self.assertEqual(
            set(outcome.outcome_probabilities), {"success", "abort", "loss"}
        )

    def test_gamblers_ruin_mission(self) -> None:
        """A generic chain: reach the goal before ruin."""
        p, size = 0.4, 5
        rows = [[0.0] * (size + 1) for _ in range(size + 1)]
        rows[0][0] = rows[size][size] = 1.0
        for i in range(1, size):
            rows[i][i + 1] = p
            rows[i][i - 1] = 1 - p
        chain = MarkovChain(tuple(tuple(row) for row in rows))
        outcome = discrete_mission_outcome(chain, 2, (size,), (0,))
        ratio = (1 - p) / p
        exact = (1 - ratio**2) / (1 - ratio**size)
        self.assertAlmostEqual(outcome.success_probability, exact, 14)
        self.assertAlmostEqual(outcome.failure_probability, 1 - exact, 14)
        self.assertEqual(outcome.other_probability, 0.0)
        with self.assertRaises(ValueError):
            discrete_mission_outcome(chain, 2, (3,), (0,))
        with self.assertRaises(ValueError):
            discrete_mission_outcome(chain, 2, (0,), (0,))

    def test_invalid_stages(self) -> None:
        """Reserved names, certain retry, and probabilities above one."""
        with self.assertRaises(ValueError):
            MissionStage("loss", 0.5)
        with self.assertRaises(ValueError):
            MissionStage("a", 0.0, retry=1.0)
        with self.assertRaises(ValueError):
            MissionStage("a", 0.7, retry=0.4)
        with self.assertRaises(ValueError):
            MissionStage("a", 0.5, retry=0.2, failure=0.1)
        with self.assertRaises(ValueError):
            MissionStageModel((MissionStage("a", 0.5), MissionStage("a", 0.5)))


if __name__ == "__main__":
    unittest.main()
