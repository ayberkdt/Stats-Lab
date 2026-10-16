"""Continuous-time Markov chains: uniformization, rewards, GTH, absorption.

Reference values marked "oracle" were computed once in a separate
verification environment (scipy 1.18 ``expm`` / ``eig``; mpmath 1.4 ``expm``
at 60 significant digits for the stiff cases; Van Loan's augmented-matrix
exponential for time integrals) and pasted here as literals. No numerical
library is imported by the tests themselves.
"""

from __future__ import annotations

import math
import unittest

from probstat.processes import (
    BoundedValue,
    ContinuousTimeMarkovChain,
    MarkovChain,
)
from probstat.processes.ctmc import _poisson_window

FOUR_RATES = {
    ("a", "b"): 1.5,
    ("a", "c"): 0.3,
    ("b", "a"): 0.7,
    ("b", "d"): 2.0,
    ("c", "d"): 0.4,
    ("c", "a"): 1.1,
    ("d", "a"): 0.25,
    ("d", "b"): 0.6,
}
FOUR_START = {"a": 0.6, "c": 0.4}

# Oracle: scipy.linalg.expm, row vector (0.6, 0, 0.4, 0) times expm(Q t).
FOUR_TRANSIENT = {
    0.5: (
        0.3936614908585202,
        0.20455581220570135,
        0.23845812958202187,
        0.1633245673537568,
    ),
    3.0: (
        0.19787742135275438,
        0.22767973212270776,
        0.04766969449342673,
        0.5267731520311106,
    ),
    40.0: (
        0.18708354689902718,
        0.22603792926704377,
        0.03741670937980543,
        0.5494618144541291,
    ),
}
# Oracle: normalized left null vector of Q from scipy.linalg.eig.
FOUR_STATIONARY = (
    0.18708354689902632,
    0.22603792926704233,
    0.03741670937980526,
    0.5494618144541261,
)
# Oracle: Van Loan block exponential expm([[Q, I], [0, 0]] * 3), top-right block.
FOUR_OCCUPANCY_T3 = (
    0.8722000420958141,
    0.6399103571240596,
    0.4093268787568782,
    1.078562722023247,
)

# Stiff chain with rates from 1e-6 to 1e3 (q = 1000.000003, so q t = 1e5 at
# t = 100).
STIFF_RATES = {
    (0, 1): 1e-6,
    (0, 2): 2e-6,
    (1, 0): 1e3,
    (1, 3): 3e-6,
    (2, 0): 5e2,
    (2, 3): 1e-3,
    (3, 1): 2.0,
    (3, 2): 1e-1,
}
# Oracle: mpmath.expm at 60 digits, row of state 0.
STIFF_TRANSIENT = {
    1e-3: (
        0.999999997794002,
        6.321205576186251e-10,
        1.5738766374812202e-09,
        8.527275859205377e-16,
    ),
    1.0: (
        0.9999999949983326,
        1.0000033345673646e-09,
        3.999992314166535e-09,
        1.6717778156165689e-12,
    ),
    100.0: (
        0.9999999949980977,
        1.0000038043717745e-09,
        3.999992361245037e-09,
        1.9061868441229296e-12,
    ),
}
# Oracle: mpmath Van Loan block exponential at 60 digits, t = 100.
STIFF_OCCUPANCY_T100 = (
    99.99999949981968,
    9.99993786103479e-08,
    3.9999123597351736e-07,
    1.897071653643931e-10,
)


def l1(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return the L1 distance between two vectors."""
    return math.fsum(abs(a - b) for a, b in zip(left, right, strict=True))


class ConstructionTests(unittest.TestCase):
    """Validation of states, rates, and initial laws."""

    def test_mapping_and_matrix_give_the_same_generator(self) -> None:
        """A rate dictionary and the equivalent generator matrix agree."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        again = ContinuousTimeMarkovChain(("a", "b", "c", "d"), chain.generator)
        self.assertEqual(chain.generator, again.generator)
        for row in chain.generator:
            self.assertLess(abs(math.fsum(row)), 1e-15)
        self.assertEqual(chain.rate("b", "d"), 2.0)
        self.assertAlmostEqual(chain.exit_rate("a"), 1.8, places=15)

    def test_invalid_specifications_raise(self) -> None:
        """Negative rates, bad rows, unknown keys and labels are rejected."""
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain(("a", "b"), {("a", "b"): -1.0})
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain(("a", "b"), {("a", "z"): 1.0})
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain(("a", "b"), {("a", "a"): 1.0})
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain(("a", "b"), [[-1.0, 1.0], [1.0, -0.5]])
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain(("a", "b"), [[-1.0, 1.0], [-0.5, 0.5]])
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain(("a", "a"), {})
        with self.assertRaises(ValueError):
            ContinuousTimeMarkovChain((), {})
        with self.assertRaises(TypeError):
            ContinuousTimeMarkovChain("ab", {})
        with self.assertRaises(TypeError):
            ContinuousTimeMarkovChain(("a", "b"), {("a", "b"): True})

    def test_initial_laws(self) -> None:
        """Labels, mappings, and vectors are accepted; bad laws are not."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        self.assertEqual(chain.distribution("b"), (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(chain.distribution(FOUR_START), (0.6, 0.0, 0.4, 0.0))
        self.assertEqual(chain.distribution([0.6, 0.0, 0.4, 0.0]), (0.6, 0.0, 0.4, 0.0))
        for bad in ("z", {"a": 0.5}, [1.0, 0.0], [1.5, -0.5, 0.0, 0.0]):
            with self.assertRaises(ValueError):
                chain.distribution(bad)

    def test_invalid_analysis_arguments(self) -> None:
        """Times, tolerances, methods, and term limits are validated."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        with self.assertRaises(ValueError):
            chain.transient("a", -1.0)
        with self.assertRaises(ValueError):
            chain.transient("a", 1.0, tolerance=0.0)
        with self.assertRaises(ValueError):
            chain.transient("a", 1.0, method="pade")
        with self.assertRaises(ValueError):
            chain.transient("a", 1e6, method="series", max_terms=1000)


class TransientTests(unittest.TestCase):
    """Uniformization against matrix exponentials and closed forms."""

    def test_four_state_chain_against_expm(self) -> None:
        """L1 error is below 1e-13 and below the reported bound."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        for method in ("series", "squaring"):
            for t, expected in FOUR_TRANSIENT.items():
                solution = chain.transient(FOUR_START, t, method=method)
                error = l1(solution.probabilities, expected)
                self.assertLess(error, 1e-13)
                self.assertLessEqual(error, solution.error_bound)
                self.assertLess(solution.error_bound, 1e-10)

    def test_transient_many_matches_single_calls(self) -> None:
        """One shared pass gives the same answers as separate calls."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        times = (3.0, 0.0, 0.5, 40.0)
        many = chain.transient_many(FOUR_START, times, method="series")
        self.assertEqual(many[1].probabilities, (0.6, 0.0, 0.4, 0.0))
        self.assertEqual(many[1].method, "exact")
        for t, solution in zip(times, many, strict=True):
            single = chain.transient(FOUR_START, t, method="series")
            self.assertLess(l1(single.probabilities, solution.probabilities), 1e-15)

    def test_two_state_closed_form(self) -> None:
        """``p_up(t) = mu/(l+mu) + l/(l+mu) exp(-(l+mu) t)``."""
        failure, repair = 0.3, 2.0
        chain = ContinuousTimeMarkovChain(
            ("up", "down"), {("up", "down"): failure, ("down", "up"): repair}
        )
        times = (0.01, 0.1, 1.0, 10.0, 300.0)
        for method in ("series", "squaring"):
            for solution in chain.transient_many("up", times, method=method):
                exact = repair / (failure + repair) + failure / (
                    failure + repair
                ) * math.exp(-(failure + repair) * solution.time)
                bound = solution.total(("up",))
                self.assertLessEqual(abs(bound.value - exact), bound.error_bound)
                self.assertLess(abs(bound.value - exact), 1e-14)

    def test_stiff_chain_against_high_precision_expm(self) -> None:
        """Rates 1e-6 to 1e3, q t up to 1e5: tiny probabilities stay accurate."""
        chain = ContinuousTimeMarkovChain((0, 1, 2, 3), STIFF_RATES)
        for t, expected in STIFF_TRANSIENT.items():
            for method in ("series", "squaring"):
                solution = chain.transient(0, t, method=method)
                error = l1(solution.probabilities, expected)
                self.assertLessEqual(error, solution.error_bound)
                for computed, exact in zip(
                    solution.probabilities, expected, strict=True
                ):
                    self.assertLess(abs(computed - exact) / exact, 1e-11)
        long_run = chain.transient(0, 100.0, method="series")
        self.assertGreater(long_run.terms, 100_000)

    def test_large_q_t_uses_squaring_and_stays_bounded(self) -> None:
        """Q t = 1e9: the series would need 1e9 terms; squaring is exact enough."""
        lam, mu = 1e-6, 1e3
        chain = ContinuousTimeMarkovChain(
            (0, 1, 2),
            {(0, 1): 2 * lam, (1, 0): mu, (1, 2): lam},
        )
        solution = chain.transient(0, 1e6)
        self.assertTrue(solution.method.startswith("squaring"))
        down = solution.total((2,))
        # Oracle: mpmath.expm at 60 digits gives 1.99999999e-09.
        self.assertLessEqual(abs(down.value - 1.99999999e-09), down.error_bound)
        self.assertLess(abs(down.value - 1.99999999e-09) / 1.99999999e-09, 1e-9)

    def test_poisson_window_bounds_are_rigorous(self) -> None:
        """The certified tails dominate the true Poisson tails."""
        # Oracle: scipy.stats.poisson.cdf(L - 1) and .sf(R - 1).
        cases = {
            1e3: (769, 1251, 1.1830811447669437e-14, 1.2084853925232114e-14),
            1e5: (97601, 102419, 1.2880940397919154e-14, 1.2884575076073695e-14),
        }
        for mean, (left, right, lower, upper) in cases.items():
            got = _poisson_window(mean, 2.5e-13, 2.5e-13)
            self.assertEqual(got[:2], (left, right))
            self.assertLessEqual(lower, got[2])
            self.assertLessEqual(upper, got[3])
            self.assertLessEqual(got[2], 2.5e-13)
            self.assertLessEqual(got[3], 2.5e-13)
        left, right, tau_lower, tau_upper = _poisson_window(1e7, 1e-15, 1e-15)
        self.assertLess(left, 1e7)
        self.assertGreater(right, 1e7)
        self.assertGreater(tau_lower, 0.0)
        self.assertLessEqual(tau_upper, 1e-15)
        self.assertEqual(_poisson_window(1e-8, 1e-12, 1e-12)[0], 0)


class RewardTests(unittest.TestCase):
    """Accumulated rewards and occupancy times."""

    def test_occupancy_against_van_loan(self) -> None:
        """``int_0^3 pi(s) ds`` and a signed cost-rate reward."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        for method in ("series", "squaring"):
            occupancy = chain.expected_time_in_states(FOUR_START, 3.0, method=method)
            error = l1(occupancy.occupancy, FOUR_OCCUPANCY_T3)
            self.assertLess(error, 1e-13)
            self.assertLessEqual(error, occupancy.error_bound)
            self.assertAlmostEqual(math.fsum(occupancy.occupancy), 3.0, places=13)
        rewards = {"a": 3.0, "b": -1.0, "c": 0.5, "d": 10.0}
        reward = chain.expected_accumulated_reward(FOUR_START, 3.0, rewards)
        exact = math.fsum(
            r * o
            for r, o in zip((3.0, -1.0, 0.5, 10.0), FOUR_OCCUPANCY_T3, strict=True)
        )
        self.assertIsInstance(reward, BoundedValue)
        self.assertLessEqual(abs(reward.value - exact), reward.error_bound)
        self.assertLess(abs(reward.value - exact), 1e-12)

    def test_stiff_occupancy(self) -> None:
        """Stiff chain at q t = 1e5 against the high-precision oracle."""
        chain = ContinuousTimeMarkovChain((0, 1, 2, 3), STIFF_RATES)
        for method in ("series", "squaring"):
            occupancy = chain.expected_time_in_states(0, 100.0, method=method)
            error = l1(occupancy.occupancy, STIFF_OCCUPANCY_T100)
            self.assertLessEqual(error, occupancy.error_bound)
            self.assertLess(error, 1e-9)
            rare = occupancy.total((3,))
            self.assertLessEqual(
                abs(rare.value - STIFF_OCCUPANCY_T100[3]), rare.error_bound
            )

    def test_expected_failures_of_one_unit(self) -> None:
        """``lambda int_0^t A(s) ds`` in closed form."""
        failure, repair, horizon = 0.4, 3.0, 2.5
        chain = ContinuousTimeMarkovChain(
            ("up", "down"), {("up", "down"): failure, ("down", "up"): repair}
        )
        total = failure + repair
        uptime = repair * horizon / total + failure / total**2 * (
            1.0 - math.exp(-total * horizon)
        )
        count = chain.expected_transition_count("up", horizon, [("up", "down")])
        self.assertLess(abs(count.value - failure * uptime), 1e-14)
        self.assertLessEqual(abs(count.value - failure * uptime), count.error_bound)
        with self.assertRaises(ValueError):
            chain.expected_transition_count("up", 1.0, [("up", "up")])


class LongRunTests(unittest.TestCase):
    """GTH stationary laws, limits, absorption, and the jump chain."""

    def test_stationary_law(self) -> None:
        """GTH agrees with the eigenvector oracle and the long-run transient."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        stationary = chain.steady_state()
        self.assertLess(l1(stationary, FOUR_STATIONARY), 1e-15)
        self.assertTrue(chain.is_irreducible())

    def test_birth_death_product_form(self) -> None:
        """Stiff birth-death chain: GTH keeps relative accuracy at 1e-24."""
        births, deaths = (1e-6, 2e-6, 3e-6), (1e3, 2e3, 3e3)
        rates = {}
        for j in range(3):
            rates[(j, j + 1)] = births[j]
            rates[(j + 1, j)] = deaths[j]
        chain = ContinuousTimeMarkovChain((0, 1, 2, 3), rates)
        weights = [1.0]
        for j in range(3):
            weights.append(weights[-1] * births[j] / deaths[j])
        expected = [w / math.fsum(weights) for w in weights]
        for got, want in zip(chain.steady_state(), expected, strict=True):
            self.assertLess(abs(got - want) / want, 1e-14)

    def test_several_closed_classes(self) -> None:
        """The stationary law is not unique; the limit depends on the start."""
        chain = ContinuousTimeMarkovChain(
            ("left", "mid", "right"), {("mid", "left"): 1.0, ("mid", "right"): 3.0}
        )
        with self.assertRaises(ValueError):
            chain.steady_state()
        self.assertEqual(chain.absorbing_states(), ("left", "right"))
        limit = chain.limiting_distribution("mid")
        self.assertAlmostEqual(limit[0], 0.25, places=15)
        self.assertAlmostEqual(limit[2], 0.75, places=15)

    def test_absorption_against_discrete_chain(self) -> None:
        """A birth-death chain with absorbing ends versus its jump chain."""
        up, down, size = 0.4, 0.6, 6
        rates = {}
        for j in range(1, size - 1):
            rates[(j, j + 1)] = up
            rates[(j, j - 1)] = down
        chain = ContinuousTimeMarkovChain(tuple(range(size)), rates)
        analysis = chain.absorption({2: 0.5, 3: 0.5})
        jump = chain.embedded_chain().absorption()
        for start in range(1, size - 1):
            for target in (0, size - 1):
                self.assertAlmostEqual(
                    analysis.probability(start, target),
                    jump.probability(start, target),
                    places=14,
                )
            # Constant exit rate: mean time = expected jumps / (up + down).
            self.assertAlmostEqual(
                analysis.mean_time(start), jump.steps(start) / (up + down), places=13
            )
        ratio = down / up
        win = (1 - ratio**2) / (1 - ratio ** (size - 1))
        self.assertAlmostEqual(analysis.probability(2, size - 1), win, places=14)
        self.assertAlmostEqual(
            analysis.initial_mean_time,
            0.5 * analysis.mean_time(2) + 0.5 * analysis.mean_time(3),
            places=14,
        )
        self.assertAlmostEqual(math.fsum(analysis.initial_absorption), 1.0, places=14)

    def test_stiff_mean_time_to_absorption(self) -> None:
        """``(3 lambda + mu) / (2 lambda^2)`` with lambda = 1e-6, mu = 1e3."""
        lam, mu = 1e-6, 1e3
        chain = ContinuousTimeMarkovChain(
            (0, 1, 2), {(0, 1): 2 * lam, (1, 0): mu, (1, 2): lam}
        )
        mttf = chain.absorption(0).initial_mean_time
        exact = (3 * lam + mu) / (2 * lam**2)
        self.assertLess(abs(mttf - exact) / exact, 1e-14)

    def test_absorption_requires_absorbing_closed_classes(self) -> None:
        """A recurrent class of two states must be made absorbing first."""
        chain = ContinuousTimeMarkovChain(
            ("a", "b", "c"), {("a", "b"): 1.0, ("b", "c"): 1.0, ("c", "b"): 1.0}
        )
        with self.assertRaises(ValueError):
            chain.absorption("a")
        absorbed = chain.make_absorbing(("b",))
        self.assertEqual(absorbed.absorption("a").probability("a", "b"), 1.0)

    def test_embedded_chain(self) -> None:
        """Jump probabilities ``q_ij / -q_ii``; absorbing states stay put."""
        chain = ContinuousTimeMarkovChain(
            ("x", "y", "z"), {("x", "y"): 1.0, ("x", "z"): 3.0}
        )
        jump = chain.embedded_chain()
        self.assertIsInstance(jump, MarkovChain)
        self.assertEqual(jump.transition[0], (0.0, 0.25, 0.75))
        self.assertEqual(jump.transition[1], (0.0, 1.0, 0.0))


class SimulationCrossCheckTests(unittest.TestCase):
    """Seeded Monte Carlo agrees with the analytic answers."""

    def test_paths_agree_with_transient_and_occupancy(self) -> None:
        """State frequencies and mean occupancy within 4.5 standard errors."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        paths = 3000
        counts = dict.fromkeys(chain.states, 0)
        time_in_d = []
        for replicate in range(paths):
            path = chain.simulate(FOUR_START, 3.0, seed=1000 + replicate)
            counts[path.state_at(3.0)] += 1
            time_in_d.append(path.time_in("d"))
        for label, exact in zip(chain.states, FOUR_TRANSIENT[3.0], strict=True):
            share = counts[label] / paths
            error = math.sqrt(exact * (1 - exact) / paths)
            self.assertLess(abs(share - exact), 4.5 * error)
        mean = math.fsum(time_in_d) / paths
        spread = math.sqrt(
            math.fsum((x - mean) ** 2 for x in time_in_d) / (paths - 1) / paths
        )
        self.assertLess(abs(mean - FOUR_OCCUPANCY_T3[3]), 4.5 * spread)

    def test_simulation_is_reproducible(self) -> None:
        """The same seed gives the same path."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        first = chain.simulate("a", 5.0, seed=7)
        second = chain.simulate("a", 5.0, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first.times[0], 0.0)


if __name__ == "__main__":
    unittest.main()
