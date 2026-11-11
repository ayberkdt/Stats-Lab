"""Semi-Markov reliability and epistemic common-cause parameters (ADR-0076).

Reference values come from the existing CTMC solver (uniformization with
certified bounds), from closed forms, from adaptive quadrature over the
epistemic laws, and from seeded simulation. No numerical library outside
the package is imported.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Beta, Gamma, LogNormal, Truncated, Uniform, Weibull
from probstat.modeling import IntervalInput, RandomInput
from probstat.numerics import integrate
from probstat.processes import ContinuousTimeMarkovChain, SemiMarkovProcess
from probstat.systems import (
    AlphaFactorGroup,
    BetaFactorGroup,
    ExponentialComponent,
    FixedProbability,
    MarkovComponent,
    MultipleGreekLetterGroup,
    SemiMarkovComponent,
    SemiMarkovReliabilityModel,
    SystemModel,
    alternating_renewal_unit,
    ccf_interval_bounds,
    ccf_parameter_uncertainty,
    k_out_of_n,
    repairable_k_out_of_n,
    series,
    single_repairable_unit,
)
from probstat.workflow import MissionAssessment

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


def two_of_three(beta: float, *, basis: str = "probability") -> SystemModel:
    """Return a 2-of-3 system whose members form one beta-factor group.

    References:
        [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

    """
    return SystemModel(
        k_out_of_n(2, "A", "B", "C"),
        {name: FixedProbability(0.99) for name in "ABC"},
        [BetaFactorGroup(("A", "B", "C"), beta)],
        ccf_basis=basis,  # type: ignore[arg-type]
    )


class ExponentialSojournTests(unittest.TestCase):
    """Exponential sojourns reproduce the CTMC solver."""

    def test_transient_law_matches_uniformization(self) -> None:
        """The state law agrees with ``pi(0) exp(Q t)`` to 1e-8 (L1)."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        process = SemiMarkovProcess.from_ctmc(chain)
        for t in (0.5, 3.0):
            semi = process.transient(FOUR_START, t)
            exact = chain.transient(FOUR_START, t)
            error = math.fsum(
                abs(x - y)
                for x, y in zip(semi.probabilities, exact.probabilities, strict=True)
            )
            self.assertLess(error, 1e-8)
            self.assertLess(error, semi.error_estimate)
        matrix = process.transition_matrix(0.5)
        for row in matrix.matrix:
            self.assertAlmostEqual(math.fsum(row), 1.0, delta=1e-10)

    def test_steady_state_and_mean_times_match(self) -> None:
        """Limiting law and mean first-passage times equal the CTMC values."""
        chain = ContinuousTimeMarkovChain(("a", "b", "c", "d"), FOUR_RATES)
        process = SemiMarkovProcess.from_ctmc(chain)
        for x, y in zip(process.steady_state(), chain.steady_state(), strict=True):
            self.assertAlmostEqual(x, y, delta=1e-14)
        absorbed = chain.make_absorbing(["d"])
        expected = absorbed.absorption(FOUR_START).initial_mean_time
        self.assertAlmostEqual(
            process.mean_first_passage_time(["d"], FOUR_START), expected, delta=1e-12
        )

    def test_repairable_two_out_of_three_measures(self) -> None:
        """A(t), R(t), MTTF and A_inf of a repairable 2-of-3 agree to 1e-8."""
        markov = repairable_k_out_of_n(3, 2, 0.4, 1.5)
        semi = SemiMarkovReliabilityModel(
            SemiMarkovProcess.from_ctmc(markov.chain),
            markov.up_states,
            markov.initial,
        )
        for t in (0.7, 2.5):
            self.assertAlmostEqual(
                semi.availability(t).value, markov.availability(t).value, delta=1e-8
            )
            self.assertAlmostEqual(
                semi.unavailability(t).value,
                markov.unavailability(t).value,
                delta=1e-8,
            )
            self.assertAlmostEqual(
                semi.reliability(t).value, markov.reliability(t).value, delta=1e-8
            )
        self.assertAlmostEqual(
            semi.mean_time_to_failure(),
            markov.mean_time_to_failure(),
            delta=1e-10 * markov.mean_time_to_failure(),
        )
        self.assertAlmostEqual(
            semi.steady_state_availability(),
            markov.steady_state_availability(),
            delta=1e-14,
        )

    def test_component_matches_markov_component(self) -> None:
        """A semi-Markov block equals the Markov block inside a system."""
        markov = single_repairable_unit(0.3, 2.0)
        semi = SemiMarkovReliabilityModel(
            SemiMarkovProcess.from_ctmc(markov.chain), ("up",), "up"
        )
        exact = SystemModel(
            series("unit", "valve"),
            {"unit": MarkovComponent(markov), "valve": FixedProbability(0.995)},
        )
        block = SemiMarkovComponent(semi)
        model = SystemModel(
            series("unit", "valve"),
            {"unit": block, "valve": FixedProbability(0.995)},
        )
        for t in (0.5, 4.0):
            self.assertAlmostEqual(
                model.availability(t), exact.availability(t), delta=1e-8
            )
            self.assertAlmostEqual(
                model.unreliability(t), exact.unreliability(t), delta=1e-8
            )
        self.assertGreater(block.last_error_estimate, 0.0)
        self.assertLess(block.last_error_estimate, 1e-5)


class AlternatingRenewalTests(unittest.TestCase):
    """Weibull up times and lognormal repair times."""

    up = Weibull(1.5, 10.0)
    down = LogNormal(0.0, 0.5)

    def test_steady_state_availability_is_exact(self) -> None:
        """``A_inf = E[U] / (E[U] + E[D])`` and the MTTF is ``E[U]``."""
        unit = alternating_renewal_unit(self.up, self.down)
        expected = self.up.mean / (self.up.mean + self.down.mean)
        self.assertAlmostEqual(unit.steady_state_availability(), expected, delta=1e-15)
        self.assertAlmostEqual(
            unit.steady_state_unavailability(), 1.0 - expected, delta=1e-15
        )
        self.assertAlmostEqual(unit.mean_time_to_failure(), self.up.mean, delta=1e-12)
        self.assertAlmostEqual(
            unit.reliability(6.0).value, self.up.sf(6.0), delta=1e-12
        )

    def test_transient_availability_against_simulation(self) -> None:
        """The renewal solution lies within 4.5 standard errors of simulation."""
        unit = alternating_renewal_unit(self.up, self.down)
        curve = unit.availability_curve(18.0, steps=180)
        times = (3.0, 9.0, 18.0)
        simulated = unit.process.monte_carlo_probabilities(
            "up", ["up"], times, samples=6000, seed=2026
        )
        for t, estimate, error in zip(
            times, simulated.estimates, simulated.standard_errors, strict=True
        ):
            value = curve.values[round(t / 0.1)]
            self.assertLess(abs(value - estimate), 4.5 * error)
        self.assertLess(curve.max_error_estimate, 1e-4)
        point = unit.availability(9.0)
        self.assertAlmostEqual(
            point.value + unit.unavailability(9.0).value, 1.0, delta=1e-9
        )
        path = unit.process.simulate("up", 50.0, seed=1)
        self.assertEqual(path.states[0], "up")

    def test_block_inside_mission_assessment(self) -> None:
        """A semi-Markov block is usable in ``SystemModel`` and missions."""
        block = SemiMarkovComponent(alternating_renewal_unit(self.up, self.down))
        system = SystemModel(
            series("pump", "valve"),
            {"pump": block, "valve": ExponentialComponent(1e-3)},
        )
        result = MissionAssessment(system, 5.0, compute_mttf=False).reliability()
        expected = self.up.sf(5.0) * math.exp(-5e-3)
        self.assertAlmostEqual(result.reliability, expected, delta=1e-10)


class AbsorptionTests(unittest.TestCase):
    """Mean time to absorption from the fundamental matrix."""

    def test_three_state_closed_form(self) -> None:
        """``m = (I - P_TT)^-1 tau`` on a 3-state example, and simulation."""
        laws = {
            ("1", "2"): (0.7, Weibull(2.0, 1.5)),
            ("1", "3"): (0.3, LogNormal(0.2, 0.4)),
            ("2", "1"): (0.4, Gamma(3.0, 2.0)),
            ("2", "3"): (0.6, Weibull(0.8, 2.0)),
        }
        process = SemiMarkovProcess(("1", "2", "3"), laws)
        tau1 = 0.7 * laws[("1", "2")][1].mean + 0.3 * laws[("1", "3")][1].mean
        tau2 = 0.4 * laws[("2", "1")][1].mean + 0.6 * laws[("2", "3")][1].mean
        determinant = 1.0 - 0.7 * 0.4
        m1 = (tau1 + 0.7 * tau2) / determinant
        m2 = (0.4 * tau1 + tau2) / determinant
        analysis = process.absorption("1")
        self.assertAlmostEqual(analysis.mean_time("1"), m1, delta=1e-13 * m1)
        self.assertAlmostEqual(analysis.mean_time("2"), m2, delta=1e-13 * m2)
        self.assertAlmostEqual(analysis.initial_mean_time, m1, delta=1e-13 * m1)
        self.assertAlmostEqual(analysis.fundamental[0][0], 1.0 / determinant, 12)
        self.assertAlmostEqual(process.mean_first_passage_time(["3"], "1"), m1, 12)
        generator = random.Random(5)
        total = 0.0
        squares = 0.0
        count = 3000
        for _ in range(count):
            path = process.simulate("1", 1e6, seed=generator.getrandbits(62))
            total += path.times[-1]
            squares += path.times[-1] ** 2
        mean = total / count
        error = math.sqrt((squares / count - mean * mean) / count)
        self.assertLess(abs(mean - m1), 4.5 * error)
        cdf = process.first_passage_curve(["3"], 12.0, "1", steps=120)
        self.assertGreater(cdf.values[-1], 0.9)
        self.assertEqual(cdf.values[0], 0.0)


class EpistemicCCFTests(unittest.TestCase):
    """Uncertain CCF parameters propagated to the system reliability."""

    law = Beta(2.0, 18.0)

    def test_exact_moments_match_quadrature(self) -> None:
        """Exact ``E[R]`` and ``Var[R]`` equal integrals over the beta law."""
        system = two_of_three(0.1)
        result = ccf_parameter_uncertainty(system, 1.0, {(0, "beta"): self.law})
        self.assertEqual(result.method, "exact")

        def reliability(beta: float) -> float:
            """Return R(beta) of the 2-of-3 system.

            References:
                [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

            """
            return two_of_three(beta).reliability(1.0)

        mean = integrate(
            lambda b: reliability(b) * self.law.pdf(b), 0.0, 1.0, rel_tol=1e-13
        ).value
        variance = integrate(
            lambda b: (reliability(b) - mean) ** 2 * self.law.pdf(b),
            0.0,
            1.0,
            rel_tol=1e-12,
        ).value
        self.assertAlmostEqual(result.mean, mean, delta=1e-13)
        self.assertAlmostEqual(result.variance, variance, delta=1e-9 * variance)
        self.assertAlmostEqual(result.mean + result.unreliability_mean, 1.0, 14)
        self.assertEqual(result.first_order["0:beta"], 1.0)
        self.assertEqual(result.ccf_contribution, 1.0)
        sampled = ccf_parameter_uncertainty(
            system, 1.0, {(0, "beta"): self.law}, method="monte-carlo", samples=2000
        )
        self.assertLess(abs(sampled.mean - mean), 4.5 * sampled.mean_standard_error)

    def test_truncated_lognormal_uses_quadrature_moments(self) -> None:
        """A non-beta law on [0, 1] gives exact moments by quadrature."""
        law = Truncated(LogNormal(math.log(0.08), 0.6), 0.0, 1.0)
        result = ccf_parameter_uncertainty(
            two_of_three(0.1), 1.0, {(0, "beta"): RandomInput("beta", law)}
        )
        mean = integrate(
            lambda b: two_of_three(b).reliability(1.0) * law.pdf(b),
            0.0,
            1.0,
            rel_tol=1e-12,
        ).value
        self.assertAlmostEqual(result.mean, mean, delta=1e-11)

    def test_variance_share_of_ccf_parameter(self) -> None:
        """Exact first-order shares agree with the Saltelli estimate."""
        system = two_of_three(0.1)
        parameters = {(0, "beta"): self.law, (0, "total"): Beta(2.0, 198.0)}
        exact = ccf_parameter_uncertainty(system, 1.0, parameters)
        sampled = ccf_parameter_uncertainty(
            system, 1.0, parameters, method="monte-carlo", samples=2000, seed=0
        )
        self.assertEqual(exact.method, "exact")
        self.assertGreater(exact.ccf_contribution, 0.0)
        self.assertLess(exact.ccf_contribution, 1.0)
        self.assertAlmostEqual(exact.ccf_contribution, exact.first_order["0:beta"])
        self.assertAlmostEqual(
            exact.ccf_variance, exact.ccf_contribution * exact.variance
        )
        self.assertLessEqual(math.fsum(exact.first_order.values()), 1.0 + 1e-12)
        # Saltelli estimates scatter by about 0.02 (beta) and 0.06 (total).
        for label, delta in (("0:beta", 0.05), ("0:total", 0.15)):
            self.assertAlmostEqual(
                exact.first_order[label], sampled.first_order[label], delta=delta
            )
        self.assertLess(
            abs(exact.mean - sampled.mean), 4.5 * sampled.mean_standard_error
        )

    def test_mgl_exact_and_hazard_and_alpha_use_monte_carlo(self) -> None:
        """MGL parameters are exact; hazard split and alpha factors sample."""
        mgl = SystemModel(
            k_out_of_n(2, "A", "B", "C"),
            {name: FixedProbability(0.98) for name in "ABC"},
            [MultipleGreekLetterGroup(("A", "B", "C"), (0.1, 0.3))],
            ccf_basis="probability",
        )
        laws = {(0, "beta"): self.law, (0, "gamma"): Beta(3.0, 7.0)}
        exact = ccf_parameter_uncertainty(mgl, 1.0, laws)
        sampled = ccf_parameter_uncertainty(
            mgl, 1.0, laws, method="monte-carlo", samples=1500
        )
        self.assertEqual(exact.method, "exact")
        self.assertLess(
            abs(exact.mean - sampled.mean), 4.5 * sampled.mean_standard_error
        )
        hazard = SystemModel(
            k_out_of_n(2, "A", "B", "C"),
            {name: ExponentialComponent(1e-3) for name in "ABC"},
            [BetaFactorGroup(("A", "B", "C"), 0.1)],
        )
        result = ccf_parameter_uncertainty(
            hazard, 100.0, {(0, "beta"): self.law}, samples=500
        )
        self.assertEqual(result.method, "monte-carlo")
        self.assertEqual(result.basis, "hazard")
        alpha = SystemModel(
            k_out_of_n(2, "A", "B", "C"),
            {name: FixedProbability(0.98) for name in "ABC"},
            [AlphaFactorGroup(("A", "B", "C"), (0.95, 0.03, 0.02))],
        )
        result = ccf_parameter_uncertainty(
            alpha,
            1.0,
            {(0, "alpha_3"): Uniform(0.0, 0.05)},
            measure="availability",
            samples=300,
        )
        self.assertEqual(result.method, "monte-carlo")
        self.assertGreater(result.variance, 0.0)

    def test_interval_bounds_bracket_draws(self) -> None:
        """Interval bounds contain Monte Carlo draws; monotonicity is detected."""
        system = two_of_three(0.1)
        bounds = ccf_interval_bounds(
            system, 1.0, {(0, "beta"): IntervalInput("beta", 0.02, 0.2)}
        )
        self.assertEqual(bounds.monotonicity["0:beta"], "increasing")
        self.assertTrue(bounds.monotone)
        generator = random.Random(7)
        for _ in range(200):
            beta = generator.uniform(0.02, 0.2)
            self.assertTrue(bounds.contains(two_of_three(beta).unreliability(1.0)))
        self.assertAlmostEqual(
            bounds.unreliability_upper, two_of_three(0.2).unreliability(1.0), 15
        )
        self.assertAlmostEqual(
            bounds.unreliability_lower, two_of_three(0.02).unreliability(1.0), 15
        )
        series_system = SystemModel(
            series("A", "B", "C"),
            {name: FixedProbability(0.99) for name in "ABC"},
            [BetaFactorGroup(("A", "B", "C"), 0.1)],
            ccf_basis="probability",
        )
        decreasing = ccf_interval_bounds(series_system, 1.0, {(0, "beta"): (0.0, 0.5)})
        self.assertEqual(decreasing.monotonicity["0:beta"], "decreasing")

    def test_pbox_from_interval_and_distribution(self) -> None:
        """Corner distributions span a probability box that contains draws."""
        system = two_of_three(0.1)
        total = Beta(1.0, 99.0)
        bounds = ccf_interval_bounds(
            system,
            1.0,
            {(0, "beta"): (0.02, 0.2)},
            distributions={(0, "total"): total},
            samples=400,
            seed=3,
        )
        self.assertEqual(len(bounds.corners), 2)
        self.assertEqual(bounds.samples, 400)
        level = sorted(bounds.pbox.sorted_outputs[0])[200]
        low, high = bounds.pbox.bounds(level)
        self.assertLessEqual(low, high)
        self.assertGreater(high - low, 0.0)
        for lower, upper in zip(
            bounds.pbox.lower_cdf, bounds.pbox.upper_cdf, strict=True
        ):
            self.assertLessEqual(lower, upper)


class InvalidInputTests(unittest.TestCase):
    """Invalid specifications raise."""

    def test_semi_markov_validation(self) -> None:
        """Kernel, law, state and step errors are reported."""
        law = Weibull(1.5, 1.0)
        with self.assertRaises(ValueError):
            SemiMarkovProcess(("a", "b"), {("a", "b"): (0.5, law)})
        with self.assertRaises(ValueError):
            SemiMarkovProcess(("a", "b"), {("a", "c"): (1.0, law)})
        with self.assertRaises(ValueError):
            SemiMarkovProcess(("a", "b"), {("a", "b"): (1.0, Uniform(-1.0, 1.0))})
        with self.assertRaises(TypeError):
            SemiMarkovProcess(("a", "b"), {("a", "b"): (1.0, object())})
        with self.assertRaises(TypeError):
            SemiMarkovProcess("ab", {})
        with self.assertRaises(ValueError):
            SemiMarkovProcess(("a", "a"), {})
        process = SemiMarkovProcess(("a", "b"), {("a", "b"): (1.0, law)})
        with self.assertRaises(ValueError):
            process.transient("a", 1.0, steps=1)
        with self.assertRaises(ValueError):
            process.transient("a", -1.0)
        with self.assertRaises(ValueError):
            process.first_passage_curve([], 1.0, "a")
        with self.assertRaises(ValueError):
            SemiMarkovReliabilityModel(process, ("a",), "b")
        with self.assertRaises(ValueError):
            SemiMarkovReliabilityModel(process, ("a", "b"), "a")
        with self.assertRaises(TypeError):
            SemiMarkovComponent(single_repairable_unit(1.0, 1.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SemiMarkovComponent(alternating_renewal_unit(law, law)).reliability(-1.0)

    def test_ccf_validation(self) -> None:
        """Unknown keys, bad laws and inapplicable methods raise."""
        system = two_of_three(0.1)
        law = Beta(2.0, 18.0)
        with self.assertRaises(KeyError):
            ccf_parameter_uncertainty(system, 1.0, {(0, "gamma"): law})
        with self.assertRaises(KeyError):
            ccf_parameter_uncertainty(system, 1.0, {(1, "beta"): law})
        with self.assertRaises(ValueError):
            ccf_parameter_uncertainty(system, 1.0, {})
        with self.assertRaises(ValueError):
            ccf_parameter_uncertainty(system, 1.0, {(0, "beta"): LogNormal(0.0, 1.0)})
        with self.assertRaises(ValueError):
            ccf_parameter_uncertainty(
                two_of_three(0.1, basis="hazard"),
                1.0,
                {(0, "beta"): law},
                method="exact",
            )
        with self.assertRaises(ValueError):
            ccf_parameter_uncertainty(system, 1.0, {(0, "beta"): law}, method="x")
        with self.assertRaises(TypeError):
            ccf_parameter_uncertainty(system, 1.0, {("0", "beta"): law})  # type: ignore[dict-item]
        plain = SystemModel(series("A", "B"), {n: FixedProbability(0.9) for n in "AB"})
        with self.assertRaises(ValueError):
            ccf_parameter_uncertainty(plain, 1.0, {(0, "beta"): law})
        with self.assertRaises(ValueError):
            ccf_interval_bounds(system, 1.0, {(0, "beta"): (0.3, 0.1)})
        with self.assertRaises(ValueError):
            ccf_interval_bounds(
                system,
                1.0,
                {(0, "beta"): (0.1, 0.3)},
                distributions={(0, "beta"): law},
            )
        with self.assertRaises(ValueError):
            ccf_interval_bounds(system, 1.0, {})
        with self.assertRaises(TypeError):
            ccf_interval_bounds(system, 1.0, {(0, "beta"): 0.1})


if __name__ == "__main__":
    unittest.main()
