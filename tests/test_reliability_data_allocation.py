"""Tests for reliability data estimation and reliability allocation.

Literal oracle values were computed once with SciPy 1.18 (scipy.stats.beta,
gamma, chi2, binom; scipy.optimize.minimize with SLSQP and Nelder-Mead) in a
separate verification environment.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Beta, Gamma
from probstat.modeling import RandomInput
from probstat.reliability import (
    MettasCost,
    agree_allocation,
    arinc_allocation,
    cost_optimal_allocation,
    cost_optimal_series_allocation,
    crow_amsaa_fit,
    demand_failure_probability,
    demonstration_test_time,
    equal_apportionment,
    failure_rate_bounds,
    failure_rate_estimate,
    feasibility_of_objectives,
    success_run_sample_size,
)

TIMES = (33.0, 76, 145, 347, 555, 811, 1212, 1499, 1830, 2245, 2799, 3156, 3770, 4401)


class DemandAndRateTests(unittest.TestCase):
    """Beta and Gamma posteriors and classical bounds."""

    def test_demand_failures(self) -> None:
        """Jeffreys Beta(2.5, 48.5) and Clopper-Pearson for 2 of 50 (scipy)."""
        estimate = demand_failure_probability(2, 50)
        self.assertEqual(estimate.posterior, Beta(2.5, 48.5))
        self.assertAlmostEqual(
            estimate.credible_interval[0], 0.0115630489864489, places=11
        )
        self.assertAlmostEqual(
            estimate.credible_interval[1], 0.106328492500747, places=11
        )
        self.assertAlmostEqual(
            estimate.confidence_interval.lower, 0.0071537195312938, places=11
        )
        self.assertAlmostEqual(
            estimate.confidence_interval.upper, 0.120614155422044, places=11
        )
        uniform = demand_failure_probability(2, 50, prior="uniform")
        self.assertAlmostEqual(
            uniform.credible_interval[0], 0.0162234154945384, places=11
        )
        user = demand_failure_probability(0, 10, prior=Beta(1.0, 20.0))
        self.assertEqual(user.posterior, Beta(1.0, 30.0))
        self.assertIsInstance(estimate.random_input("p_demand"), RandomInput)
        with self.assertRaises(ValueError):
            demand_failure_probability(3, 2)

    def test_failure_rate_posteriors(self) -> None:
        """Jeffreys Gamma(r + 1/2, T) and a conjugate user prior (scipy)."""
        estimate = failure_rate_estimate(3, 12000.0)
        self.assertEqual(estimate.posterior, Gamma(3.5, 12000.0))
        self.assertAlmostEqual(
            estimate.credible_interval[0] / 9.03062462207524e-05, 1.0, places=10
        )
        self.assertAlmostEqual(
            estimate.credible_interval[1] / 0.00058613085205584, 1.0, places=10
        )
        user = failure_rate_estimate(3, 12000.0, prior=Gamma(1.5, 1000.0))
        self.assertAlmostEqual(user.posterior.mean, 0.000346153846153846, places=15)
        self.assertAlmostEqual(
            user.credible_interval[1] / 0.000650729907870017, 1.0, places=10
        )

    def test_chi_square_bounds(self) -> None:
        """Time- and failure-terminated bounds equal scipy chi2 quantiles."""
        two = failure_rate_bounds(3, 12000.0)
        self.assertAlmostEqual(two.lower / 6.81409539303294e-05, 1.0, places=11)
        self.assertAlmostEqual(two.upper / 0.000646138043994394, 1.0, places=11)
        failure = failure_rate_bounds(3, 12000.0, termination="failure")
        self.assertAlmostEqual(failure.upper / 0.000524649468489332, 1.0, places=11)
        zero = failure_rate_bounds(0, 12000.0)
        self.assertEqual(zero.lower, 0.0)
        self.assertAlmostEqual(zero.upper / 0.000249644356129499, 1.0, places=11)
        # chi2_{0.95}(2) / (2T) = -ln(0.05) / (2T) * 2 / 2.
        self.assertAlmostEqual(zero.upper, -math.log(0.05) / 12000.0, places=15)
        upper = failure_rate_bounds(3, 12000.0, sided="upper", confidence=0.95)
        self.assertEqual(upper.lower, 0.0)
        self.assertAlmostEqual(upper.upper / 0.000646138043994394, 1.0, places=11)
        self.assertEqual(failure_rate_bounds(3, 1.0, sided="lower").upper, math.inf)
        with self.assertRaises(ValueError):
            failure_rate_bounds(0, 1.0, termination="failure")


class DemonstrationTests(unittest.TestCase):
    """Success-run theorem and exponential test time."""

    def test_success_run(self) -> None:
        """N = ceil(ln(1-C)/ln R); r > 0 by binomial search (scipy)."""
        self.assertEqual(success_run_sample_size(0.9, 0.9), 22)
        self.assertEqual(success_run_sample_size(0.99, 0.95), 299)
        self.assertEqual(success_run_sample_size(0.95, 0.9, allowed_failures=2), 105)
        self.assertEqual(success_run_sample_size(0.9, 0.8, allowed_failures=1), 29)

    def test_test_time(self) -> None:
        """T = chi2_C(2r+2)/(2 lambda) (scipy chi2.ppf)."""
        self.assertAlmostEqual(
            demonstration_test_time(1e-3, 0.9), 2302.58509299405, places=8
        )
        self.assertAlmostEqual(
            demonstration_test_time(1e-3, 0.9, allowed_failures=2),
            5322.32033783421,
            places=7,
        )


class CrowAMSAATests(unittest.TestCase):
    """Power-law NHPP MLE against closed forms and direct maximisation."""

    def test_time_terminated(self) -> None:
        """Beta = n / sum ln(T/t_i); scipy Nelder-Mead on the likelihood agrees."""
        fit = crow_amsaa_fit(TIMES, end_time=5000.0)
        self.assertAlmostEqual(fit.beta, 0.553128793879408, places=12)
        self.assertAlmostEqual(fit.beta, 0.5531287921, places=8)  # direct MLE
        self.assertAlmostEqual(fit.lam, 0.12592768471936, places=12)
        self.assertAlmostEqual(fit.beta_unbiased, 13 / 14 * fit.beta, places=15)
        self.assertAlmostEqual(fit.beta_interval[0], 0.334403396652241, places=11)
        self.assertAlmostEqual(fit.beta_interval[1], 0.816598620290196, places=11)
        self.assertAlmostEqual(fit.instantaneous_mtbf, 645.67757291753, places=8)
        self.assertAlmostEqual(fit.expected_failures(5000.0), 14.0, places=10)
        self.assertAlmostEqual(fit.cumulative_mtbf, 5000.0 / 14)

    def test_failure_terminated(self) -> None:
        """Sum runs to n - 1; unbiased factor (n - 2)/n."""
        fit = crow_amsaa_fit(TIMES)
        self.assertEqual(fit.termination, "failure")
        self.assertAlmostEqual(fit.beta, 0.59513498117214, places=12)
        self.assertAlmostEqual(fit.beta_unbiased, 0.510115698147549, places=12)
        self.assertAlmostEqual(fit.lam, 0.0949999027151292, places=12)
        self.assertAlmostEqual(fit.beta_interval[0], 0.326881216557959, places=11)
        self.assertAlmostEqual(fit.beta_interval[1], 0.826496652292644, places=11)
        with self.assertRaises(ValueError):
            crow_amsaa_fit((1.0, 1.0, 2.0))

    def test_unbiased_estimator_by_simulation(self) -> None:
        """Mean of (n-1)/n beta_hat over simulated time-terminated tests is beta."""
        import random

        rng = random.Random(7)
        beta, lam, horizon = 0.6, 0.5, 100.0
        values = []
        mean_count = lam * horizon**beta
        for _ in range(3000):
            # Poisson count by inversion, then order statistics of u^(1/beta).
            n, p, u = 0, math.exp(-mean_count), rng.random()
            cumulative = p
            while u > cumulative:
                n += 1
                p *= mean_count / n
                cumulative += p
            if n < 3:
                continue
            times = sorted(horizon * rng.random() ** (1 / beta) for _ in range(n))
            values.append(crow_amsaa_fit(times, end_time=horizon).beta_unbiased)
        mean = math.fsum(values) / len(values)
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
        self.assertLess(abs(mean - beta), 4 * sd / math.sqrt(len(values)) + 0.01)


class WeightingAllocationTests(unittest.TestCase):
    """Equal, ARINC, AGREE, and feasibility-of-objectives."""

    def test_equal_and_arinc(self) -> None:
        """Products of allocated reliabilities equal the target."""
        equal = equal_apportionment(0.9, 3, mission_time=100.0)
        self.assertAlmostEqual(math.prod(equal.reliabilities), 0.9, places=14)
        arinc = arinc_allocation(0.95, [1e-4, 2e-4, 5e-4], 100.0)
        self.assertAlmostEqual(arinc.system_reliability, 0.95, places=14)
        rates = arinc.failure_rates or ()
        self.assertAlmostEqual(rates[1] / rates[0], 2.0, places=12)
        self.assertAlmostEqual(math.fsum(rates), -math.log(0.95) / 100.0, places=16)

    def test_agree_exact_and_approximate(self) -> None:
        """Exact form meets the target exactly; approximate is close."""
        args = (0.95, [10, 20, 70], [1.0, 0.9, 0.8], [100.0, 80.0, 100.0])
        exact = agree_allocation(*args)
        self.assertAlmostEqual(exact.system_reliability, 0.95, places=13)
        approx = agree_allocation(*args, form="approximate")
        self.assertAlmostEqual(
            approx.failure_rates[0], 10 * -math.log(0.95) / (100 * 100.0)
        )  # type: ignore[index]
        self.assertLess(abs(approx.system_reliability - 0.95), 5e-4)
        # Unit with E = 1: exact and approximate coincide.
        self.assertAlmostEqual(
            exact.failure_rates[0], approx.failure_rates[0], places=15
        )  # type: ignore[index]

    def test_feasibility_of_objectives(self) -> None:
        """Weights are rating products."""
        result = feasibility_of_objectives(0.95, [[5, 6, 5, 5], [7, 6, 10, 2]], 100.0)
        rates = result.failure_rates or ()
        self.assertAlmostEqual(rates[1] / rates[0], 840 / 750, places=12)
        self.assertAlmostEqual(result.system_reliability, 0.95, places=14)


COSTS = (
    MettasCost(0.1, 0.80, 0.999),
    MettasCost(0.5, 0.85, 0.999),
    MettasCost(0.9, 0.70, 0.995),
)


class CostOptimalAllocationTests(unittest.TestCase):
    """Lagrange (series) and augmented Lagrangian (any structure)."""

    def test_mettas_derivative(self) -> None:
        """Analytic derivative matches a central difference."""
        cost = COSTS[0]
        h = 1e-7
        numeric = (cost(0.95 + h) - cost(0.95 - h)) / (2 * h)
        self.assertAlmostEqual(cost.derivative(0.95) / numeric, 1.0, places=6)
        self.assertAlmostEqual(
            math.exp(cost.log_derivative(0.95)), cost.derivative(0.95)
        )
        self.assertEqual(cost(0.8), 1.0)

    def test_series_lagrange_matches_slsqp_and_kkt(self) -> None:
        """Exact KKT solution vs scipy SLSQP; R_i c_i'(R_i) is constant."""
        exact = cost_optimal_series_allocation(0.9, COSTS)
        self.assertTrue(exact.constraint_satisfied)
        self.assertAlmostEqual(exact.system_reliability, 0.9, places=12)
        for value, oracle in zip(
            exact.reliabilities, (0.94647929, 0.97065596, 0.979639), strict=True
        ):
            self.assertAlmostEqual(value, oracle, places=6)
        self.assertAlmostEqual(exact.cost, 26.8821496159, delta=1e-5)
        marginals = [
            r * c.derivative(r) for r, c in zip(exact.reliabilities, COSTS, strict=True)
        ]
        self.assertLess(max(marginals) / min(marginals) - 1.0, 1e-8)

    def test_general_solver_agrees_with_lagrange_on_series(self) -> None:
        """Augmented Lagrangian on prod R_i reproduces the exact optimum."""
        exact = cost_optimal_series_allocation(0.9, COSTS)
        general = cost_optimal_allocation(0.9, COSTS, lambda r: r[0] * r[1] * r[2])
        self.assertTrue(general.constraint_satisfied)
        self.assertGreaterEqual(general.system_reliability, 0.9)
        self.assertAlmostEqual(general.cost, exact.cost, places=6)
        for a, b in zip(general.reliabilities, exact.reliabilities, strict=True):
            self.assertAlmostEqual(a, b, places=6)

    def test_series_parallel_structure(self) -> None:
        """R1 (1 - (1-R2)(1-R3)) >= 0.95 against scipy SLSQP."""
        result = cost_optimal_allocation(
            0.95, COSTS, lambda r: r[0] * (1 - (1 - r[1]) * (1 - r[2]))
        )
        self.assertTrue(result.constraint_satisfied)
        for value, oracle in zip(
            result.reliabilities, (0.95193783, 0.9410546, 0.96546452), strict=True
        ):
            self.assertAlmostEqual(value, oracle, places=5)
        # SLSQP stops slightly infeasible (g = 0.94999996), hence its lower cost.
        self.assertAlmostEqual(result.cost, 22.9268635589, delta=1e-4)

    def test_unreachable_target(self) -> None:
        """Targets above the product of upper bounds are rejected."""
        with self.assertRaises(ValueError):
            cost_optimal_series_allocation(0.999, COSTS)


if __name__ == "__main__":
    unittest.main()
