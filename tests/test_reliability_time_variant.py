"""Tests for time-variant reliability: Rice, PHI2, and first passage."""

from __future__ import annotations

import math
import os
import unittest

from probstat.reliability.time_variant import (
    HarmonicProcess,
    Matern32Process,
    first_passage_comparison,
    first_passage_monte_carlo,
    first_passage_subset_simulation,
    phi2_outcrossing_probability,
    phi2_outcrossing_rate,
    phi2_time_variant,
    rice_upcrossing_rate,
)


def pdf(x: float) -> float:
    """Return the standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def cdf(x: float) -> float:
    """Return the standard normal CDF through erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def simpson(f, a: float, b: float, n: int = 4000) -> float:  # type: ignore[no-untyped-def]
    """Integrate ``f`` on ``[a, b]`` by composite Simpson's rule."""
    h = (b - a) / n
    total = f(a) + f(b)
    for k in range(1, n):
        total += (4.0 if k % 2 else 2.0) * f(a + k * h)
    return total * h / 3.0


def outcrossing_probability_oracle(beta1: float, beta2: float, rho: float) -> float:
    """Return ``P(Z1 < b1, Z2 >= b2)`` by a one-dimensional quadrature.

    ``Z2 | Z1 = x`` is ``N(rho x, 1 - rho^2)``; independent of the library's
    bivariate normal routine.
    """
    s = math.sqrt(1.0 - rho * rho)
    return simpson(
        lambda x: pdf(x) * (1.0 - cdf((beta2 - rho * x) / s)), -9.0, beta1, 6000
    )


class RiceFormulaTests(unittest.TestCase):
    """Closed-form Rice mean upcrossing rate."""

    def test_constant_barrier_classical_formula(self) -> None:
        """Nu = sigma_dot exp(-b^2/(2 sigma^2)) / (2 pi sigma)."""
        value = rice_upcrossing_rate(sigma=2.0, sigma_rate=3.0, threshold=4.0)
        self.assertAlmostEqual(value, 3.0 / (2.0 * math.pi * 2.0) * math.exp(-2.0), 14)

    def test_mean_shift_is_equivalent_to_barrier_shift(self) -> None:
        """Only (b - m) matters for a constant mean."""
        a = rice_upcrossing_rate(sigma=1.5, sigma_rate=2.0, threshold=3.0, mean=1.0)
        b = rice_upcrossing_rate(sigma=1.5, sigma_rate=2.0, threshold=2.0)
        self.assertAlmostEqual(a, b, 15)

    def test_moving_barrier_matches_expectation_integral(self) -> None:
        """Nu = f_X(b) E[(X' - b')^+] by an independent quadrature."""
        sigma, rate, b, slope = 1.3, 2.2, 2.0, -0.7
        oracle = (
            pdf(b / sigma)
            / sigma
            * simpson(
                lambda v: max(v - slope, 0.0) * pdf(v / rate) / rate,
                -12 * rate,
                12 * rate,
                40000,
            )
        )
        value = rice_upcrossing_rate(
            sigma=sigma, sigma_rate=rate, threshold=b, threshold_rate=slope
        )
        # kink of (v - b')^+ limits Simpson to about 1e-7 relative
        self.assertAlmostEqual(value / oracle, 1.0, 6)

    def test_barrier_receding_lowers_and_approaching_raises_rate(self) -> None:
        """A falling barrier is crossed more often than a rising one."""
        base = rice_upcrossing_rate(sigma=1.0, sigma_rate=1.0, threshold=2.0)
        rising = rice_upcrossing_rate(
            sigma=1.0, sigma_rate=1.0, threshold=2.0, threshold_rate=1.0
        )
        falling = rice_upcrossing_rate(
            sigma=1.0, sigma_rate=1.0, threshold=2.0, threshold_rate=-1.0
        )
        self.assertLess(rising, base)
        self.assertGreater(falling, base)

    def test_invalid_inputs_raise(self) -> None:
        """Non-positive scales and non-finite values are rejected."""
        with self.assertRaises(ValueError):
            rice_upcrossing_rate(sigma=0.0, sigma_rate=1.0, threshold=1.0)
        with self.assertRaises(ValueError):
            rice_upcrossing_rate(sigma=1.0, sigma_rate=-1.0, threshold=1.0)
        with self.assertRaises(ValueError):
            rice_upcrossing_rate(sigma=1.0, sigma_rate=1.0, threshold=math.inf)
        with self.assertRaises(TypeError):
            rice_upcrossing_rate(sigma="1", sigma_rate=1.0, threshold=1.0)  # type: ignore[arg-type]


class PHI2Tests(unittest.TestCase):
    """PHI2 outcrossing rate on linear Gaussian limit states."""

    def test_probability_matches_independent_quadrature(self) -> None:
        """Phi(-b2) - Phi_2 equals the conditional-integral oracle."""
        for b1, b2, rho in [(2.0, 2.0, 0.9), (1.0, 2.5, 0.3), (3.0, 2.0, 0.99)]:
            value = phi2_outcrossing_probability(b1, b2, rho)
            oracle = outcrossing_probability_oracle(b1, b2, rho)
            self.assertAlmostEqual(value, oracle, delta=1e-9)

    def test_rate_converges_to_rice_for_harmonic_process(self) -> None:
        """For X = cos/sin sum, beta = r and rho = cos(w dt); rate -> Rice."""
        process = HarmonicProcess((1.0,), (2.0,))
        exact = rice_upcrossing_rate(sigma=1.0, sigma_rate=2.0, threshold=1.5)
        errors = []
        for dt in (1e-1, 1e-2, 1e-3):
            rate = phi2_outcrossing_rate(1.5, 1.5, process.correlation(dt), dt)
            errors.append(abs(rate / exact - 1.0))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 2e-3)

    def test_time_variant_search_matches_rice_for_harmonic(self) -> None:
        """Two FORM runs per grid time reproduce the closed-form rate."""
        process = HarmonicProcess((1.0,), (2.0,))
        result = phi2_time_variant(
            process.limit_state(1.5), 2, 3.0, steps=6, dt=1e-3, start=(0.3, 0.4)
        )
        exact = rice_upcrossing_rate(sigma=1.0, sigma_rate=2.0, threshold=1.5)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "converged")
        for rate, beta in zip(result.rates, result.betas, strict=True):
            self.assertAlmostEqual(rate / exact, 1.0, delta=1e-4)
            self.assertAlmostEqual(beta, 1.5, 7)
        self.assertAlmostEqual(result.initial_probability, cdf(-1.5), 10)
        self.assertAlmostEqual(
            result.upper_bound, cdf(-1.5) + exact * 3.0, delta=1e-4 * exact * 3.0
        )
        self.assertLess(result.poisson_probability, result.upper_bound)

    def test_moving_barrier_matches_rice_with_slope(self) -> None:
        """G = r + c t - X(t): beta(t) varies and nu follows Rice with b' = c."""
        process = HarmonicProcess((1.0, 0.5), (1.0, 3.0))
        slope = 0.4

        def g(u: tuple[float, ...], t: float) -> float:
            return 1.2 + slope * t - process.value(u, t)

        result = phi2_time_variant(g, 4, 2.0, steps=4, dt=1e-4, start=(0.1,) * 4)
        sigma, rate = process.sigma, process.sigma_rate
        self.assertTrue(result.converged)
        for t, nu, beta in zip(result.times, result.rates, result.betas, strict=True):
            exact = rice_upcrossing_rate(
                sigma=sigma,
                sigma_rate=rate,
                threshold=1.2 + slope * t,
                threshold_rate=slope,
            )
            self.assertAlmostEqual(beta, (1.2 + slope * t) / sigma, 7)
            self.assertAlmostEqual(nu / exact, 1.0, delta=2e-3)

    def test_union_bound_covers_exact_single_harmonic_first_passage(self) -> None:
        """A cosine of Rayleigh amplitude R fails over a period iff R >= r.

        Exact P_f = exp(-r^2/2) for unit variance and T = one period; the
        PHI2 bound P_f(0) + nu T must be at least that.
        """
        process = HarmonicProcess((1.0,), (2.0 * math.pi,))
        for r in (2.0, 3.0):
            result = phi2_time_variant(
                process.limit_state(r), 2, 1.0, steps=4, dt=1e-4, start=(0.2, 0.3)
            )
            exact = math.exp(-0.5 * r * r)
            self.assertGreaterEqual(result.upper_bound, exact)
            self.assertLess(result.upper_bound, 1.25 * exact)

    def test_failure_cases(self) -> None:
        """Invalid arguments, vanishing gradient, and non-convergence."""
        with self.assertRaises(ValueError):
            phi2_outcrossing_rate(1.0, 1.0, 0.5, 0.0)
        with self.assertRaises(ValueError):
            phi2_outcrossing_probability(1.0, 1.0, 1.5)
        with self.assertRaises(TypeError):
            phi2_time_variant("g", 2, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            phi2_time_variant(lambda u, t: 1.0 + 0.0 * t, 2, 1.0)  # zero gradient
        with self.assertRaises(ValueError):
            phi2_time_variant(lambda u, t: 1.0 - u[0], 2, 1.0, start=(0.0,))
        nonlinear = phi2_time_variant(
            lambda u, t: 3.0 - u[0] - 0.5 * u[1] ** 2 + 0.0 * t,
            2,
            1.0,
            steps=1,
            start=(0.5, 0.5),
            max_iterations=1,
        )
        self.assertFalse(nonlinear.converged)
        self.assertIn("FORM", nonlinear.stop_reason)


class ProcessTests(unittest.TestCase):
    """Grid representations of the Gaussian processes."""

    def test_matern_path_covariance_is_the_kernel(self) -> None:
        """The path map M z has covariance M M^T equal to the Matern-3/2 kernel."""
        process = Matern32Process(1.7, 0.8, mean=0.3)
        steps, h = 8, 0.15
        dim = process.noise_dimension(steps)
        base = process.path([0.0] * dim, h, steps)
        columns = []
        for j in range(dim):
            unit = [0.0] * dim
            unit[j] = 1.0
            columns.append(
                [a - b for a, b in zip(process.path(unit, h, steps), base, strict=True)]
            )
        for i in range(steps + 1):
            self.assertAlmostEqual(base[i], 0.3, 14)
            for k in range(steps + 1):
                cov = sum(col[i] * col[k] for col in columns)
                expected = 1.7**2 * process.correlation(abs(i - k) * h)
                self.assertAlmostEqual(cov, expected, 11)

    def test_matern_moments_match_rice_inputs(self) -> None:
        """sigma_rate = sqrt(3) sigma / ell equals sqrt(-k''(0))."""
        process = Matern32Process(2.0, 1.5)
        eps = 1e-3
        curvature = 2.0 * (1.0 - process.correlation(eps)) * process.sigma**2 / eps**2
        self.assertAlmostEqual(math.sqrt(curvature), process.sigma_rate, delta=2e-3)

    def test_harmonic_moments(self) -> None:
        """Variance, derivative variance, and correlation of the sum."""
        process = HarmonicProcess((1.0, 0.5), (1.0, 3.0))
        self.assertAlmostEqual(process.sigma, math.sqrt(1.25), 14)
        self.assertAlmostEqual(process.sigma_rate, math.sqrt(1.0 + 0.25 * 9.0), 14)
        self.assertAlmostEqual(
            process.correlation(0.7), (math.cos(0.7) + 0.25 * math.cos(2.1)) / 1.25, 14
        )

    def test_invalid_processes_raise(self) -> None:
        """Non-positive parameters, wrong lengths, singular steps."""
        with self.assertRaises(ValueError):
            Matern32Process(-1.0, 1.0)
        with self.assertRaises(ValueError):
            HarmonicProcess((1.0,), (1.0, 2.0))
        with self.assertRaises(ValueError):
            HarmonicProcess((), ())
        process = Matern32Process(1.0, 1.0)
        with self.assertRaises(ValueError):
            process.path([0.0] * 5, 0.1, 2)
        with self.assertRaises(ValueError):
            process.path([0.0] * 6, 1e-6, 2)  # closed-form innovation unreliable


class FirstPassageTests(unittest.TestCase):
    """Trajectory-space first-passage estimators."""

    def test_monte_carlo_matches_exact_single_harmonic(self) -> None:
        """P_f = exp(-r^2/2) for one period of a unit-variance cosine."""
        process = HarmonicProcess((1.0,), (2.0 * math.pi,))
        r = 1.5
        result = first_passage_monte_carlo(
            process, r, horizon=1.0, steps=200, samples=4000, seed=11
        )
        exact = math.exp(-0.5 * r * r)
        # grid maximum is R cos(delta) with delta <= pi/200: relative bias 1e-4
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)
        self.assertTrue(result.converged)
        self.assertEqual(result.evaluations, 4000)

    def test_subset_simulation_matches_exact_single_harmonic(self) -> None:
        """Rare event exp(-r^2/2) = 2.2e-3 recovered within its c.o.v."""
        process = HarmonicProcess((1.0,), (2.0 * math.pi,))
        r = 3.5
        exact = math.exp(-0.5 * r * r)
        result = first_passage_subset_simulation(
            process,
            r,
            horizon=1.0,
            steps=100,
            samples_per_level=500,
            sampler="cs",
            seed=5,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)
        self.assertLess(result.evaluations, 5000)

    def test_seed_reproducibility_and_no_failure_status(self) -> None:
        """Equal seeds give equal estimates; zero failures is flagged."""
        process = Matern32Process(1.0, 1.0)
        a = first_passage_monte_carlo(
            process, 1.0, horizon=2.0, steps=20, samples=200, seed=3
        )
        b = first_passage_monte_carlo(
            process, 1.0, horizon=2.0, steps=20, samples=200, seed=3
        )
        self.assertEqual(a.probability, b.probability)
        none = first_passage_monte_carlo(
            process, 9.0, horizon=2.0, steps=20, samples=100, seed=3
        )
        self.assertFalse(none.converged)
        self.assertEqual(none.probability, 0.0)
        self.assertTrue(math.isnan(none.standard_error))

    def test_time_varying_barrier_is_accepted(self) -> None:
        """A lower barrier than a constant one cannot reduce first passage."""
        process = Matern32Process(1.0, 1.0)
        high = first_passage_monte_carlo(
            process, 2.0, horizon=2.0, steps=10, samples=500, seed=4
        )
        sloped = first_passage_monte_carlo(
            process,
            [2.0 - 0.1 * k for k in range(11)],
            horizon=2.0,
            steps=10,
            samples=500,
            seed=4,
        )
        self.assertGreaterEqual(sloped.probability, high.probability)

    def test_invalid_arguments(self) -> None:
        """Wrong types, sizes, and barrier lengths are rejected."""
        process = Matern32Process(1.0, 1.0)
        with self.assertRaises(TypeError):
            first_passage_monte_carlo(
                object(),
                1.0,
                horizon=1.0,
                steps=5,
                samples=10,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            first_passage_monte_carlo(
                process, [1.0, 2.0], horizon=1.0, steps=5, samples=10
            )
        with self.assertRaises(ValueError):
            first_passage_monte_carlo(process, 1.0, horizon=0.0, steps=5, samples=10)
        with self.assertRaises(ValueError):
            first_passage_monte_carlo(process, 1.0, horizon=1.0, steps=5, samples=1)
        with self.assertRaises(TypeError):
            first_passage_monte_carlo(
                process,
                True,
                horizon=1.0,
                steps=5,
                samples=10,  # type: ignore[arg-type]
            )


class ComparisonTests(unittest.TestCase):
    """Rice/PHI2 bounds against trajectory estimators on a Matern process."""

    def test_fast_comparison_orders_the_quantities(self) -> None:
        """Grid estimates obey the grid union bound; Rice bound fails at low b."""
        process = Matern32Process(1.0, 1.0)
        rows = first_passage_comparison(
            process,
            [0.0, 2.0, 3.0],
            horizon=5.0,
            steps=50,
            mc_samples=1000,
            subset_samples=300,
            seed=21,
        )
        low, mid, high = rows
        for row in rows:
            self.assertLessEqual(row.rice_poisson, row.rice_bound + 1e-12)
            self.assertLessEqual(
                row.monte_carlo.probability,
                row.grid_union_bound + 4.0 * row.monte_carlo.standard_error,
            )
            # PHI2 with dt = h/10 reproduces the Rice bound to O(dt)
            self.assertAlmostEqual(row.phi2_bound / row.rice_bound, 1.0, delta=0.03)
        # low threshold: union bound exceeds one, crossings cluster
        self.assertGreater(low.rice_bound, 1.0)
        self.assertLess(low.monte_carlo.probability, 1.0)
        # high threshold: Poisson/bound within a few percent of the grid estimate
        self.assertLess(
            abs(high.rice_poisson - high.monte_carlo.probability),
            4.0 * high.monte_carlo.standard_error + 0.15 * high.rice_poisson,
        )
        self.assertGreater(mid.rice_bound, mid.monte_carlo.probability)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_slow_threshold_sweep_report(self) -> None:
        """Print the full threshold sweep (Matern 3/2, T = 10 ell)."""
        process = Matern32Process(1.0, 1.0)
        rows = first_passage_comparison(
            process,
            [0.0, 1.0, 2.0, 3.0, 3.5],
            horizon=10.0,
            steps=200,
            mc_samples=20000,
            subset_samples=1000,
            seed=1,
        )
        for row in rows:
            print(
                f"b={row.threshold:.1f} rice_bound={row.rice_bound:.4g} "
                f"poisson={row.rice_poisson:.4g} phi2={row.phi2_bound:.4g} "
                f"grid_union={row.grid_union_bound:.4g} "
                f"mc={row.monte_carlo.probability:.4g}"
                f"+-{row.monte_carlo.standard_error:.2g} "
                f"ss={row.subset.probability:.4g} "
                f"cov={row.subset.coefficient_of_variation:.2g}"
            )
            self.assertLessEqual(
                row.monte_carlo.probability,
                row.grid_union_bound + 4.0 * row.monte_carlo.standard_error,
            )


if __name__ == "__main__":
    unittest.main()
