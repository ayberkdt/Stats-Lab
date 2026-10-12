"""Tests for exact limit-theorem results and their simulation counterparts."""

from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from probstat.distributions import Binomial, Gamma
from probstat.monte_carlo import (
    simulate_clt_distance,
    simulate_running_means,
    simulate_weak_law,
)
from probstat.probability import (
    BERRY_ESSEEN_CONSTANT,
    bernoulli_summand,
    bernstein_bound,
    berry_esseen_bound,
    chebyshev_bound,
    clt_error_profile,
    compare_tail_bounds,
    delta_method_bernoulli,
    delta_method_exponential_reciprocal,
    delta_method_normal_exp,
    delta_method_normal_square,
    edgeworth_cdf,
    exponential_summand,
    hoeffding_bound,
    kolmogorov_distance_to_normal,
    normal_summand,
    sample_mean_two_sided_tail,
    sample_mean_upper_tail,
    standardized_mean_cdf,
)
from probstat.probability.limit_theorems import _binomial_pmf


def phi(z: float) -> float:
    """Return the standard normal CDF for independent cross-checks."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def exponential_draw(rng: random.Random) -> float:
    """Draw one Exponential(1) summand."""
    return rng.expovariate(1.0)


class BerryEsseenTests(unittest.TestCase):
    """Exact Kolmogorov distances against the Berry-Esseen bound."""

    def test_constant_and_moments(self) -> None:
        """Shevtsova's constant and closed-form third absolute moments."""
        self.assertEqual(BERRY_ESSEEN_CONSTANT, 0.4748)
        self.assertAlmostEqual(
            exponential_summand(1.0).third_absolute_central_moment,
            12.0 / math.e - 2.0,
            places=14,
        )
        half = bernoulli_summand(0.5)
        self.assertAlmostEqual(half.third_absolute_central_moment, 0.125, places=15)
        self.assertAlmostEqual(
            berry_esseen_bound(0.125, 0.5, 100), 0.04748, places=15
        )
        self.assertAlmostEqual(exponential_summand(3.0).skewness, 2.0, places=14)

    def test_bound_dominates_exact_bernoulli_distance(self) -> None:
        """For several p and n the exact distance stays below the bound."""
        worst_ratio = 0.0
        for p in (0.5, 0.3, 0.1, 0.01):
            summand = bernoulli_summand(p)
            for row in clt_error_profile(summand, (1, 2, 5, 10, 30, 100, 500)):
                self.assertLess(row.kolmogorov_normal, row.berry_esseen)
                worst_ratio = max(worst_ratio, row.kolmogorov_normal / row.berry_esseen)
        # Fair-coin means come closest: n = 100 gives 0.0398 against 0.0475.
        self.assertGreater(worst_ratio, 0.8)
        self.assertLess(worst_ratio, 1.0)

    def test_fair_coin_distance_is_half_the_central_atom(self) -> None:
        """For p = 1/2 and n = 10 the distance is C(10, 5) / 2^11."""
        distance = kolmogorov_distance_to_normal(bernoulli_summand(0.5), 10)
        self.assertAlmostEqual(distance, math.comb(10, 5) / 2**11, places=14)

    def test_lattice_distance_matches_brute_force_evaluation(self) -> None:
        """Jump-point evaluation agrees with a dense one-sided scan."""
        summand = bernoulli_summand(0.2)
        n = 15
        scale = math.sqrt(n * 0.2 * 0.8)
        brute = 0.0
        for k in range(n + 1):
            z = (k - n * 0.2) / scale
            for point in (z - 1e-9, z):
                brute = max(
                    brute, abs(standardized_mean_cdf(summand, n, point) - phi(point))
                )
        exact = kolmogorov_distance_to_normal(summand, n)
        self.assertAlmostEqual(exact, brute, delta=1e-8)

    def test_exponential_distance_matches_dense_grid(self) -> None:
        """Grid-plus-golden maximization agrees with a 40001-point scan."""
        summand = exponential_summand(1.0)
        n = 5
        brute = max(
            abs(standardized_mean_cdf(summand, n, z) - phi(z))
            for z in (-4.0 + 12.0 * i / 40000 for i in range(40001))
        )
        refined = kolmogorov_distance_to_normal(summand, n)
        self.assertGreaterEqual(refined, brute - 1e-12)
        self.assertLess(refined - brute, 1e-6)

    def test_exponential_and_normal_summands(self) -> None:
        """The bound holds for exponential sums; normal sums are exact."""
        for row in clt_error_profile(exponential_summand(2.0), (1, 3, 10, 100)):
            self.assertLess(row.kolmogorov_normal, row.berry_esseen)
        normal = normal_summand(1.0, 2.0)
        self.assertEqual(kolmogorov_distance_to_normal(normal, 7), 0.0)
        self.assertAlmostEqual(
            standardized_mean_cdf(normal_summand(1.0, 2.0), 7, 0.3), phi(0.3)
        )

    def test_exact_cdfs_match_distribution_objects(self) -> None:
        """Binomial and gamma CDFs from the distribution catalog agree."""
        n, p, z = 20, 0.35, 0.4
        threshold = n * p + z * math.sqrt(n * p * (1 - p))
        self.assertAlmostEqual(
            standardized_mean_cdf(bernoulli_summand(p), n, z),
            Binomial(n, p).cdf(math.floor(threshold)),
            places=13,
        )
        rate = 2.5
        mean_value = 1.0 / rate + z / (rate * math.sqrt(n))
        self.assertAlmostEqual(
            standardized_mean_cdf(exponential_summand(rate), n, z),
            Gamma(n, n * rate).cdf(mean_value),
            places=12,
        )


class EdgeworthTests(unittest.TestCase):
    """The one-term Edgeworth correction for skewed continuous sums."""

    def test_edgeworth_improves_on_normal_for_exponential_sums(self) -> None:
        """Exact distances: n=10 gives 0.0421 (normal) vs 0.0080 (Edgeworth)."""
        rows = clt_error_profile(exponential_summand(1.0), (2, 5, 10, 30, 100))
        for row in rows:
            assert row.kolmogorov_edgeworth is not None
            self.assertLess(row.kolmogorov_edgeworth, 0.65 * row.kolmogorov_normal)
        by_n = {row.n: row for row in rows}
        self.assertAlmostEqual(by_n[10].kolmogorov_normal, 0.04211, delta=5e-5)
        edge10 = by_n[10].kolmogorov_edgeworth
        edge100 = by_n[100].kolmogorov_edgeworth
        assert edge10 is not None and edge100 is not None
        self.assertAlmostEqual(edge10, 0.00798, delta=5e-5)
        # Edgeworth error is O(1/n): a tenfold n cuts it about tenfold.
        self.assertLess(edge100 / edge10, 0.15)

    def test_normal_error_has_the_edgeworth_leading_constant(self) -> None:
        """sqrt(n) sup|F_n - Phi| -> gamma phi(0) / 6 = 1/(3 sqrt(2 pi))."""
        distance = kolmogorov_distance_to_normal(exponential_summand(1.0), 1000)
        limit = 1.0 / (3.0 * math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(math.sqrt(1000) * distance, limit, delta=2e-3)

    def test_edgeworth_formula_and_lattice_rejection(self) -> None:
        """The correction vanishes at z = +-1 and lattice summands raise."""
        self.assertAlmostEqual(edgeworth_cdf(1.0, 4, 2.0), phi(1.0), places=15)
        self.assertAlmostEqual(
            edgeworth_cdf(0.0, 4, 2.0),
            0.5 + 2.0 / (6.0 * 2.0) / math.sqrt(2.0 * math.pi),
            places=15,
        )
        with self.assertRaises(ValueError):
            kolmogorov_distance_to_normal(bernoulli_summand(0.3), 10, edgeworth=True)


class TailBoundTests(unittest.TestCase):
    """Chebyshev, Hoeffding, and Bernstein bounds against exact tails."""

    def test_hoeffding_dominates_exact_binomial_tails(self) -> None:
        """All bounds lie above exact one- and two-sided binomial tails."""
        for p in (0.5, 0.3, 0.05):
            summand = bernoulli_summand(p)
            for n in (10, 50, 200, 1000):
                for t in (0.02, 0.05, 0.1, 0.2):
                    row = compare_tail_bounds(summand, n, t)
                    assert row.hoeffding is not None and row.bernstein is not None
                    self.assertLessEqual(row.exact_upper, row.hoeffding)
                    self.assertLessEqual(row.exact_upper, row.bernstein)
                    self.assertLessEqual(row.exact_two_sided, row.chebyshev)
                    self.assertLessEqual(
                        row.exact_two_sided,
                        hoeffding_bound(0.0, 1.0, n, t, two_sided=True),
                    )

    def test_exact_binomial_tail_matches_catalog(self) -> None:
        """P(Xbar - p >= t) equals the binomial survival function."""
        n, p, t = 100, 0.3, 0.1
        expected = Binomial(n, p).sf(math.ceil(n * (p + t)) - 1)
        self.assertAlmostEqual(
            sample_mean_upper_tail(bernoulli_summand(p), n, t), expected, places=14
        )
        # Reported in the guide: exact 0.02099 vs Hoeffding 0.1353.
        row = compare_tail_bounds(bernoulli_summand(p), n, t)
        self.assertAlmostEqual(row.exact_upper, 0.020989, delta=1e-6)
        self.assertAlmostEqual(row.hoeffding or 0.0, math.exp(-2.0), places=14)

    def test_exponential_tails_and_unbounded_support(self) -> None:
        """Gamma tails are exact and bounded-support bounds are withheld."""
        n, t = 40, 0.25
        exact = sample_mean_two_sided_tail(exponential_summand(1.0), n, t)
        gamma = Gamma(n, n)
        self.assertAlmostEqual(
            exact, 1.0 - gamma.cdf(1.0 + t) + gamma.cdf(1.0 - t), places=12
        )
        row = compare_tail_bounds(exponential_summand(1.0), n, t)
        self.assertIsNone(row.hoeffding)
        self.assertIsNone(row.bernstein)
        self.assertLessEqual(row.exact_two_sided, row.chebyshev)

    def test_bound_formulas(self) -> None:
        """Closed forms of the three inequalities."""
        self.assertAlmostEqual(chebyshev_bound(0.25, 100, 0.1), 0.25)
        self.assertAlmostEqual(
            hoeffding_bound(0.0, 2.0, 50, 0.2), math.exp(-2.0 * 50 * 0.04 / 4.0)
        )
        self.assertAlmostEqual(
            bernstein_bound(0.1, 0.5, 80, 0.1, two_sided=True),
            2.0 * math.exp(-80 * 0.01 / (2.0 * (0.1 + 0.5 * 0.1 / 3.0))),
        )
        self.assertEqual(chebyshev_bound(1.0, 1, 0.1), 1.0)


class DeltaMethodTests(unittest.TestCase):
    """Delta-method variances against exact small-sample variances."""

    def test_exact_formulas_match_simulation(self) -> None:
        """Var(Xbar^2) for normal data, n = 5, within Monte Carlo error."""
        rng = random.Random(20261012)
        mu, sigma, n, reps = 1.0, 1.0, 5, 40000
        values = []
        for _ in range(reps):
            mean = math.fsum(rng.gauss(mu, sigma) for _ in range(n)) / n
            values.append(mean * mean)
        center = math.fsum(values) / reps
        variance = math.fsum((v - center) ** 2 for v in values) / (reps - 1)
        fourth = math.fsum((v - center) ** 4 for v in values) / reps
        se = math.sqrt((fourth - variance**2) / reps)
        comparison = delta_method_normal_square(mu, sigma, n)
        self.assertAlmostEqual(comparison.exact_variance, 0.88, places=14)
        self.assertAlmostEqual(comparison.delta_variance, 0.8, places=14)
        self.assertLess(abs(variance - comparison.exact_variance), 4.0 * se)

    def test_ratio_tends_to_one(self) -> None:
        """Exact/delta ratios shrink toward one as n grows."""
        for build in (
            lambda n: delta_method_normal_square(1.0, 1.0, n),
            lambda n: delta_method_normal_exp(0.0, 1.0, n),
            lambda n: delta_method_exponential_reciprocal(1.0, n),
        ):
            ratios = [build(n).ratio for n in (5, 50, 500)]
            self.assertGreater(ratios[0], ratios[1])
            self.assertGreater(ratios[1], ratios[2])
            self.assertLess(abs(ratios[2] - 1.0), 0.02)
        self.assertAlmostEqual(
            delta_method_exponential_reciprocal(1.0, 5).exact_variance, 25.0 / 48.0
        )

    def test_first_order_failure_at_a_stationary_point(self) -> None:
        """g(p) = p(1-p) at p = 1/2 has g' = 0 but positive exact variance."""
        comparison = delta_method_bernoulli(lambda x: x * (1.0 - x), 0.0, 0.5, 10)
        self.assertEqual(comparison.delta_variance, 0.0)
        self.assertGreater(comparison.exact_variance, 0.0)
        self.assertEqual(comparison.ratio, math.inf)


class SimulationTests(unittest.TestCase):
    """Simulation helpers agree with the exact results they illustrate."""

    def test_weak_law_frequencies_match_exact_tails(self) -> None:
        """Simulated P(|Xbar - 1| > 0.2) agrees with the gamma tail."""
        points = simulate_weak_law(
            exponential_draw, 1.0, 0.2, (10, 50, 200), replications=1500, seed=11
        )
        for point in points:
            exact = sample_mean_two_sided_tail(exponential_summand(1.0), point.n, 0.2)
            spread = math.sqrt(exact * (1.0 - exact) / 1500)
            self.assertLess(abs(point.probability - exact), 4.0 * spread + 1e-3)
        self.assertGreater(points[0].probability, points[-1].probability)

    def test_running_mean_uniform_deviation(self) -> None:
        """Finite-horizon sup-deviation frequencies are monotone in m."""
        result = simulate_running_means(
            exponential_draw, 1.0, 0.2, (10, 100, 1000), paths=150, seed=4
        )
        self.assertEqual(len(result.paths), 150)
        self.assertEqual(len(result.paths[0]), 3)
        freq = result.sup_exceedance
        self.assertGreaterEqual(freq[0], freq[1])
        self.assertGreaterEqual(freq[1], freq[2])
        pointwise = sample_mean_two_sided_tail(exponential_summand(1.0), 10, 0.2)
        self.assertGreater(freq[0], pointwise - 4.0 * math.sqrt(0.25 / 150))
        again = simulate_running_means(
            exponential_draw, 1.0, 0.2, (10, 100, 1000), paths=150, seed=4
        )
        self.assertEqual(again, result)

    def test_clt_simulation_is_near_the_exact_distance(self) -> None:
        """Simulated KS distance is within KS sampling noise of the exact one."""
        reps = 4000
        sim = simulate_clt_distance(
            exponential_draw, 1.0, 1.0, 10, replications=reps, seed=5
        )
        exact = kolmogorov_distance_to_normal(exponential_summand(1.0), 10)
        self.assertLess(abs(sim.ks_distance - exact), 1.63 / math.sqrt(reps))
        self.assertAlmostEqual(sim.sampling_scale, 1.0 / math.sqrt(reps))


class LargeSampleAccuracyTests(unittest.TestCase):
    """Exactness claims hold for very large ``n`` (review regressions)."""

    def test_exponential_distance_at_huge_n(self) -> None:
        """The grid step grew like ``sqrt(n)/1600`` and missed the peak.

        Oracles from mpmath (34 digits): ``P(n, n + z sqrt(n))`` by quadrature
        of the gamma density, maximized by golden-section search. At
        ``n = 1e8`` the old value was ``5.934e-6`` against ``1.3298e-5``.
        """
        summand = exponential_summand(1.0)
        normal = kolmogorov_distance_to_normal(summand, 10**8)
        self.assertLess(abs(normal / 1.32980760155051e-5 - 1.0), 1e-8)
        edge = kolmogorov_distance_to_normal(summand, 10**8, edgeworth=True)
        self.assertLess(abs(edge / 6.34526965746175e-10 - 1.0), 1e-6)
        edge7 = kolmogorov_distance_to_normal(summand, 10**7, edgeworth=True)
        self.assertLess(abs(edge7 / 6.34615076983067e-9 - 1.0), 1e-6)

    def test_binomial_masses_keep_relative_precision(self) -> None:
        """``lgamma`` differences lost ``n eps``: ``1.3e-8`` at ``n = 4e6``.

        Oracles from mpmath (40 digits) for ``C(n, k) p^k (1 - p)^(n - k)``
        with the double ``p``.
        """
        fair = _binomial_pmf(4 * 10**6, 0.5)
        self.assertLess(abs(fair[2 * 10**6] / 0.00039894225546754093 - 1.0), 1e-13)
        skewed = _binomial_pmf(10**6, 0.3)
        for k, oracle in (
            (300000, 0.00087056315463668078),
            (299000, 8.0449410686192802e-5),
            (302500, 3.0638089393926125e-10),
        ):
            with self.subTest(k=k):
                self.assertLess(abs(skewed[k] / oracle - 1.0), 1e-12)
        self.assertEqual(skewed[0], 0.0)
        self.assertAlmostEqual(math.fsum(skewed), 1.0, places=13)

    def test_fair_coin_distance_at_a_million(self) -> None:
        """``n = 1e6``: half the central atom ``C(n, n/2) / 2^(n+1)``.

        Oracle ``3.9894218066587504e-4`` from mpmath (40 digits); the old
        masses gave a relative error of ``7e-7``.
        """
        distance = kolmogorov_distance_to_normal(bernoulli_summand(0.5), 10**6)
        self.assertLess(abs(distance / 0.00039894218066587504 - 1.0), 1e-12)

    def test_moderate_binomial_masses_match_rational_arithmetic(self) -> None:
        """Saddle-point masses against exact rational masses.

        The relative error stays below ``4 eps``-ish per unit of
        ``|log p(k)|``: far-tail masses come from ``exp`` of a large
        exponent, whose rounding is amplified by its size.
        """
        cases = ((1, 0.5), (2, 0.01), (10, 0.3), (40, 0.9999), (200, 0.003))
        for n, p in (*cases, (1000, 0.5)):
            masses = _binomial_pmf(n, p)
            exact = Fraction(p)
            for k, mass in enumerate(masses):
                oracle = float(math.comb(n, k) * exact**k * (1 - exact) ** (n - k))
                if oracle == 0.0:
                    continue
                with self.subTest(n=n, k=k):
                    allowed = 4e-15 * (1.0 + abs(math.log(oracle))) * oracle
                    self.assertLessEqual(abs(mass - oracle), allowed)


if __name__ == "__main__":
    unittest.main()
