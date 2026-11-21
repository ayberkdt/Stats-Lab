"""Rank-1 lattice rules, CBC/Korobov construction, and mean/median RQMC.

Oracles: exact rational arithmetic for the points, enumeration of the dual
lattice for trigonometric exactness, the closed form ``gamma / (6 n^2)`` of
the criterion in one dimension, a quadrature check of the reproducing
property of the kernel the criterion is stated for, a Monte Carlo average
over shifts of the kernel worst-case error, brute-force minimization over
all generating vectors, integrands with known integrals, and seeded
replicate studies for unbiasedness, error decay, and interval coverage.
"""

from __future__ import annotations

import cmath
import itertools
import math
import statistics
import unittest
from collections.abc import Callable
from fractions import Fraction

from probstat.foundations import RandomStream
from probstat.sampling import (
    LatticeRule,
    RQMCResult,
    cbc_generating_vector,
    korobov_generating_vector,
    korobov_vector,
    lattice_squared_error,
    median_random_lattice,
    monte_carlo_design,
    rank1_lattice,
    rqmc_mean,
    rqmc_median,
)

Point = tuple[float, ...]


def bernoulli2(x: float) -> float:
    """Return the Bernoulli polynomial ``B_2(x) = x^2 - x + 1/6``."""
    return x * x - x + 1.0 / 6.0


def sobolev_kernel(x: Point, y: Point, weights: tuple[float, ...]) -> float:
    """Return the unanchored Sobolev kernel the criterion is stated for."""
    value = 1.0
    for xj, yj, gamma in zip(x, y, weights, strict=True):
        value *= 1.0 + gamma * (
            (xj - 0.5) * (yj - 0.5) + 0.5 * bernoulli2(abs(xj - yj))
        )
    return value


def kernel_worst_case_squared(
    points: tuple[Point, ...], weights: tuple[float, ...]
) -> float:
    """Return ``-1 + n^-2 sum_ik K(x_i, x_k)`` (``int K(x, y) dy = 1``)."""
    n = len(points)
    total = math.fsum(sobolev_kernel(x, y, weights) for x in points for y in points)
    return total / (n * n) - 1.0


def bessel_i0(c: float) -> float:
    """Return the modified Bessel function ``I_0(c)`` by its power series."""
    return sum((c / 2.0) ** (2 * k) / math.factorial(k) ** 2 for k in range(40))


def periodic(c: float, s: int) -> Callable[[Point], float]:
    """Return ``prod_j exp(c sin(2 pi x_j)) / I_0(c)``, which integrates to 1."""
    norm = bessel_i0(c) ** s

    def integrand(x: Point) -> float:
        """Evaluate the smooth periodic product integrand."""
        return math.exp(c * sum(math.sin(2.0 * math.pi * v) for v in x)) / norm

    return integrand


def linear_product(x: Point) -> float:
    """Return ``prod_j (1 + c_j (x_j - 1/2))`` with ``c_j = j / 2``; integral 1."""
    return math.prod(1.0 + 0.5 * (j + 1) * (v - 0.5) for j, v in enumerate(x))


def quadratic_product(x: Point) -> float:
    """Return ``prod_j (1 + B_2(x_j))``; integral 1."""
    return math.prod(1.0 + bernoulli2(v) for v in x)


def xexp_product(x: Point) -> float:
    """Return ``prod_j x_j exp(x_j)``; integral 1 because ``int x e^x = 1``."""
    return math.prod(v * math.exp(v) for v in x)


def rmse(estimates: list[float], truth: float = 1.0) -> float:
    """Return the root mean squared error of estimates of ``truth``."""
    return math.sqrt(statistics.fmean((e - truth) ** 2 for e in estimates))


class LatticePointTests(unittest.TestCase):
    """The point set, its group structure, and the random shift."""

    def test_points_are_the_exact_rationals(self) -> None:
        """Unshifted points equal ``(k z_j mod n) / n`` in rational arithmetic."""
        n, z = 12, (1, 5, 7)
        design = rank1_lattice(n, z)
        self.assertEqual((design.size, design.dimension), (12, 3))
        self.assertEqual(
            (design.method, design.randomized, design.seed), ("lattice", False, None)
        )
        for k, point in enumerate(design.points):
            for zj, x in zip(z, point, strict=True):
                exact = Fraction(k * zj, n) % 1
                self.assertEqual(x, float(exact))
        self.assertEqual(design.points[0], (0.0, 0.0, 0.0))

    def test_projections_and_group_closure(self) -> None:
        """Each coordinate visits every ``m / n`` once; sums stay in the set."""
        n, z = 13, (1, 5, 8)
        points = rank1_lattice(n, z).points
        for axis in range(3):
            self.assertEqual(sorted(round(p[axis] * n) for p in points), list(range(n)))
        residues = {tuple(round(v * n) for v in p) for p in points}
        for a in residues:
            for b in residues:
                total = tuple((u + v) % n for u, v in zip(a, b, strict=True))
                self.assertIn(total, residues)

    def test_generating_vector_is_reduced_modulo_n(self) -> None:
        """``z`` and ``z + n`` (and negative representatives) give one lattice."""
        self.assertEqual(
            rank1_lattice(7, (1, 3)).points, rank1_lattice(7, (8, -4)).points
        )

    def test_shift_is_one_common_rotation(self) -> None:
        """Shifted minus unshifted points is one vector modulo one, in (0, 1)."""
        n, z = 31, (1, 12, 9)
        plain = rank1_lattice(n, z).points
        shifted = rank1_lattice(n, z, shift=True, seed=5)
        self.assertTrue(shifted.randomized)
        self.assertEqual(shifted.seed, 5)
        delta = shifted.points[0]
        for p, q in zip(plain, shifted.points, strict=True):
            for x, y, d in zip(p, q, delta, strict=True):
                self.assertTrue(0.0 < y < 1.0)
                gap = (y - x - d) % 1.0
                self.assertLess(min(gap, 1.0 - gap), 1e-12)
        self.assertEqual(shifted.points, rank1_lattice(n, z, shift=True, seed=5).points)
        self.assertEqual(
            shifted.points,
            rank1_lattice(n, z, shift=True, seed=RandomStream(5)).points,
        )
        self.assertNotEqual(
            shifted.points, rank1_lattice(n, z, shift=True, seed=6).points
        )

    def test_shifted_points_are_uniform_on_average(self) -> None:
        """The mean first coordinate of the first point over shifts is 1/2."""
        draws = [
            rank1_lattice(5, (1, 2), shift=True, seed=s).points[0][0]
            for s in range(2000)
        ]
        error = statistics.fmean(draws) - 0.5
        self.assertLess(abs(error), 4.0 * math.sqrt(1.0 / 12.0 / 2000))

    def test_trigonometric_exactness_by_enumeration(self) -> None:
        """``exp(2 pi i h.x)`` is integrated exactly unless ``h.z = 0 mod n``."""
        for n, z in ((7, (1, 3)), (11, (1, 3, 4)), (12, (1, 5, 7))):
            points = rank1_lattice(n, z).points
            dual = 0
            for h in itertools.product(range(-5, 6), repeat=len(z)):
                average = (
                    sum(
                        cmath.exp(
                            2j * math.pi * sum(a * x for a, x in zip(h, p, strict=True))
                        )
                        for p in points
                    )
                    / n
                )
                in_dual = sum(a * b for a, b in zip(h, z, strict=True)) % n == 0
                dual += in_dual
                # The exact integral is 1 for h = 0 and 0 otherwise; the rule
                # returns 1 on the whole dual lattice and 0 off it.
                self.assertAlmostEqual(
                    abs(average - (1.0 if in_dual else 0.0)), 0.0, places=11
                )
            self.assertGreater(dual, 1)

    def test_input_validation(self) -> None:
        """Bad sizes and generating vectors are rejected with reasons."""
        with self.assertRaises(TypeError):
            rank1_lattice(7.0, (1, 2))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            rank1_lattice(0, (1,))
        with self.assertRaises(ValueError):
            rank1_lattice(7, ())
        with self.assertRaises(TypeError):
            rank1_lattice(7, (1, 2.0))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "not coprime"):
            rank1_lattice(12, (1, 3))
        with self.assertRaisesRegex(ValueError, "not coprime"):
            rank1_lattice(7, (1, 7))


class CriterionTests(unittest.TestCase):
    """The shift-averaged squared worst-case error and what it means."""

    def test_one_dimension_closed_form(self) -> None:
        """In one dimension ``e^2 = gamma / (6 n^2)`` for every admissible z."""
        for n, z, gamma in ((7, 3, 1.0), (12, 5, 0.3), (101, 37, 2.5), (1, 1, 1.0)):
            value = lattice_squared_error(n, (z,), weights=(gamma,))
            self.assertAlmostEqual(value, gamma / (6 * n * n), delta=1e-15)

    def test_single_sum_equals_double_sum_of_shift_invariant_kernel(self) -> None:
        """The group structure collapses the pair sum; checked by enumeration."""
        weights = (1.0, 0.6, 0.2)
        for n, z in ((9, (1, 2, 4)), (11, (1, 3, 4)), (16, (1, 7, 5))):
            points = rank1_lattice(n, z).points
            pairs = math.fsum(
                math.prod(
                    1.0 + g * bernoulli2((xj - yj) % 1.0)
                    for xj, yj, g in zip(x, y, weights, strict=True)
                )
                for x in points
                for y in points
            )
            self.assertAlmostEqual(
                lattice_squared_error(n, z, weights=weights),
                pairs / (n * n) - 1.0,
                delta=1e-13,
            )

    def test_kernel_reproduces_under_the_stated_inner_product(self) -> None:
        """``(int f)(int K_y) + (1/gamma) int f' K_y' = f(y)`` for ``f = exp``."""
        gamma, cells = 0.7, 2000

        def derivative(x: float, y: float) -> float:
            """Return ``d/dx K(x, y)`` of the one-dimensional kernel."""
            sign = 1.0 if x > y else -1.0
            return gamma * ((y - 0.5) + (abs(x - y) - 0.5) * sign)

        for y in (0.0, 0.23, 0.5, 0.9):
            total = 0.0
            # Midpoint rule on each side of the kink at x = y.
            for lower, upper in ((0.0, y), (y, 1.0)):
                if upper <= lower:
                    continue
                h = (upper - lower) / cells
                total += h * math.fsum(
                    math.exp(lower + (i + 0.5) * h)
                    * derivative(lower + (i + 0.5) * h, y)
                    for i in range(cells)
                )
            # int f = e - 1 and int K(x, y) dx = 1.
            self.assertAlmostEqual(
                (math.e - 1.0) + total / gamma, math.exp(y), places=6
            )

    def test_criterion_is_the_average_over_shifts_of_the_worst_case_error(self) -> None:
        """A Monte Carlo average over shifts of the kernel error matches e^2."""
        weights = (1.0, 0.5)
        for n, z in ((7, (1, 3)), (8, (1, 5))):
            values = [
                kernel_worst_case_squared(
                    rank1_lattice(n, z, shift=True, seed=seed).points, weights
                )
                for seed in range(600)
            ]
            mean = statistics.fmean(values)
            se = statistics.stdev(values) / math.sqrt(len(values))
            target = lattice_squared_error(n, z, weights=weights)
            self.assertLess(abs(mean - target), 4.0 * se)
            self.assertLess(se, 0.05 * target)
            # The unshifted lattice (one particular shift) has a different
            # worst-case error: the criterion is an average, not a constant.
            unshifted = kernel_worst_case_squared(rank1_lattice(n, z).points, weights)
            self.assertGreater(abs(unshifted - target), 10.0 * se)

    def test_criterion_does_not_depend_on_a_shift_or_on_sign(self) -> None:
        """``e^2`` is a function of ``(n, z, gamma)``; ``z`` and ``-z`` agree."""
        weights = (1.0, 1.0, 1.0)
        base = lattice_squared_error(31, (1, 12, 9), weights=weights)
        self.assertAlmostEqual(
            base, lattice_squared_error(31, (30, 19, 9), weights=weights), delta=1e-15
        )
        self.assertEqual(base, lattice_squared_error(31, (1, 12, 9)))
        # A vector with two equal components is far worse.
        self.assertGreater(lattice_squared_error(31, (1, 1, 9)), 3.0 * base)

    def test_validation(self) -> None:
        """Weights must be positive, finite, and one per dimension."""
        with self.assertRaisesRegex(ValueError, "one entry per dimension"):
            lattice_squared_error(7, (1, 2), weights=(1.0,))
        with self.assertRaisesRegex(ValueError, "positive"):
            lattice_squared_error(7, (1, 2), weights=(1.0, 0.0))
        with self.assertRaises(ValueError):
            lattice_squared_error(7, (1, 2), weights=(1.0, math.inf))


class ConstructionTests(unittest.TestCase):
    """Component-by-component and Korobov searches against brute force."""

    def brute_force(self, n: int, s: int, weights: tuple[float, ...]) -> float:
        """Return the minimum of the criterion over all of ``{1..n-1}^s``."""
        return min(
            lattice_squared_error(n, z, weights=weights)
            for z in itertools.product(range(1, n), repeat=s)
        )

    def test_cbc_is_globally_optimal_in_one_and_two_dimensions(self) -> None:
        """For ``s <= 2`` the CBC criterion equals the brute-force minimum."""
        for n in (5, 7, 11, 13, 17):
            for s in (1, 2):
                weights = (1.0, 0.5)[:s]
                rule = cbc_generating_vector(n, s, weights=weights)
                self.assertAlmostEqual(
                    rule.squared_error, self.brute_force(n, s, weights), delta=1e-15
                )

    def test_cbc_in_three_dimensions_against_brute_force(self) -> None:
        """Greedy CBC cannot beat the optimum; here it attains it."""
        weights = (1.0, 0.5, 0.25)
        for n in (7, 11, 13):
            rule = cbc_generating_vector(n, 3, weights=weights)
            best = self.brute_force(n, 3, weights)
            self.assertGreaterEqual(rule.squared_error, best - 1e-15)
            # Observed on these cases, not a property of the greedy search.
            self.assertAlmostEqual(rule.squared_error, best, delta=1e-15)

    def test_cbc_result_is_consistent(self) -> None:
        """Reported values match the criterion of the returned vector."""
        rule = cbc_generating_vector(251, 4)
        self.assertIsInstance(rule, LatticeRule)
        self.assertEqual(rule.generating_vector, (1, 70, 97, 118))
        self.assertEqual(
            (rule.size, rule.dimension, rule.construction), (251, 4, "cbc")
        )
        self.assertIsNone(rule.multiplier)
        self.assertEqual(rule.candidates_evaluated, 1 + 3 * 125)
        self.assertEqual(rule.weights, (1.0,) * 4)
        for d in range(1, 5):
            self.assertAlmostEqual(
                rule.squared_error_by_dimension[d - 1],
                lattice_squared_error(251, rule.generating_vector[:d]),
                delta=1e-15,
            )
        self.assertAlmostEqual(
            rule.squared_error_by_dimension[0], 1 / (6 * 251**2), delta=1e-15
        )
        self.assertEqual(rule.squared_error, rule.squared_error_by_dimension[-1])
        self.assertAlmostEqual(rule.worst_case_error**2, rule.squared_error)
        # Adding a coordinate cannot make the worst case easier.
        history = rule.squared_error_by_dimension
        self.assertTrue(all(a < b for a, b in zip(history, history[1:], strict=False)))
        self.assertEqual(
            rule.points().points, rank1_lattice(251, rule.generating_vector).points
        )
        # Each greedy step is a true minimum over its candidates.
        for d in (2, 3, 4):
            head = rule.generating_vector[: d - 1]
            best = min(lattice_squared_error(251, (*head, c)) for c in range(1, 251))
            self.assertAlmostEqual(history[d - 1], best, delta=1e-15)

    def test_cbc_is_not_worse_than_korobov_on_the_test_set(self) -> None:
        """CBC beats a fixed Korobov vector and the best one (observed)."""
        for n, s in ((127, 3), (251, 4), (509, 4)):
            cbc = cbc_generating_vector(n, s)
            fixed = lattice_squared_error(n, korobov_vector(n, s, 76))
            best = korobov_generating_vector(n, s)
            self.assertLess(cbc.squared_error, fixed)
            self.assertLessEqual(cbc.squared_error, best.squared_error)
            self.assertLessEqual(best.squared_error, fixed)

    def test_korobov_search_is_exhaustive_over_multipliers(self) -> None:
        """The chosen multiplier minimizes the criterion over all ``a``."""
        weights = (1.0, 0.7, 0.4)
        for n in (11, 13, 31):
            rule = korobov_generating_vector(n, 3, weights=weights)
            values = {
                a: lattice_squared_error(n, korobov_vector(n, 3, a), weights=weights)
                for a in range(1, n)
            }
            self.assertAlmostEqual(
                rule.squared_error, min(values.values()), delta=1e-15
            )
            self.assertEqual(rule.construction, "korobov")
            assert rule.multiplier is not None
            self.assertAlmostEqual(
                values[rule.multiplier], rule.squared_error, delta=1e-15
            )
            self.assertEqual(
                rule.generating_vector, korobov_vector(n, 3, rule.multiplier)
            )
            self.assertEqual(rule.candidates_evaluated, n // 2)

    def test_korobov_vector(self) -> None:
        """Powers of the multiplier modulo ``n``."""
        self.assertEqual(korobov_vector(13, 4, 5), (1, 5, 12, 8))
        self.assertEqual(korobov_vector(13, 1, 5), (1,))
        with self.assertRaisesRegex(ValueError, "not coprime"):
            korobov_vector(12, 3, 4)
        with self.assertRaises(ValueError):
            korobov_vector(1, 3, 1)
        with self.assertRaises(TypeError):
            korobov_vector(13, 3, 2.0)  # type: ignore[arg-type]

    def test_weights_change_the_construction(self) -> None:
        """Decaying weights give a different rule with a smaller criterion."""
        flat = cbc_generating_vector(127, 4)
        decaying = cbc_generating_vector(127, 4, weights=(1.0, 0.25, 1 / 9, 1 / 16))
        self.assertLess(decaying.squared_error, flat.squared_error)
        self.assertEqual(decaying.weights, (1.0, 0.25, 1 / 9, 1 / 16))

    def test_construction_validation(self) -> None:
        """Composite sizes and bad dimensions are rejected."""
        for build in (cbc_generating_vector, korobov_generating_vector):
            with self.assertRaisesRegex(ValueError, "prime"):
                build(12, 2)
            for composite in (1, 4, 9, 91, 1021 * 1021):
                with self.assertRaisesRegex(ValueError, "prime"):
                    build(composite, 2)
            with self.assertRaises(ValueError):
                build(7, 0)
            with self.assertRaises(TypeError):
                build(7, 2.0)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                build(7, 2, weights=(1.0, -1.0))
        self.assertEqual(cbc_generating_vector(2, 3).generating_vector, (1, 1, 1))
        self.assertEqual(korobov_generating_vector(2, 2).multiplier, 1)

    def test_error_decay_for_a_smooth_periodic_integrand(self) -> None:
        """The unshifted CBC lattice error falls far faster than ``n^-1/2``."""
        f = periodic(1.0, 3)
        errors = {}
        for n in (31, 127, 509):
            points = cbc_generating_vector(n, 3).points().points
            errors[n] = abs(math.fsum(f(p) for p in points) / n - 1.0)
        self.assertLess(errors[31], 0.1)
        self.assertLess(errors[127], 1e-3 * errors[31] * 4.0)
        self.assertLess(errors[509], 1e-8)
        # Monte Carlo would gain a factor 4 from 31 to 509 points.
        self.assertGreater(errors[31] / errors[509], 1e6)


class RandomizedIntegrationTests(unittest.TestCase):
    """Mean and median of independently randomized rules."""

    def test_result_fields_and_cost(self) -> None:
        """The mean result reports replicates, SE, interval, and ``R n``."""
        result = rqmc_mean(linear_product, 3, 31, replicates=6, seed=3)
        self.assertIsInstance(result, RQMCResult)
        self.assertEqual((result.estimator, result.method), ("mean", "lattice"))
        self.assertEqual((result.replicates, result.size, result.dimension), (6, 31, 3))
        self.assertEqual(result.evaluations, 6 * 31)
        self.assertEqual(result.seed, 3)
        self.assertEqual(len(result.replicate_estimates), 6)
        self.assertAlmostEqual(
            result.estimate, statistics.fmean(result.replicate_estimates)
        )
        deviation = statistics.stdev(result.replicate_estimates)
        assert result.standard_error is not None and result.interval is not None
        self.assertAlmostEqual(result.standard_error, deviation / math.sqrt(6))
        self.assertAlmostEqual(result.replicate_standard_deviation, deviation)
        # t quantile with 5 degrees of freedom at 0.975.
        half = 2.570581836 * result.standard_error
        self.assertAlmostEqual(result.interval[0], result.estimate - half, places=9)
        self.assertAlmostEqual(result.interval[1], result.estimate + half, places=9)
        self.assertEqual(result.confidence, 0.95)
        self.assertIn("coverage not guaranteed", result.interval_method)
        self.assertIn("standard error", result.summary())
        vector = cbc_generating_vector(31, 3).generating_vector
        self.assertEqual(result.generating_vectors, (vector,) * 6)
        self.assertLess(abs(result.estimate - 1.0), 5.0 * result.standard_error)

    def test_replicates_are_shifted_lattice_averages(self) -> None:
        """Replicate ``r`` is the average over the documented child stream."""
        z = (1, 12, 9)
        result = rqmc_mean(
            xexp_product, 3, 31, replicates=3, generating_vector=z, seed=8
        )
        for r, value in enumerate(result.replicate_estimates):
            design = rank1_lattice(
                31, z, shift=True, seed=RandomStream(8).child("replicate", r)
            )
            direct = math.fsum(xexp_product(p) for p in design.points) / 31
            self.assertEqual(value, direct)

    def test_constant_integrand(self) -> None:
        """A constant is integrated exactly with a zero-width interval."""
        for method in ("lattice", "sobol-owen", "sobol-shift", "halton"):
            size = 31 if method == "lattice" else 32
            result = rqmc_mean(
                lambda x: 2.5, 2, size, replicates=4, method=method, seed=1
            )
            self.assertEqual(result.estimate, 2.5)
            self.assertEqual(result.standard_error, 0.0)
            self.assertEqual(result.interval, (2.5, 2.5))
            self.assertEqual(result.method, method)

    def test_reproducibility_and_shared_replicates(self) -> None:
        """Equal seeds reproduce; mean and median see the same replicates."""
        first = rqmc_mean(xexp_product, 2, 31, replicates=5, seed=4)
        again = rqmc_mean(xexp_product, 2, 31, replicates=5, seed=RandomStream(4))
        self.assertEqual(first, again)
        median = rqmc_median(xexp_product, 2, 31, replicates=5, seed=4)
        self.assertEqual(median.replicate_estimates, first.replicate_estimates)
        self.assertEqual(median.estimate, sorted(first.replicate_estimates)[2])
        other = rqmc_mean(xexp_product, 2, 31, replicates=5, seed=5)
        self.assertNotEqual(first.estimate, other.estimate)

    def test_median_has_no_interval(self) -> None:
        """The median result carries no standard error and says why."""
        result = rqmc_median(linear_product, 2, 31, replicates=7, seed=2)
        self.assertEqual(result.estimator, "median")
        self.assertIsNone(result.standard_error)
        self.assertIsNone(result.interval)
        self.assertIsNone(result.confidence)
        self.assertIn("no interval", result.interval_method)
        self.assertIn("no interval", result.summary())
        self.assertEqual(result.evaluations, 7 * 31)
        self.assertIn(result.estimate, result.replicate_estimates)
        self.assertGreater(result.replicate_standard_deviation, 0.0)

    def test_known_integrals_with_every_method(self) -> None:
        """Mean and median recover integrals equal to one for every rule."""
        cases = (
            ("lattice", 127),
            ("sobol-owen", 128),
            ("sobol-shift", 128),
            ("halton", 128),
        )
        for method, size in cases:
            for f in (linear_product, quadratic_product, xexp_product):
                mean = rqmc_mean(f, 3, size, replicates=9, method=method, seed=11)
                median = rqmc_median(f, 3, size, replicates=9, method=method, seed=11)
                assert mean.standard_error is not None
                self.assertLess(abs(mean.estimate - 1.0), 5.0 * mean.standard_error)
                self.assertLess(abs(mean.estimate - 1.0), 0.03)
                self.assertLess(abs(median.estimate - 1.0), 0.03)
                self.assertEqual(mean.generating_vectors == (), method != "lattice")

    def test_mean_is_unbiased_for_shifted_lattices(self) -> None:
        """The average error over 300 seeds is within three standard errors."""
        for f, n in ((linear_product, 31), (xexp_product, 31), (periodic(1.0, 3), 61)):
            errors = [
                rqmc_mean(f, 3, n, replicates=2, seed=seed).estimate - 1.0
                for seed in range(300)
            ]
            se = statistics.stdev(errors) / math.sqrt(len(errors))
            self.assertGreater(se, 0.0)
            self.assertLess(abs(statistics.fmean(errors)), 3.0 * se)

    def test_error_decay_against_plain_monte_carlo(self) -> None:
        """Mean and median beat Monte Carlo at equal cost, increasingly so."""
        f, s, replicates, seeds = quadratic_product, 3, 11, range(20)
        ratio = {}
        for n in (31, 127, 509):
            mean = rmse(
                [
                    rqmc_mean(f, s, n, replicates=replicates, seed=k).estimate
                    for k in seeds
                ]
            )
            median = rmse(
                [
                    rqmc_median(f, s, n, replicates=replicates, seed=k).estimate
                    for k in seeds
                ]
            )
            plain = rmse(
                [
                    statistics.fmean(
                        f(p)
                        for p in monte_carlo_design(replicates * n, s, seed=k).points
                    )
                    for k in seeds
                ]
            )
            # A fixed good lattice has no outlying replicates, so the median
            # only pays its efficiency loss: within a factor 2.5 of the mean.
            self.assertLess(median, 2.5 * mean)
            self.assertLess(mean, 2.5 * median)
            ratio[n] = (plain / mean, plain / median, mean, median)
        self.assertGreater(ratio[31][0], 50.0)
        self.assertGreater(ratio[509][0], 1000.0)
        self.assertGreater(ratio[509][1], 1000.0)
        # From n = 31 to 509 Monte Carlo gains about 4; the lattice gains > 50.
        self.assertGreater(ratio[31][2] / ratio[509][2], 50.0)
        self.assertGreater(ratio[31][3] / ratio[509][3], 50.0)

    def test_t_interval_coverage_is_measured_not_assumed(self) -> None:
        """Coverage is near nominal in one case and below it in another."""

        def coverage(f: Callable[[Point], float], s: int, replicates: int) -> float:
            """Return the fraction of 1000 seeded intervals containing 1."""
            hits = 0
            for seed in range(1000, 2000):
                result = rqmc_mean(f, s, 31, replicates=replicates, seed=seed)
                assert result.interval is not None
                hits += result.interval[0] <= 1.0 <= result.interval[1]
            return hits / 1000

        # Smooth product in two dimensions, R = 10: close to 95 %.
        self.assertTrue(0.93 <= coverage(xexp_product, 2, 10) <= 0.97)
        # One dimension, linear integrand: each replicate error is exactly
        # uniform on +-c / (2 n), so the coverage is that of a t interval for
        # five uniform variates, 0.9348 (direct simulation, 400 000 samples).
        # Three binomial standard errors over 1000 seeds are 0.0234.
        low = coverage(linear_product, 1, 5)
        self.assertAlmostEqual(low, 0.9348, delta=0.0234)

    def test_one_dimensional_linear_replicate_law(self) -> None:
        """For ``f = 1 + c (x - 1/2)`` the replicate error is ``c (u - 1/2)/n``."""
        n = 31
        result = rqmc_mean(linear_product, 1, n, replicates=400, seed=9)
        scaled = [(q - 1.0) * n / 0.5 for q in result.replicate_estimates]
        self.assertLessEqual(max(abs(v) for v in scaled), 0.5 + 1e-9)
        self.assertAlmostEqual(statistics.pvariance(scaled), 1.0 / 12.0, delta=0.01)

    def test_validation(self) -> None:
        """Bad replicate counts, methods, and integrand values are rejected."""
        with self.assertRaisesRegex(ValueError, "at least 2"):
            rqmc_mean(linear_product, 2, 31, replicates=1)
        with self.assertRaisesRegex(ValueError, "odd"):
            rqmc_median(linear_product, 2, 31, replicates=10)
        with self.assertRaisesRegex(ValueError, "odd"):
            rqmc_median(linear_product, 2, 31, replicates=1)
        with self.assertRaisesRegex(ValueError, "method"):
            rqmc_mean(linear_product, 2, 31, method="lhs")
        with self.assertRaisesRegex(ValueError, "prime"):
            rqmc_mean(linear_product, 2, 32, seed=1)
        with self.assertRaisesRegex(ValueError, "components"):
            rqmc_mean(linear_product, 2, 32, generating_vector=(1, 3, 5), seed=1)
        with self.assertRaisesRegex(ValueError, "only to method='lattice'"):
            rqmc_mean(linear_product, 2, 32, method="halton", weights=(1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "confidence"):
            rqmc_mean(linear_product, 2, 31, confidence=1.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rqmc_mean(lambda x: math.nan, 2, 31, seed=1)
        with self.assertRaises(TypeError):
            rqmc_mean(3.0, 2, 31)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            rqmc_median(linear_product, 0, 31)
        # A supplied vector lifts the prime requirement.
        result = rqmc_mean(linear_product, 2, 32, generating_vector=(1, 13), seed=1)
        self.assertEqual(result.generating_vectors[0], (1, 13))


class MedianRandomLatticeTests(unittest.TestCase):
    """The construction-free median of randomly generated lattice rules."""

    def test_result_and_reproducibility(self) -> None:
        """Vectors are drawn from ``{1..n-1}^s``; the median is a replicate."""
        result = median_random_lattice(xexp_product, 3, 127, replicates=9, seed=21)
        self.assertEqual(
            (result.estimator, result.method), ("median", "random-lattice")
        )
        self.assertEqual(result.evaluations, 9 * 127)
        self.assertEqual(len(result.generating_vectors), 9)
        self.assertGreater(len(set(result.generating_vectors)), 1)
        for vector in result.generating_vectors:
            self.assertEqual(len(vector), 3)
            self.assertTrue(all(1 <= c <= 126 for c in vector))
        self.assertEqual(result.estimate, statistics.median(result.replicate_estimates))
        self.assertIsNone(result.interval)
        self.assertIsNone(result.standard_error)
        self.assertEqual(
            result, median_random_lattice(xexp_product, 3, 127, replicates=9, seed=21)
        )
        self.assertLess(abs(result.estimate - 1.0), 0.1)

    def test_replicates_match_the_recorded_vectors(self) -> None:
        """Unshifted replicates are the plain lattice averages of their vectors."""
        f = periodic(1.0, 2)
        result = median_random_lattice(f, 2, 61, replicates=5, shift=False, seed=4)
        for vector, value in zip(
            result.generating_vectors, result.replicate_estimates, strict=True
        ):
            points = rank1_lattice(61, vector).points
            self.assertEqual(value, math.fsum(f(p) for p in points) / 61)

    def test_generating_vectors_are_uniform(self) -> None:
        """Pooled components have the mean and range of ``U{1..n-1}``."""
        n = 11
        components = [
            c
            for seed in range(60)
            for vector in median_random_lattice(
                lambda x: 1.0, 2, n, replicates=5, seed=seed
            ).generating_vectors
            for c in vector
        ]
        self.assertEqual(set(components), set(range(1, n)))
        se = math.sqrt((10**2 - 1) / 12 / len(components))
        self.assertLess(abs(statistics.fmean(components) - 6.0), 4.0 * se)

    def test_median_is_far_better_than_the_mean_of_random_lattices(self) -> None:
        """For a smooth periodic integrand the median wins by a large factor."""
        f = periodic(1.0, 3)
        medians, means, bad = [], [], 0
        for seed in range(30):
            result = median_random_lattice(f, 3, 509, replicates=11, seed=seed)
            medians.append(result.estimate)
            means.append(statistics.fmean(result.replicate_estimates))
            bad += sum(abs(q - 1.0) > 1e-3 for q in result.replicate_estimates)
        self.assertLess(rmse(medians), 1e-4)
        self.assertGreater(rmse(means), 100.0 * rmse(medians))
        # A minority of the random vectors is bad; they dominate the mean.
        self.assertTrue(0 < bad < 0.25 * 30 * 11, bad)

    def test_median_keeps_most_of_the_accuracy_of_a_constructed_rule(self) -> None:
        """Random vectors lose to CBC but still beat Monte Carlo clearly."""
        f, seeds = quadratic_product, range(20)
        random_median = rmse(
            [
                median_random_lattice(f, 3, 127, replicates=11, seed=k).estimate
                for k in seeds
            ]
        )
        constructed = rmse(
            [rqmc_median(f, 3, 127, replicates=11, seed=k).estimate for k in seeds]
        )
        plain = rmse(
            [
                statistics.fmean(
                    f(p) for p in monte_carlo_design(11 * 127, 3, seed=k).points
                )
                for k in seeds
            ]
        )
        self.assertLess(constructed, random_median)
        self.assertLess(random_median, plain / 20.0)

    def test_validation(self) -> None:
        """Composite sizes and even replicate counts are rejected."""
        with self.assertRaisesRegex(ValueError, "prime"):
            median_random_lattice(linear_product, 2, 32)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            median_random_lattice(linear_product, 2, 2)
        with self.assertRaisesRegex(ValueError, "odd"):
            median_random_lattice(linear_product, 2, 31, replicates=4)
        with self.assertRaises(TypeError):
            median_random_lattice(None, 2, 31)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
