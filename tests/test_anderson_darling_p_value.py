"""Anderson-Darling null distribution and p-values for a fully specified CDF.

Oracles:

- Marsaglia and Marsaglia (2004), Section 2: the 30-digit values
  ``ADinf(9) = 0.999960465988612484992562014458`` and
  ``ADinf(10) = 0.999986184964589314168018038088`` and the 20-digit
  percentage points ``1.9329578327415937304`` (90 %),
  ``2.4923671600494096176`` (95 %), ``3.8781250216053948842`` (99 %); the
  classical Anderson-Darling (1954) points ``2.492`` and ``3.857``, the
  latter shown there to be too small.
- An independent 30-digit quadrature of the Anderson-Darling integral
  (Section 2, first display) with mpmath 1.3.0 under Python 3.12.1:
  ``sqrt(2 pi)/z sum_j binomial(-1/2, j)(4j+1) e^(-t_j) quad(lambda w:
  exp(z/(8(1+w^2)) - w^2 t_j), [0, 1, inf])``, ``mp.dps = 30``.
- ``E[A_n^2] = 1`` for every ``n``: from ``A_n^2 = n int (F_n - t)^2 /
  (t (1 - t)) dt`` (Section 1) and ``E[(F_n(t) - t)^2] = t (1 - t) / n``,
  so the limit law has mean ``int_0^inf (1 - ADinf(z)) dz = 1``.
- The exact ``n = 1`` law ``sqrt(1 - 4 exp(-1 - z))`` (Section 1, derived
  in [EQ-INF-GOF-AD-004]) and seeded simulation of ``A_1^2``.
- Seeded simulations of 20 000 uniform samples at ``n = 5, 10, 50``,
  judged by binomial standard errors plus the stated ``5e-4`` accuracy.
- Section 3: the largest error of ``ADinf`` as a finite-n law is about
  ``0.044 / n``, near the 33rd percentile.

References:
    [REF-MARSAGLIA-MARSAGLIA-2004], [REF-ANDERSON-DARLING-1954],
    [EQ-INF-GOF-AD-002], [EQ-INF-GOF-AD-003], [EQ-INF-GOF-AD-004].

"""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal, Uniform
from probstat.inference import (
    AndersonDarlingResult,
    anderson_darling_asymptotic_cdf,
    anderson_darling_cdf,
    anderson_darling_test,
    fit_normal_mle,
)

REPLICATIONS = 20_000
STATED_ACCURACY = 5e-4


def _uniform_statistics(size: int, seed: int) -> list[float]:
    """Simulate ``A_n^2`` for uniform samples (fully specified null).

    References:
        [EQ-INF-GOF-AD-001].

    """
    rng = random.Random(seed)
    weights = [2 * rank - 1 for rank in range(1, size + 1)]
    values: list[float] = []
    for _ in range(REPLICATIONS):
        sample = sorted(rng.random() for _ in range(size))
        total = math.fsum(
            weight * (math.log(low) + math.log1p(-high))
            for weight, low, high in zip(weights, sample, reversed(sample), strict=True)
        )
        values.append(-size - total / size)
    return values


def _simpson(function: object, lower: float, upper: float, panels: int) -> float:
    """Integrate a smooth function by the composite Simpson rule.

    References:
        [REF-MARSAGLIA-MARSAGLIA-2004, Section 1].

    """
    assert callable(function)
    width = (upper - lower) / panels
    total = function(lower) + function(upper)
    for index in range(1, panels):
        total += (4 if index % 2 else 2) * function(lower + index * width)
    return float(total * width / 3)


class AsymptoticLawTests(unittest.TestCase):
    """``ADinf`` against the published and quadrature values."""

    def test_published_thirty_digit_values(self) -> None:
        """ADinf(9) and ADinf(10) match the paper's Maple values."""
        self.assertAlmostEqual(
            anderson_darling_asymptotic_cdf(9.0),
            0.999960465988612484992562014458,
            delta=1e-14,
        )
        self.assertAlmostEqual(
            anderson_darling_asymptotic_cdf(10.0),
            0.999986184964589314168018038088,
            delta=1e-14,
        )

    def test_percentage_points(self) -> None:
        """The 20-digit 90/95/99 % points map to 0.90, 0.95, 0.99."""
        points = (
            (1.9329578327415937304, 0.90),
            (2.4923671600494096176, 0.95),
            (3.8781250216053948842, 0.99),
        )
        for point, level in points:
            self.assertAlmostEqual(
                anderson_darling_asymptotic_cdf(point), level, delta=1e-13
            )

    def test_classical_points_of_anderson_and_darling(self) -> None:
        """2.492 is the 95 % point to 3 digits; 3.857 is not the 99 % point."""
        self.assertAlmostEqual(
            anderson_darling_asymptotic_cdf(2.492), 0.95, delta=5e-5
        )
        self.assertAlmostEqual(
            anderson_darling_asymptotic_cdf(3.857), 0.98976, delta=1e-5
        )

    def test_independent_quadrature(self) -> None:
        """The recursion agrees with a 30-digit quadrature to 1e-15."""
        reference = {
            0.25: 0.029606672521625274977,
            0.5: 0.25318562646965551557,
            1.0: 0.64273332678598091399,
            2.0: 0.90816322505874788327,
            3.0: 0.9726352116659736407,
            4.0: 0.9912818130860867131,
            6.0: 0.99903254805847907708,
        }
        for point, value in reference.items():
            self.assertAlmostEqual(
                anderson_darling_asymptotic_cdf(point), value, delta=1e-15
            )

    def test_mean_of_limit_law_is_one(self) -> None:
        """int_0^40 (1 - ADinf) = E[A^2] = 1 (the tail beyond 40 is < 1e-18)."""
        mean = _simpson(
            lambda z: 1.0 - anderson_darling_asymptotic_cdf(z), 0.0, 40.0, 4000
        )
        self.assertAlmostEqual(mean, 1.0, delta=1e-9)

    def test_boundaries_and_far_tail(self) -> None:
        """Zero near the origin, one at and beyond z = 40, monotone between."""
        self.assertEqual(anderson_darling_asymptotic_cdf(0.0), 0.0)
        self.assertEqual(anderson_darling_asymptotic_cdf(-3.0), 0.0)
        self.assertEqual(anderson_darling_asymptotic_cdf(math.inf), 1.0)
        for point in (40.0, 1_000.0, 1e8):
            self.assertEqual(anderson_darling_asymptotic_cdf(point), 1.0)
        grid = [0.02 * k for k in range(1, 2000)]
        values = [anderson_darling_asymptotic_cdf(z) for z in grid]
        for left, right in zip(values[:-1], values[1:], strict=True):
            self.assertLessEqual(left, right + 1e-13)

    def test_validation(self) -> None:
        """NaN and non-real arguments are rejected."""
        with self.assertRaises(ValueError):
            anderson_darling_asymptotic_cdf(math.nan)
        with self.assertRaises(TypeError):
            anderson_darling_asymptotic_cdf(True)
        with self.assertRaises(TypeError):
            anderson_darling_asymptotic_cdf("1.0")  # type: ignore[arg-type]


class FiniteSampleLawTests(unittest.TestCase):
    """``P(A_n^2 < z)`` against exact and simulated laws."""

    def test_single_observation_is_exact(self) -> None:
        """The n = 1 law equals sqrt(1 - 4 exp(-1 - z)) and the simulated law."""
        threshold = math.log(4.0) - 1.0
        self.assertEqual(anderson_darling_cdf(threshold, 1), 0.0)
        for point in (0.4, 0.5, 1.0, 2.492, 5.0):
            self.assertAlmostEqual(
                anderson_darling_cdf(point, 1),
                math.sqrt(1.0 - 4.0 * math.exp(-1.0 - point)),
                delta=1e-15,
            )
        rng = random.Random(101)
        draws = [rng.random() for _ in range(REPLICATIONS)]
        statistics = [-1.0 - math.log(u * (1.0 - u)) for u in draws]
        law = Uniform(0.0, 1.0)
        for u, statistic in zip(draws[:5], statistics[:5], strict=True):
            self.assertAlmostEqual(
                anderson_darling_test([u], law).statistic, statistic, delta=1e-12
            )
        for point in (0.45, 0.7, 1.0, 2.0, 4.0):
            empirical = sum(s < point for s in statistics) / REPLICATIONS
            exact = anderson_darling_cdf(point, 1)
            error = math.sqrt(exact * (1.0 - exact) / REPLICATIONS)
            self.assertLess(abs(empirical - exact), 4.0 * error + 1e-12)

    def test_simulated_laws(self) -> None:
        """The n = 5, 10, 50 laws agree with 20 000 simulated statistics."""
        points = (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.9)
        for size, seed in ((5, 5), (10, 10), (50, 50)):
            statistics = _uniform_statistics(size, seed)
            for point in points:
                empirical = sum(s < point for s in statistics) / REPLICATIONS
                model = anderson_darling_cdf(point, size)
                error = math.sqrt(model * (1.0 - model) / REPLICATIONS)
                with self.subTest(size=size, point=point):
                    self.assertLess(
                        abs(empirical - model), 4.0 * error + STATED_ACCURACY
                    )

    def test_largest_correction_is_about_0_044_over_n(self) -> None:
        """The largest |P_n - ADinf| is about 0.044/n, near ADinf = 1/3."""
        grid = [0.005 * k for k in range(20, 1000)]
        for size in (8, 16, 32, 64, 128):
            gaps = [
                (
                    abs(
                        anderson_darling_cdf(z, size)
                        - anderson_darling_asymptotic_cdf(z)
                    ),
                    z,
                )
                for z in grid
            ]
            gap, where = max(gaps)
            self.assertAlmostEqual(gap * size, 0.044, delta=0.0035)
            self.assertAlmostEqual(
                anderson_darling_asymptotic_cdf(where), 0.33, delta=0.03
            )

    def test_monotone_and_bounded(self) -> None:
        """Values lie in [0, 1] and decrease by at most 1e-4 / n anywhere."""
        grid = [0.01 * k for k in range(1, 3000)]
        for size in (2, 3, 5, 10, 100, 10_000):
            values = [anderson_darling_cdf(z, size) for z in grid]
            self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
            for left, right in zip(values[:-1], values[1:], strict=True):
                self.assertLessEqual(left, right + 1e-4 / size)
            self.assertEqual(anderson_darling_cdf(math.inf, size), 1.0)
            self.assertEqual(anderson_darling_cdf(-math.inf, size), 0.0)
            self.assertEqual(anderson_darling_cdf(0.0, size), 0.0)

    def test_upper_tail_floor_is_conservative(self) -> None:
        """Huge finite statistics keep a p-value of about 6e-4 / n."""
        for size in (2, 10, 1000):
            tail = 1.0 - anderson_darling_cdf(1e6, size)
            self.assertAlmostEqual(tail * size, 6e-4, delta=1e-6)

    def test_validation(self) -> None:
        """Invalid sample sizes and statistics are rejected."""
        with self.assertRaises(ValueError):
            anderson_darling_cdf(1.0, 0)
        with self.assertRaises(TypeError):
            anderson_darling_cdf(1.0, 2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            anderson_darling_cdf(math.nan, 5)


class TestPValueTests(unittest.TestCase):
    """``anderson_darling_test`` reports the case-0 p-value."""

    def test_p_value_is_upper_tail_of_finite_law(self) -> None:
        """The p-value is 1 - P(A_n^2 < a), labelled Marsaglia for n >= 2."""
        sample = [
            0.0392, 0.0884, 0.260, 0.310, 0.454,
            0.644, 0.797, 0.813, 0.921, 0.960,
        ]  # fmt: skip
        result = anderson_darling_test(sample, Uniform(0.0, 1.0))
        self.assertEqual(
            result.p_value, 1.0 - anderson_darling_cdf(result.statistic, 10)
        )
        self.assertIn("Marsaglia", result.method or "")
        self.assertGreater(result.p_value or 0.0, 0.5)
        clustered = [
            0.0015, 0.0078, 0.0676, 0.0961, 0.106,
            0.107, 0.835, 0.861, 0.948, 0.992,
        ]  # fmt: skip
        self.assertLess(
            anderson_darling_test(clustered, Uniform(0.0, 1.0)).p_value or 1.0, 0.01
        )
        single = anderson_darling_test([0.3], Uniform(0.0, 1.0))
        self.assertIn("exact", single.method or "")

    def test_boundary_observation_has_zero_p_value(self) -> None:
        """An observation with null tail probability zero gives p = 0."""
        result = anderson_darling_test((0.0, 0.5, 0.7), Uniform(0.0, 1.0))
        self.assertEqual(result.statistic, math.inf)
        self.assertEqual(result.p_value, 0.0)

    def test_result_stays_backward_compatible(self) -> None:
        """Two-field construction still works; new fields are validated."""
        legacy = AndersonDarlingResult(0.5, 10)
        self.assertIsNone(legacy.p_value)
        self.assertIsNone(legacy.method)
        with self.assertRaises(ValueError):
            AndersonDarlingResult(0.5, 10, 1.5)
        with self.assertRaises(ValueError):
            AndersonDarlingResult(0.5, 10, 0.5, " ")

    def test_case_zero_p_values_are_too_large_after_fitting(self) -> None:
        """Against a fitted normal, case-0 p-values almost never fall below 0.1."""
        rng = random.Random(30)
        p_values = []
        for _ in range(300):
            sample = [rng.gauss(0.0, 1.0) for _ in range(30)]
            law = fit_normal_mle(sample)
            self.assertIsInstance(law, Normal)
            p_values.append(anderson_darling_test(sample, law).p_value or 0.0)
        below = sum(p < 0.10 for p in p_values) / len(p_values)
        # A calibrated test gives 0.10 (binomial standard error 0.017).
        self.assertLess(below, 0.02)
        self.assertGreater(sum(p_values) / len(p_values), 0.7)


if __name__ == "__main__":
    unittest.main()
