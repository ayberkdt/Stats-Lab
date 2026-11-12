"""Tests for robust M-estimation of location and regression and Theil-Sen.

Oracles:

* closed forms on symmetric data (the M-estimate of a sample symmetric about
  ``c`` is ``c``);
* roots of the estimating equations computed offline with SciPy 1.14.1 and
  NumPy 2.2.6 (``scipy.optimize.brentq`` for location,
  ``scipy.optimize.fsolve`` on ``X' psi(r/s) = 0`` with
  ``s = 1.4826 median|r|`` for Huber regression, an independent NumPy IRLS
  from least squares for the biweight, and the sandwich covariance in NumPy);
* ``scipy.stats.theilslopes(y, x, confidence, method="joint")`` for the
  Theil-Sen slope and Sen's interval;
* seeded simulations judged by the binomial standard error of a coverage;
* breakdown demonstrations with planted outliers.

The stack-loss data are Brownlee's plant data as distributed with R
[REF-R-DATASETS-STACKLOSS].
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.inference import (
    BISQUARE_TUNING,
    HUBER_TUNING,
    fit_robust_linear_model,
    fit_simple_linear_regression,
    psi_function,
    robust_location,
    theil_sen_slope,
)

AIR = (
    80,
    80,
    75,
    62,
    62,
    62,
    62,
    62,
    58,
    58,
    58,
    58,
    58,
    58,
    50,
    50,
    50,
    50,
    50,
    56,
    70,
)
WATER = (
    27,
    27,
    25,
    24,
    22,
    23,
    24,
    24,
    23,
    18,
    18,
    17,
    18,
    19,
    18,
    18,
    19,
    19,
    20,
    20,
    20,
)
ACID = (
    89,
    88,
    90,
    87,
    87,
    87,
    93,
    93,
    87,
    80,
    89,
    88,
    82,
    93,
    89,
    86,
    72,
    79,
    80,
    82,
    91,
)
LOSS = (42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15)
STACK_DESIGN = tuple((1.0, a, w, c) for a, w, c in zip(AIR, WATER, ACID, strict=True))
NAMES = ("(Intercept)", "air", "water", "acid")
MAD = 1.482602218505602


def mad_about_zero(values: list[float]) -> float:
    """Return ``1.4826 median|v|`` computed independently of the library."""
    return MAD * statistics.median(abs(v) for v in values)


class PsiFunctionTests(unittest.TestCase):
    """The score functions match their closed forms."""

    def test_huber_and_biweight_scores(self) -> None:
        """Huber clips at ``k``; the biweight redescends to zero at ``c``."""
        self.assertEqual(psi_function(0.5), 0.5)
        self.assertEqual(psi_function(3.0), HUBER_TUNING)
        self.assertEqual(psi_function(-3.0), -HUBER_TUNING)
        u = 2.0
        expected = u * (1.0 - (u / BISQUARE_TUNING) ** 2) ** 2
        self.assertAlmostEqual(psi_function(u, psi="bisquare"), expected, places=15)
        self.assertEqual(psi_function(5.0, psi="bisquare"), 0.0)
        self.assertEqual(psi_function(1.0, psi="huber", tuning=0.5), 0.5)

    def test_invalid_options(self) -> None:
        """Unknown families and non-positive constants are rejected."""
        with self.assertRaises(ValueError):
            psi_function(1.0, psi="cauchy")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            psi_function(1.0, tuning=0.0)


class RobustLocationTests(unittest.TestCase):
    """Huber and biweight location estimates with MAD scale."""

    def test_symmetric_sample_gives_its_centre(self) -> None:
        """Closed form: a sample symmetric about ``c`` has M-estimate ``c``."""
        centre = 10.25
        offsets = (-7.0, -2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 7.0)
        sample = [centre + o for o in offsets]
        for psi in ("huber", "bisquare"):
            with self.subTest(psi=psi):
                result = robust_location(sample, psi=psi)
                self.assertAlmostEqual(result.estimate, centre, places=12)
                self.assertTrue(result.converged)

    def test_matches_brent_roots_of_the_estimating_equation(self) -> None:
        """Roots from ``scipy.optimize.brentq`` (SciPy 1.14.1, xtol 1e-15).

        Calls: ``brentq(lambda m: sum(clip((x - m)/s, -1.345, 1.345)), 1.9,
        2.6)`` and the biweight analogue with ``c = 4.685``, where
        ``s = median_abs_deviation(x, scale="normal") = 0.18532527731320025``.
        """
        sample = (2.1, 2.4, 2.2, 2.3, 2.0, 9.5, 2.25, 2.15, 1.95, 2.35)
        huber = robust_location(sample)
        biweight = robust_location(sample, psi="bisquare")
        self.assertAlmostEqual(huber.scale, 0.18532527731320025, places=15)
        self.assertAlmostEqual(huber.estimate, 2.21875, places=12)
        self.assertAlmostEqual(biweight.estimate, 2.1908619980147486, places=12)
        self.assertEqual(biweight.weights[5], 0.0)
        self.assertLess(huber.weights[5], 0.05)

    def test_sandwich_standard_error_formula(self) -> None:
        """``se = s sqrt(n/(n-1) sum psi^2) / sum psi'`` recomputed by hand."""
        sample = (2.1, 2.4, 2.2, 2.3, 2.0, 9.5, 2.25, 2.15, 1.95, 2.35)
        result = robust_location(sample)
        s, mu, k = result.scale, result.estimate, HUBER_TUNING
        u = [(x - mu) / s for x in sample]
        psi = [max(-k, min(k, v)) for v in u]
        slope = sum(1.0 for v in u if abs(v) <= k)
        expected = s * math.sqrt(10 / 9 * sum(p * p for p in psi)) / slope
        self.assertAlmostEqual(result.standard_error, expected, places=14)
        self.assertLess(result.confidence_interval.lower, mu)
        self.assertGreater(result.confidence_interval.upper, mu)

    def test_breakdown_against_gross_outliers(self) -> None:
        """Four of ten values at ``1e6`` move the mean, not the M-estimates."""
        good = [4.9, 5.1, 5.0, 5.2, 4.8, 5.05]
        contaminated = good + [1e6] * 4
        self.assertGreater(statistics.fmean(contaminated), 1e5)
        for psi in ("huber", "bisquare"):
            with self.subTest(psi=psi):
                estimate = robust_location(contaminated, psi=psi).estimate
                self.assertLess(estimate, 10.0)

    def test_interval_coverage_at_the_normal_law(self) -> None:
        """Seeded simulation: 95% Huber intervals cover within 3 binomial SE.

        400 samples of size 40 from ``N(3, 2^2)``; the binomial standard
        error of the coverage is ``sqrt(0.95 0.05 / 400) = 0.0109``.
        """
        generator = random.Random(101)
        replications = 400
        covered = 0
        for _ in range(replications):
            sample = [generator.gauss(3.0, 2.0) for _ in range(40)]
            interval = robust_location(sample).confidence_interval
            covered += interval.lower <= 3.0 <= interval.upper
        coverage = covered / replications
        self.assertLess(
            abs(coverage - 0.95), 3.0 * math.sqrt(0.95 * 0.05 / replications)
        )

    def test_errors(self) -> None:
        """A zero MAD and a single observation are rejected."""
        with self.assertRaises(ValueError):
            robust_location([1.0, 1.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            robust_location([1.0])
        self.assertEqual(
            robust_location([1.0, 1.0, 1.0, 2.0], scale=1.0).sample_size, 4
        )


class RobustRegressionTests(unittest.TestCase):
    """Huber and biweight regression M-estimates on the stack-loss data."""

    def test_huber_matches_the_root_of_the_estimating_equations(self) -> None:
        """Root from ``scipy.optimize.fsolve`` (SciPy 1.14.1, xtol 1e-14).

        Call: ``fsolve(lambda b: X.T @ clip((y - X b)/s(b), -1.345, 1.345),
        ols)`` with ``s(b) = 1.4826 median|y - X b|``; the sandwich standard
        errors and the scale were recomputed in NumPy 2.2.6 at that root.
        """
        result = fit_robust_linear_model(STACK_DESIGN, LOSS, column_names=NAMES)
        expected = (
            -41.02649835240027,
            0.8293843346001096,
            0.9260659661966449,
            -0.12784672494578506,
        )
        for got, want in zip(result.coefficients, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-7 * max(1.0, abs(want)))
        self.assertAlmostEqual(result.scale, 2.44053609172099, delta=1e-8)
        errors = (
            5.672284628247426,
            0.15599134055786593,
            0.37840073794034335,
            0.07293564049202347,
        )
        for got, want in zip(result.standard_errors, errors, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-6 * want)
        self.assertEqual(result.start, "ls")
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.coefficient("air"), expected[1], delta=1e-7)

    def test_estimating_equations_hold_at_the_solution(self) -> None:
        """``sum psi(r_i/s) x_i = 0`` with ``s`` the MAD of its residuals."""
        for psi, c in (("huber", HUBER_TUNING), ("bisquare", BISQUARE_TUNING)):
            with self.subTest(psi=psi):
                result = fit_robust_linear_model(STACK_DESIGN, LOSS, psi=psi)
                residuals = [
                    y
                    - sum(b * x for b, x in zip(result.coefficients, row, strict=True))
                    for row, y in zip(STACK_DESIGN, LOSS, strict=True)
                ]
                s = mad_about_zero(residuals)
                self.assertAlmostEqual(s, result.scale, places=8)
                for j in range(4):
                    total = sum(
                        psi_function(r / s, psi=psi, tuning=c) * row[j]
                        for r, row in zip(residuals, STACK_DESIGN, strict=True)
                    )
                    self.assertLess(abs(total), 1e-6)

    def test_biweight_matches_an_independent_irls(self) -> None:
        """Values from a NumPy 2.2.6 IRLS (``lstsq`` weighted fits, LS start).

        The biweight root reached from least absolute deviations equals the
        one reached from least squares on these data.
        """
        expected = (
            -42.285350779329704,
            0.9275573227555248,
            0.6507176872142929,
            -0.11233315379090014,
        )
        for start in ("lad", "ls"):
            with self.subTest(start=start):
                result = fit_robust_linear_model(
                    STACK_DESIGN, LOSS, psi="bisquare", start=start
                )
                for got, want in zip(result.coefficients, expected, strict=True):
                    self.assertAlmostEqual(got, want, delta=1e-7 * max(1.0, abs(want)))
        result = fit_robust_linear_model(STACK_DESIGN, LOSS, psi="bisquare")
        self.assertEqual(result.start, "lad")
        self.assertLess(result.weights[20], 0.01)
        errors = (5.019191044083708, 0.12783946697673254, 0.2696370540217672)
        for got, want in zip(result.standard_errors, errors, strict=False):
            self.assertAlmostEqual(got, want, delta=1e-6 * want)

    def test_vertical_outliers_do_not_move_the_fit(self) -> None:
        """Twenty percent of responses shifted by 1000 leave the slope near 2."""
        x = [float(i) for i in range(30)]
        y = [1.0 + 2.0 * v + (0.3 if i % 2 else -0.3) for i, v in enumerate(x)]
        for i in range(0, 30, 5):
            y[i] += 1000.0
        design = [(1.0, v) for v in x]
        ols = fit_simple_linear_regression(x, y)
        self.assertGreater(abs(ols.slope - 2.0), 1.0)
        for psi in ("huber", "bisquare"):
            with self.subTest(psi=psi):
                fit = fit_robust_linear_model(design, y, psi=psi)
                self.assertLess(abs(fit.coefficients[1] - 2.0), 0.1)

    def test_one_bad_leverage_point_breaks_m_regression_but_not_theil_sen(self) -> None:
        """Breakdown ``1/n`` for leverage versus about 29% for Theil-Sen."""
        x = [float(i) for i in range(20)] + [500.0]
        y = [2.0 * v + (0.2 if i % 3 else -0.4) for i, v in enumerate(x[:-1])] + [0.0]
        huber = fit_robust_linear_model([(1.0, v) for v in x], y)
        self.assertLess(huber.coefficients[1], 0.5)
        self.assertLess(abs(theil_sen_slope(x, y).slope - 2.0), 0.05)

    def test_slope_interval_coverage_with_heavy_tails(self) -> None:
        """Seeded simulation with Student-t(3) errors, 300 replications.

        The 95% sandwich intervals of the Huber slope must cover the true
        slope within three binomial standard errors (``0.0126`` each).
        """
        generator = random.Random(2026)
        x = [i / 10.0 for i in range(50)]
        design = [(1.0, v) for v in x]
        replications = 300
        covered = 0
        for _ in range(replications):
            noise = [
                generator.gauss(0.0, 1.0)
                / math.sqrt(sum(generator.gauss(0.0, 1.0) ** 2 for _ in range(3)) / 3.0)
                for _ in x
            ]
            y = [1.0 + 0.5 * v + e for v, e in zip(x, noise, strict=True)]
            interval = fit_robust_linear_model(design, y).confidence_intervals[1]
            covered += interval.lower <= 0.5 <= interval.upper
        coverage = covered / replications
        self.assertLess(
            abs(coverage - 0.95), 3.0 * math.sqrt(0.95 * 0.05 / replications)
        )

    def test_errors(self) -> None:
        """Shape, start, and exact-fit errors are reported."""
        with self.assertRaises(ValueError):
            fit_robust_linear_model([(1.0, 1.0), (1.0, 2.0)], [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_robust_linear_model(STACK_DESIGN, LOSS, start="median")  # type: ignore[arg-type]
        exact = [(1.0, float(i)) for i in range(10)]
        with self.assertRaises(ValueError):
            fit_robust_linear_model(exact, [2.0 * i for i in range(10)])


class TheilSenTests(unittest.TestCase):
    """Theil-Sen slope and Sen's interval."""

    X = (1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10)
    Y = (1.2, 1.9, 2.6, 3.1, 3.9, 5.5, 4.8, 5.1, 6.3, 6.8, 8.4, 8.7, 30.0)

    def test_matches_scipy_theilslopes(self) -> None:
        """``scipy.stats.theilslopes(Y, X, conf, method="joint")``, SciPy 1.14.1.

        ``x`` has ties (corrected) and ``y`` has none.
        """
        result = theil_sen_slope(self.X, self.Y)
        self.assertAlmostEqual(result.slope, 0.99, places=14)
        self.assertAlmostEqual(result.intercept, 0.14999999999999947, places=14)
        self.assertAlmostEqual(
            result.confidence_interval.lower, 0.9249999999999999, places=14
        )
        self.assertAlmostEqual(result.confidence_interval.upper, 1.2, places=14)
        ninety = theil_sen_slope(self.X, self.Y, confidence=0.9).confidence_interval
        self.assertAlmostEqual(ninety.lower, 0.9333333333333332, places=14)
        self.assertAlmostEqual(ninety.upper, 1.125, places=14)
        self.assertEqual(result.pair_count, 78 - 4)

    def test_exact_line_and_brute_force_median(self) -> None:
        """An exact line returns its slope; otherwise the median of all slopes."""
        line = theil_sen_slope([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])
        self.assertEqual(line.slope, 2.0)
        self.assertEqual(line.intercept, 1.0)
        slopes = sorted(
            (self.Y[j] - self.Y[i]) / (self.X[j] - self.X[i])
            for i in range(13)
            for j in range(i + 1, 13)
            if self.X[i] != self.X[j]
        )
        self.assertEqual(
            theil_sen_slope(self.X, self.Y).slope, statistics.median(slopes)
        )

    def test_quarter_contamination_with_leverage(self) -> None:
        """Five of twenty points at ``(100, -1000)``: below the 29% breakdown."""
        x = [float(i) for i in range(15)] + [100.0] * 5
        y = [3.0 * v + 1.0 + (0.1 if i % 2 else -0.1) for i, v in enumerate(x[:15])]
        y += [-1000.0] * 5
        self.assertLess(abs(theil_sen_slope(x, y).slope - 3.0), 0.1)

    def test_errors(self) -> None:
        """Equal ``x`` and unequal lengths are rejected."""
        with self.assertRaises(ValueError):
            theil_sen_slope([1, 1, 1], [1, 2, 3])
        with self.assertRaises(ValueError):
            theil_sen_slope([1, 2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
