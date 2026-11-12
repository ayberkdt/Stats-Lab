"""Tests for exact linear quantile regression.

Oracles:

* exhaustive enumeration of every elemental basis (``C(n, p)`` exact
  interpolations of ``p`` observations), which contains an optimal vertex of
  the check-loss linear program (Koenker and Bassett 1978);
* optimal objective values of the same linear program solved offline with
  ``scipy.optimize.linprog(method="highs")`` (SciPy 1.14.1) on the
  stack-loss data [REF-R-DATASETS-STACKLOSS];
* sample quantiles for the intercept-only model, regression and scale
  equivariance, and the residual-sign counts of the subgradient condition;
* a seeded simulation of the bootstrap interval coverage judged by the
  binomial standard error.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.inference import (
    RankDeficientDesignError,
    check_loss,
    fit_quantile_regression,
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


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve a small square system by Gaussian elimination (``None`` if singular)."""
    size = len(rhs)
    a = [row[:] + [b] for row, b in zip(matrix, rhs, strict=True)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-9:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(size):
            if r != col:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col], strict=True)]
    return [a[i][size] / a[i][i] for i in range(size)]


def objective(design, response, beta, tau: float) -> float:
    """Return ``sum rho_tau(y_i - x_i' beta)``."""
    total = 0.0
    for row, y in zip(design, response, strict=True):
        r = y - sum(b * x for b, x in zip(beta, row, strict=True))
        total += r * (tau - (1.0 if r < 0.0 else 0.0))
    return total


def enumerate_vertices(design, response, tau: float) -> float:
    """Return the minimum check loss over all elemental bases."""
    p = len(design[0])
    best = math.inf
    for subset in itertools.combinations(range(len(design)), p):
        beta = solve([list(design[i]) for i in subset], [response[i] for i in subset])
        if beta is not None:
            best = min(best, objective(design, response, beta, tau))
    return best


class CheckLossTests(unittest.TestCase):
    """The check loss is the asymmetric absolute value."""

    def test_values(self) -> None:
        """``rho_tau(r) = tau r`` above zero and ``(tau - 1) r`` below."""
        self.assertAlmostEqual(check_loss(2.0, 0.25), 0.5)
        self.assertAlmostEqual(check_loss(-2.0, 0.25), 1.5)
        self.assertEqual(check_loss(0.0, 0.9), 0.0)
        with self.assertRaises(ValueError):
            check_loss(1.0, 1.0)


class StackLossTests(unittest.TestCase):
    """Regression quantiles of the stack-loss data."""

    def test_objective_matches_linprog_and_enumeration(self) -> None:
        """Optimal values from ``linprog`` (HiGHS, SciPy 1.14.1).

        The LP is ``min tau 1'u + (1 - tau) 1'v`` subject to
        ``X b + u - v = y``, ``u, v >= 0``. The median fit is also checked
        against all ``C(21, 4) = 5985`` elemental bases.
        """
        optimum = {
            0.1: 8.546495327102818,
            0.25: 16.62500000000012,
            0.5: 21.040579710145114,
            0.75: 16.252155172413946,
            0.9: 8.361674008810562,
        }
        for tau, value in optimum.items():
            with self.subTest(tau=tau):
                fit = fit_quantile_regression(STACK_DESIGN, LOSS, quantile=tau)
                self.assertAlmostEqual(fit.objective, value, delta=1e-9)
                self.assertTrue(fit.optimality_certified)
                self.assertEqual(len(fit.basis), 4)
                for i in fit.basis:
                    self.assertLess(abs(fit.residuals[i]), 1e-9)
        median = fit_quantile_regression(STACK_DESIGN, LOSS)
        self.assertAlmostEqual(
            median.objective, enumerate_vertices(STACK_DESIGN, LOSS, 0.5), delta=1e-9
        )

    def test_median_coefficients(self) -> None:
        """The unique median-regression vertex, as returned by ``linprog``."""
        fit = fit_quantile_regression(
            STACK_DESIGN, LOSS, column_names=("(Intercept)", "air", "water", "acid")
        )
        expected = (
            -39.68985507246374,
            0.8318840579710131,
            0.5739130434782685,
            -0.060869565217392556,
        )
        for got, want in zip(fit.coefficients, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-9)
        self.assertAlmostEqual(fit.coefficient("air"), expected[1], delta=1e-9)

    def test_subgradient_sign_counts(self) -> None:
        """With an intercept, ``N- <= n tau <= N- + N0`` at every solution."""
        for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
            with self.subTest(tau=tau):
                fit = fit_quantile_regression(STACK_DESIGN, LOSS, quantile=tau)
                negative = sum(r < -1e-9 for r in fit.residuals)
                zero = sum(abs(r) <= 1e-9 for r in fit.residuals)
                self.assertLessEqual(negative, 21 * tau + 1e-12)
                self.assertGreaterEqual(negative + zero, 21 * tau - 1e-12)


class ExactSmallCaseTests(unittest.TestCase):
    """Small designs checked against closed forms and enumeration."""

    def test_intercept_only_is_the_sample_quantile(self) -> None:
        """For ``n tau`` not an integer the solution is ``x_(ceil(n tau))``."""
        sample = [7.0, 1.0, 4.0, 9.0, 2.0, 8.0, 3.0, 6.0, 5.0, 10.0, 0.5]
        ordered = sorted(sample)
        design = [(1.0,)] * len(sample)
        for tau in (0.1, 0.25, 0.5, 0.6, 0.95):
            with self.subTest(tau=tau):
                fit = fit_quantile_regression(design, sample, quantile=tau)
                expected = ordered[math.ceil(len(sample) * tau) - 1]
                self.assertEqual(fit.coefficients[0], expected)

    def test_random_designs_against_enumeration(self) -> None:
        """Objective equals the best elemental basis, ties and integers included."""
        generator = random.Random(4)
        for case in range(40):
            n = generator.randint(6, 11)
            p = generator.randint(1, 3)
            integer = case % 2 == 0
            design = [
                (1.0,)
                + tuple(
                    float(generator.randint(0, 3)) if integer else generator.gauss(0, 1)
                    for _ in range(p - 1)
                )
                for _ in range(n)
            ]
            response = [
                float(generator.randint(0, 4))
                if integer
                else generator.expovariate(1.0)
                for _ in range(n)
            ]
            tau = generator.choice((0.2, 0.5, 0.75))
            try:
                fit = fit_quantile_regression(design, response, quantile=tau)
            except RankDeficientDesignError:
                continue
            with self.subTest(case=case):
                self.assertAlmostEqual(
                    fit.objective, enumerate_vertices(design, response, tau), delta=1e-9
                )

    def test_equivariance(self) -> None:
        """Regression, scale, and sign equivariance of the regression quantile.

        ``beta(y + X g) = beta + g``, ``beta(a y) = a beta``, and
        ``beta_tau(-y) = -beta_{1-tau}`` for a unique solution.
        """
        base = fit_quantile_regression(STACK_DESIGN, LOSS, quantile=0.3)
        shift = (2.0, -0.5, 0.25, 1.0)
        shifted = [
            y + sum(g * x for g, x in zip(shift, row, strict=True))
            for row, y in zip(STACK_DESIGN, LOSS, strict=True)
        ]
        moved = fit_quantile_regression(STACK_DESIGN, shifted, quantile=0.3)
        scaled = fit_quantile_regression(
            STACK_DESIGN, [3.0 * y for y in LOSS], quantile=0.3
        )
        mirrored = fit_quantile_regression(
            STACK_DESIGN, [-y for y in LOSS], quantile=0.7
        )
        for j in range(4):
            self.assertAlmostEqual(
                moved.coefficients[j], base.coefficients[j] + shift[j], delta=1e-8
            )
            self.assertAlmostEqual(
                scaled.coefficients[j], 3.0 * base.coefficients[j], delta=1e-8
            )
            self.assertAlmostEqual(
                mirrored.coefficients[j], -base.coefficients[j], delta=1e-8
            )

    def test_errors(self) -> None:
        """Rank deficiency, invalid levels, and short responses are rejected."""
        with self.assertRaises(RankDeficientDesignError):
            fit_quantile_regression(
                [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)], [1.0, 2.0, 3.0]
            )
        with self.assertRaises(ValueError):
            fit_quantile_regression(STACK_DESIGN, LOSS, quantile=0.0)
        with self.assertRaises(ValueError):
            fit_quantile_regression(STACK_DESIGN, LOSS[:-1])
        with self.assertRaises(ValueError):
            fit_quantile_regression(STACK_DESIGN, LOSS, interval="rank")  # type: ignore[arg-type]


class BootstrapTests(unittest.TestCase):
    """Pairs-bootstrap intervals."""

    def test_reproducible_and_contains_the_estimate(self) -> None:
        """A fixed seed reproduces the interval; the estimate lies inside."""
        first = fit_quantile_regression(
            STACK_DESIGN, LOSS, interval="bootstrap", replicates=100, seed=7
        )
        second = fit_quantile_regression(
            STACK_DESIGN, LOSS, interval="bootstrap", replicates=100, seed=7
        )
        assert first.confidence_intervals is not None
        assert second.confidence_intervals is not None
        assert first.standard_errors is not None
        self.assertEqual(first.confidence_intervals, second.confidence_intervals)
        self.assertEqual(first.replicates, 100)
        self.assertEqual(first.seed, 7)
        for beta, interval, error in zip(
            first.coefficients,
            first.confidence_intervals,
            first.standard_errors,
            strict=True,
        ):
            self.assertLessEqual(interval.lower, beta)
            self.assertGreaterEqual(interval.upper, beta)
            self.assertGreater(error, 0.0)

    def test_coverage_of_the_median_slope(self) -> None:
        """Seeded simulation: 60 data sets, ``n = 40``, 100 replicates each.

        The 90% percentile interval for the median-regression slope (true
        value 2, Laplace-like errors) must cover within three binomial
        standard errors, ``3 sqrt(0.9 0.1 / 60) = 0.116``.
        """
        generator = random.Random(1978)
        x = [i / 4.0 for i in range(40)]
        design = [(1.0, v) for v in x]
        replications = 60
        covered = 0
        for index in range(replications):
            y = [
                1.0 + 2.0 * v + generator.expovariate(1.0) - generator.expovariate(1.0)
                for v in x
            ]
            fit = fit_quantile_regression(
                design,
                y,
                interval="bootstrap",
                replicates=100,
                confidence=0.9,
                seed=index,
            )
            assert fit.confidence_intervals is not None
            interval = fit.confidence_intervals[1]
            covered += interval.lower <= 2.0 <= interval.upper
        coverage = covered / replications
        self.assertLess(abs(coverage - 0.9), 3.0 * math.sqrt(0.9 * 0.1 / replications))


if __name__ == "__main__":
    unittest.main()
