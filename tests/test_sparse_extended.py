"""Tests for IRLS, Huber-TV, TGV^2 (1D, 2D), MCP/SCAD and whitened data terms."""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse import sparse_extended as sparse_module
from probstat.inverse.sparse import (
    l1_regularized_least_squares,
    total_variation_regularized_least_squares,
)
from probstat.inverse.sparse_extended import (
    huber_total_variation_least_squares,
    huber_total_variation_value,
    irls_regularized_least_squares,
    mcp_penalty,
    nonconvex_sparse_least_squares,
    prox_mcp,
    prox_scad,
    scad_penalty,
    tgv2_least_squares_1d,
    tgv2_least_squares_2d,
)
from probstat.linalg import (
    DenseOperator,
    DiagonalCovariance,
    FirstDifferenceOperator,
    GridGradientOperator,
    IdentityOperator,
    ScaledIdentityCovariance,
    solve_spd,
    symmetric_eigen,
)

NAN = float("nan")


def _random_problem(
    seed: int, m: int, n: int, noise: float
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    """Return a dense Gaussian matrix, a sparse truth and noisy data."""
    rng = random.Random(seed)
    a = tuple(
        tuple(rng.gauss(0.0, 1.0) / math.sqrt(m) for _ in range(n)) for _ in range(m)
    )
    truth = [0.0] * n
    truth[2], truth[7], truth[13] = 2.0, -1.5, 1.0
    y = tuple(
        sum(a[i][j] * truth[j] for j in range(n)) + noise * rng.gauss(0.0, 1.0)
        for i in range(m)
    )
    return a, tuple(truth), y


def _tent(n: int) -> list[float]:
    """Piecewise-linear tent signal on ``n`` points with peak value 4."""
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(4.0 * t if t < 0.5 else 4.0 * (1.0 - t))
    return out


def _flat_increments(x: tuple[float, ...]) -> int:
    """Return the number of consecutive entries differing by less than 1e-6."""
    return sum(1 for a, b in zip(x, x[1:], strict=False) if abs(a - b) < 1e-6)


class IRLSTests(unittest.TestCase):
    """Iteratively reweighted norm solver."""

    def test_scalar_stationarity_closed_form(self) -> None:
        """For A = 1 the smoothed minimiser solves x - y + a x/sqrt(x^2+e^2)=0."""
        y, alpha, eps = 2.0, 1.0, 0.1
        lo, hi = 0.0, y
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid - y + alpha * mid / math.hypot(mid, eps) > 0:
                hi = mid
            else:
                lo = mid
        result = irls_regularized_least_squares(
            ((1.0,),), (y,), alpha, epsilon=eps, tolerance=1e-12
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "tolerance")
        self.assertAlmostEqual(result.solution[0], 0.5 * (lo + hi), places=9)
        # As eps -> 0 the soft threshold y - alpha is recovered within eps.
        self.assertLess(abs(result.solution[0] - (y - alpha)), eps)

    def test_matches_lasso_objective_within_smoothing_bound(self) -> None:
        """F(x_eps) lies in [F*, F* + alpha p eps]; lasso gives F*."""
        a, _, y = _random_problem(3, 30, 20, 0.02)
        alpha, n = 0.05, 20
        lasso = l1_regularized_least_squares(a, y, alpha, tolerance=1e-10)
        self.assertTrue(lasso.converged)
        gaps = []
        for eps in (1e-2, 1e-4):
            r = irls_regularized_least_squares(a, y, alpha, epsilon=eps)
            self.assertTrue(r.converged)
            gap = r.objective - lasso.objective
            self.assertGreater(gap, -1e-7)
            self.assertLess(gap, alpha * n * eps)
            gaps.append(gap)
        self.assertLess(gaps[1], gaps[0])

    def test_matches_tv_objective_and_is_monotone(self) -> None:
        """Anisotropic TV: same bound against the PDHG solver; MM monotonicity."""
        a, _, y = _random_problem(3, 30, 20, 0.02)
        alpha, eps = 0.05, 1e-4
        tv = total_variation_regularized_least_squares(
            a, y, alpha, tolerance_absolute=1e-10, tolerance_relative=1e-9
        )
        r = irls_regularized_least_squares(a, y, alpha, penalty="tv", epsilon=eps)
        self.assertTrue(r.converged, r.stop_reason)
        gap = r.objective - tv.objective
        self.assertGreater(gap, -1e-7)
        self.assertLess(gap, alpha * 19 * eps)
        # Exact inner solves up to 1e-10 give a monotone smoothed objective.
        self.assertLess(r.max_objective_increase, 1e-9)
        self.assertEqual(r.penalty_name, "tv")

    def test_bookkeeping(self) -> None:
        """Iteration counts, histories and operator counts are consistent."""
        a, _, y = _random_problem(5, 20, 14, 0.01)
        r = irls_regularized_least_squares(a, y, 0.1, epsilon=1e-3)
        self.assertEqual(len(r.inner_iteration_history), r.outer_iterations)
        self.assertEqual(sum(r.inner_iteration_history), r.inner_iterations)
        self.assertEqual(len(r.objective_history), r.outer_iterations + 1)
        self.assertEqual(len(r.true_objective_history), r.outer_iterations + 1)
        self.assertAlmostEqual(r.smoothed_objective, r.objective_history[-1])
        self.assertAlmostEqual(r.objective, r.data_misfit + r.penalty)
        self.assertGreater(r.operator_counts.forward, r.inner_iterations)
        self.assertGreaterEqual(r.smoothed_objective, r.objective)

    def test_budget_and_inner_failure_are_reported(self) -> None:
        """A tiny outer budget or inner budget is never reported as converged."""
        a, _, y = _random_problem(3, 30, 20, 0.02)
        short = irls_regularized_least_squares(a, y, 0.05, max_iterations=2)
        self.assertEqual(short.stop_reason, "max_iterations")
        self.assertFalse(short.converged)
        self.assertEqual(short.outer_iterations, 2)
        starved = irls_regularized_least_squares(
            a, y, 0.05, penalty="tv", inner_max_iterations=1
        )
        self.assertGreater(starved.inner_failures, 0)
        self.assertFalse(starved.converged)

    def test_invalid_input(self) -> None:
        """Invalid parameters and non-finite data are rejected."""
        a = ((1.0, 0.0), (0.0, 1.0))
        y = (1.0, 2.0)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, y, 0.0)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, y, 1.0, epsilon=0.0)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, y, 1.0, epsilon=NAN)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, (1.0, NAN), 1.0)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, y, 1.0, penalty="l2")
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, y, 1.0, initial=(NAN, 0.0))
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, (1.0,), 1.0)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(
                a, y, 1.0, penalty_operator=IdentityOperator(3)
            )
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(a, y, 1.0, max_iterations=0)
        with self.assertRaises(TypeError):
            irls_regularized_least_squares(a, y, 1.0, max_iterations=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            irls_regularized_least_squares(a, y, "1")  # type: ignore[arg-type]


class HuberTVTests(unittest.TestCase):
    """Huber-smoothed total variation."""

    def setUp(self) -> None:
        """Build a small deconvolution-like problem."""
        rng = random.Random(5)
        m, n = 25, 15
        self.a = tuple(
            tuple(rng.gauss(0.0, 1.0) / math.sqrt(m) for _ in range(n))
            for _ in range(m)
        )
        truth = [0.0] * 5 + [1.0] * 5 + [0.3] * 5
        self.y = tuple(
            sum(self.a[i][j] * truth[j] for j in range(n)) + 0.02 * rng.gauss(0, 1)
            for i in range(m)
        )
        self.alpha = 0.05
        self.n = n

    def test_value_closed_form_and_bounds(self) -> None:
        """h(1)=1/(2 d) with d=1.5 and h(2)=2-d/2; |s|-d/2 <= h <= |s|."""
        value = huber_total_variation_value((0.0, 1.0, 3.0), 1.5)
        self.assertAlmostEqual(value, 1.0 / 3.0 + 1.25, places=12)
        tv = 3.0
        self.assertLessEqual(value, tv)
        self.assertGreaterEqual(value, tv - 2 * 1.5 / 2)

    def test_approaches_tv_as_delta_shrinks(self) -> None:
        """|F_delta - F_TV| <= alpha p delta/2, decreasing with delta."""
        tv = total_variation_regularized_least_squares(
            self.a,
            self.y,
            self.alpha,
            tolerance_absolute=1e-10,
            tolerance_relative=1e-9,
            max_iterations=100000,
        )
        self.assertTrue(tv.converged)
        gaps = []
        for delta in (1.0, 0.1, 0.01):
            h = huber_total_variation_least_squares(self.a, self.y, self.alpha, delta)
            self.assertTrue(h.converged, h.stop_reason)
            self.assertLess(h.gradient_norm, 1e-6)
            gap = abs(h.objective - tv.objective)
            self.assertLessEqual(gap, self.alpha * (self.n - 1) * delta / 2 + 1e-8)
            self.assertLessEqual(h.objective, h.tv_objective)
            gaps.append(gap)
        self.assertTrue(gaps[0] > gaps[1] > gaps[2])
        # Step condition tau sigma L^2 < 1 holds.
        self.assertLess(h.optimization.step_product, 1.0)

    def test_large_delta_is_quadratic_smoothing(self) -> None:
        """For huge delta the solution is the Tikhonov one, lambda^2 = alpha/delta."""
        delta = 1e3
        h = huber_total_variation_least_squares(self.a, self.y, self.alpha, delta)
        self.assertTrue(h.converged)
        n = self.n
        mat = [
            [sum(row[i] * row[j] for row in self.a) for j in range(n)] for i in range(n)
        ]
        for i in range(n - 1):
            for p, q, s in (
                (i, i, 1),
                (i + 1, i + 1, 1),
                (i, i + 1, -1),
                (i + 1, i, -1),
            ):
                mat[p][q] += self.alpha / delta * s
        rhs = tuple(
            sum(self.a[k][i] * self.y[k] for k in range(len(self.y))) for i in range(n)
        )
        reference = solve_spd(tuple(tuple(r) for r in mat), rhs)
        error = max(abs(u - v) for u, v in zip(h.solution, reference, strict=True))
        # Valid exactly when every |D x| <= delta, which holds here.
        self.assertLess(
            max(abs(g) for g in FirstDifferenceOperator(n).matvec(h.solution)), delta
        )
        self.assertLess(error, 1e-6)

    def test_isotropic_grid(self) -> None:
        """A 3 x 3 grid denoising problem is solved to a small gradient norm."""
        rng = random.Random(1)
        truth = [0.0, 0.0, 1.0] * 3
        y = tuple(v + 0.05 * rng.gauss(0, 1) for v in truth)
        grad = GridGradientOperator(3, 3)
        for isotropic in (True, False):
            h = huber_total_variation_least_squares(
                IdentityOperator(9), y, 0.1, 0.05, gradient=grad, isotropic=isotropic
            )
            self.assertTrue(h.converged, h.stop_reason)
            self.assertLess(h.gradient_norm, 1e-5)

    def test_invalid_input(self) -> None:
        """Bad parameters, bad data and an uncertified gradient are rejected."""
        eye = IdentityOperator(3)
        y = (1.0, 2.0, 3.0)
        with self.assertRaises(ValueError):
            huber_total_variation_least_squares(eye, y, 0.1, 0.0)
        with self.assertRaises(ValueError):
            huber_total_variation_least_squares(eye, y, 0.0, 0.1)
        with self.assertRaises(ValueError):
            huber_total_variation_least_squares(eye, (1.0, NAN, 3.0), 0.1, 0.1)
        with self.assertRaises(ValueError):
            huber_total_variation_least_squares(
                eye, y, 0.1, 0.1, gradient=DenseOperator(((1.0, -1.0, 0.0),))
            )
        with self.assertRaises(ValueError):
            huber_total_variation_least_squares(eye, y, 0.1, 0.1, step_safety=1.0)


class TGVTests(unittest.TestCase):
    """Second-order total generalized variation in one dimension."""

    def test_norm_bound_is_certified(self) -> None:
        """The step bound dominates the dense spectral norm of K (n = 6)."""
        n = 6
        rows: list[list[float]] = []
        for i in range(n):
            rows.append([1.0 if j == i else 0.0 for j in range(n)] + [0.0] * (n - 1))
        for i in range(n - 1):
            row = [0.0] * (2 * n - 1)
            row[i], row[i + 1], row[n + i] = -1.0, 1.0, -1.0
            rows.append(row)
        for i in range(n - 2):
            row = [0.0] * (2 * n - 1)
            row[n + i], row[n + i + 1] = -1.0, 1.0
            rows.append(row)
        gram = tuple(
            tuple(sum(r[i] * r[j] for r in rows) for j in range(2 * n - 1))
            for i in range(2 * n - 1)
        )
        top = math.sqrt(symmetric_eigen(gram)[0][-1])
        result = tgv2_least_squares_1d(IdentityOperator(n), (0.0,) * n, 1.0, 1.0)
        self.assertLessEqual(top, result.norm_bound)
        self.assertLess(result.optimization.step_product, 1.0)
        self.assertIn("certified", result.norm_bound_kind)

    def test_affine_data_have_zero_penalty(self) -> None:
        """Affine data are reproduced: TGV^2 is zero on ramps."""
        y = tuple(0.5 + 0.25 * i for i in range(8))
        r = tgv2_least_squares_1d(IdentityOperator(8), y, 1.0, 1.0)
        self.assertTrue(r.converged, r.stop_reason)
        self.assertLess(
            max(abs(a - b) for a, b in zip(r.solution, y, strict=True)), 1e-5
        )
        self.assertLess(r.objective, 1e-6)
        self.assertLess(max(abs(v - 0.25) for v in r.auxiliary), 1e-5)

    def test_upper_bound_by_tv_objective(self) -> None:
        """The choice v = 0 is feasible: TGV objective <= TV objective (alpha1)."""
        rng = random.Random(2)
        n = 20
        y = tuple(v + 0.2 * rng.gauss(0, 1) for v in _tent(n))
        tv = total_variation_regularized_least_squares(IdentityOperator(n), y, 0.3)
        r = tgv2_least_squares_1d(
            IdentityOperator(n),
            y,
            0.3,
            1.0,
            tolerance_absolute=1e-6,
            tolerance_relative=1e-5,
        )
        self.assertTrue(r.converged, r.stop_reason)
        self.assertLessEqual(r.objective, tv.objective + 1e-8)

    def _compare(
        self, n: int, seed: int, noise: float
    ) -> tuple[float, float, int, int]:
        """Return (best TV rmse, TGV rmse, TV flat count, TGV flat count)."""
        rng = random.Random(seed)
        truth = _tent(n)
        y = tuple(v + noise * rng.gauss(0, 1) for v in truth)

        def rmse(x: tuple[float, ...]) -> float:
            return math.sqrt(
                sum((a - b) ** 2 for a, b in zip(x, truth, strict=True)) / n
            )

        eye = IdentityOperator(n)
        best_error, best_flat = math.inf, 0
        for alpha in (0.1, 0.2, 0.4):
            tv = total_variation_regularized_least_squares(
                eye, y, alpha, max_iterations=20000
            )
            self.assertTrue(tv.converged)
            error = rmse(tv.solution)
            if error < best_error:
                best_error, best_flat = error, _flat_increments(tv.solution)
        tgv = tgv2_least_squares_1d(
            eye,
            y,
            0.3,
            1.0,
            max_iterations=40000,
            tolerance_absolute=1e-5,
            tolerance_relative=1e-4,
        )
        self.assertTrue(tgv.converged, tgv.stop_reason)
        return best_error, rmse(tgv.solution), best_flat, _flat_increments(tgv.solution)

    def test_tgv_avoids_staircasing_fast(self) -> None:
        """Piecewise-linear signal: TV (best alpha over a grid) staircases."""
        tv_error, tgv_error, tv_flat, tgv_flat = self._compare(40, 11, 0.15)
        self.assertLess(tgv_error, 0.6 * tv_error)
        self.assertGreaterEqual(tv_flat, 10)
        self.assertLessEqual(tgv_flat, 2)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_tgv_avoids_staircasing_seeds(self) -> None:
        """Five noise seeds at n = 60: TGV beats the oracle-tuned TV every time."""
        wins = 0
        ratios = []
        for seed in range(5):
            tv_error, tgv_error, _, tgv_flat = self._compare(60, 100 + seed, 0.15)
            ratios.append(tgv_error / tv_error)
            wins += tgv_error < tv_error
            self.assertLessEqual(tgv_flat, 3)
        print(f"TGV/TV rmse ratios: {[round(r, 3) for r in ratios]}")
        self.assertEqual(wins, 5)

    def test_invalid_input(self) -> None:
        """Bad weights, sizes, steps and data are rejected."""
        eye = IdentityOperator(5)
        y = (0.0,) * 5
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(eye, y, 0.0, 1.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(eye, y, 1.0, -1.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(IdentityOperator(2), (0.0, 0.0), 1.0, 1.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(eye, (0.0, NAN, 0.0, 0.0, 0.0), 1.0, 1.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(eye, y, 1.0, 1.0, step_safety=1.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(eye, y, 1.0, 1.0, spacing=0.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_1d(eye, y, 1.0, 1.0, forward_norm_bound=-1.0)


def _mcp(t: float, lam: float, gamma: float) -> float:
    """Scalar MCP written independently of the library."""
    s = abs(t)
    return lam * s - s * s / (2 * gamma) if s <= gamma * lam else gamma * lam * lam / 2


def _scad(t: float, lam: float, a: float) -> float:
    """Scalar SCAD written independently of the library."""
    s = abs(t)
    if s <= lam:
        return lam * s
    if s <= a * lam:
        return (2 * a * lam * s - s * s - lam * lam) / (2 * (a - 1))
    return (a + 1) * lam * lam / 2


class NonconvexTests(unittest.TestCase):
    """MCP and SCAD proximal maps and proximal gradient."""

    def _brute_force(self, penalty, prox, cases) -> float:  # type: ignore[no-untyped-def]
        """Return the worst |closed form - grid minimiser| over the cases."""
        h = 0.004
        worst = 0.0
        for lam, shape, step in cases:
            for i in range(-12, 13):
                v = i * 0.4
                n_grid = int((2 * abs(v) + 2) / h)
                best = min(
                    (-abs(v) - 1 + k * h for k in range(n_grid + 1)),
                    key=lambda x: 0.5 * (x - v) ** 2 + step * penalty(x, lam, shape),
                )
                worst = max(worst, abs(prox((v,), step, lam, shape)[0] - best))
        return worst

    def test_mcp_prox_matches_grid_minimisation(self) -> None:
        """The closed form equals the brute-force minimiser to the grid step."""
        cases = [(1.0, 1.5, 0.2), (1.0, 3.0, 1.4), (0.5, 3.0, 0.9), (1.0, 1.5, 1.4)]
        self.assertLessEqual(self._brute_force(_mcp, prox_mcp, cases), 0.004)

    def test_scad_prox_matches_grid_minimisation(self) -> None:
        """The closed form equals the brute-force minimiser to the grid step."""
        cases = [(1.0, 2.5, 0.2), (1.0, 3.7, 1.4), (0.5, 3.7, 2.0), (1.0, 2.5, 1.4)]
        self.assertLessEqual(self._brute_force(_scad, prox_scad, cases), 0.004)

    def test_prox_properties(self) -> None:
        """Continuity at the breakpoints, oddness, and no bias for large |v|."""
        lam, gamma, s = 1.0, 3.0, 0.5
        for v in (s * lam, gamma * lam):
            below = prox_mcp((v - 1e-12,), s, lam, gamma)[0]
            above = prox_mcp((v + 1e-12,), s, lam, gamma)[0]
            self.assertAlmostEqual(below, above, places=9)
        lam, a = 1.0, 3.7
        for v in ((1 + s) * lam, a * lam):
            below = prox_scad((v - 1e-12,), s, lam, a)[0]
            above = prox_scad((v + 1e-12,), s, lam, a)[0]
            self.assertAlmostEqual(below, above, places=9)
        self.assertEqual(prox_mcp((10.0,), s, lam, gamma), (10.0,))
        self.assertEqual(prox_scad((10.0,), s, lam, a), (10.0,))
        self.assertEqual(
            prox_mcp((-2.0,), s, lam, gamma),
            tuple(-v for v in prox_mcp((2.0,), s, lam, gamma)),
        )
        self.assertEqual(prox_mcp((0.2,), s, lam, gamma), (0.0,))

    def test_penalty_values(self) -> None:
        """Penalty values equal the independent scalar formulas and plateau."""
        xs = (-5.0, -1.0, -0.3, 0.0, 0.7, 2.0, 9.0)
        self.assertAlmostEqual(
            mcp_penalty(xs, 0.8, 2.5), sum(_mcp(t, 0.8, 2.5) for t in xs)
        )
        self.assertAlmostEqual(
            scad_penalty(xs, 0.8, 3.7), sum(_scad(t, 0.8, 3.7) for t in xs)
        )
        self.assertAlmostEqual(mcp_penalty((7.0,), 1.0, 2.0), 1.0)
        self.assertAlmostEqual(scad_penalty((7.0,), 1.0, 3.0), 2.0)

    def test_step_and_parameter_violations(self) -> None:
        """A step at or above the convexity limit, or a bad parameter, raises."""
        with self.assertRaises(ValueError):
            prox_mcp((1.0,), 3.0, 1.0, 3.0)
        with self.assertRaises(ValueError):
            prox_mcp((1.0,), 0.5, 1.0, 1.0)
        with self.assertRaises(ValueError):
            prox_mcp((1.0,), 0.0, 1.0, 3.0)
        with self.assertRaises(ValueError):
            prox_mcp((1.0,), 0.5, -1.0, 3.0)
        with self.assertRaises(ValueError):
            prox_mcp((NAN,), 0.5, 1.0, 3.0)
        with self.assertRaises(ValueError):
            prox_scad((1.0,), 2.7, 1.0, 3.7)
        with self.assertRaises(ValueError):
            prox_scad((1.0,), 0.5, 1.0, 2.0)
        with self.assertRaises(ValueError):
            prox_scad((float("inf"),), 0.5, 1.0, 3.7)
        with self.assertRaises(ValueError):
            mcp_penalty((1.0,), 1.0, 0.5)
        with self.assertRaises(ValueError):
            scad_penalty((1.0,), 1.0, 1.5)
        with self.assertRaises(TypeError):
            prox_mcp((1.0,), "a", 1.0, 3.0)  # type: ignore[arg-type]

    def test_proximal_gradient_is_stationary_and_monotone(self) -> None:
        """MCP/SCAD iteration returns a stationary point with a decreasing objective."""
        a, truth, y = _random_problem(7, 30, 16, 0.01)
        for penalty in ("mcp", "scad"):
            r = nonconvex_sparse_least_squares(a, y, 0.05, penalty=penalty)
            self.assertTrue(r.converged, r.stop_reason)
            hist = r.optimization.objective_history
            self.assertTrue(
                all(b <= a_ + 1e-12 for a_, b in zip(hist, hist[1:], strict=False))
            )
            self.assertLess(r.gradient_mapping_norm, 1e-9)
            self.assertIn("stationary", r.guarantee)
            self.assertEqual({2, 7, 13} & set(r.support), {2, 7, 13})
            # Nearly unbiased on the large coefficients (lasso shrinks them).
            self.assertLess(abs(r.solution[2] - truth[2]), 0.1)

    def test_result_depends_on_the_start(self) -> None:
        """Two starts give two stationary points with different objectives."""
        a = ((1.0, 1.0),)
        y = (1.0,)
        left = nonconvex_sparse_least_squares(a, y, 0.3, shape=2.0, initial=(1.0, 0.0))
        centre = nonconvex_sparse_least_squares(a, y, 0.3, shape=2.0)
        for r in (left, centre):
            self.assertTrue(r.converged)
            self.assertLess(r.gradient_mapping_norm, 1e-9)
        self.assertEqual(left.support, (0,))
        self.assertEqual(centre.support, (0, 1))
        self.assertGreater(centre.objective - left.objective, 0.05)
        # Mirror start gives the mirror solution with the same objective.
        right = nonconvex_sparse_least_squares(a, y, 0.3, shape=2.0, initial=(0.0, 1.0))
        self.assertEqual(right.support, (1,))
        self.assertAlmostEqual(right.objective, left.objective, places=12)
        # The convex lasso objective does not depend on the start.
        l1a = l1_regularized_least_squares(a, y, 0.3, initial=(1.0, 0.0))
        l1b = l1_regularized_least_squares(a, y, 0.3, initial=(0.0, 0.0))
        self.assertAlmostEqual(l1a.objective, l1b.objective, places=7)

    def test_invalid_input(self) -> None:
        """Bad penalty names, parameters and data are rejected."""
        a = ((1.0, 1.0),)
        with self.assertRaises(ValueError):
            nonconvex_sparse_least_squares(a, (1.0,), 0.3, penalty="l0")
        with self.assertRaises(ValueError):
            nonconvex_sparse_least_squares(a, (1.0,), 0.3, shape=1.0)
        with self.assertRaises(ValueError):
            nonconvex_sparse_least_squares(a, (1.0,), 0.3, penalty="scad", shape=2.0)
        with self.assertRaises(ValueError):
            nonconvex_sparse_least_squares(a, (NAN,), 0.3)
        with self.assertRaises(ValueError):
            nonconvex_sparse_least_squares(a, (1.0,), -0.3)
        with self.assertRaises(ValueError):
            nonconvex_sparse_least_squares(a, (1.0,), 0.3, initial=(NAN, 0.0))


# ---------------------------------------------------------------------------
# Round 2: TGV^2 on grids, isotropic IRLS, whitened data terms
# ---------------------------------------------------------------------------

Matrix = list[list[float]]


def _dense_k_tgv2d(
    a: tuple[tuple[float, ...], ...], r: int, c: int, hy: float, hx: float
) -> Matrix:
    """Build the dense primal-dual matrix K of the 2D TGV^2 model by definition.

    Rows: data ``A x``; ``M (G x - v)`` as ``[dx comps; dy comps]``; ``E v`` as
    ``[e11; e22; (D_y v_1 + D_x v_2)/sqrt 2]``. Columns: ``x, v_1, v_2``.
    """
    n = r * c
    dx = [[0.0] * n for _ in range(n)]
    dy = [[0.0] * n for _ in range(n)]
    for i in range(r):
        for j in range(c):
            k = i * c + j
            if j + 1 < c:
                dx[k][k], dx[k][k + 1] = -1.0 / hx, 1.0 / hx
            if i + 1 < r:
                dy[k][k], dy[k][k + c] = -1.0 / hy, 1.0 / hy
    zero = [0.0] * n
    root = math.sqrt(0.5)
    rows: Matrix = [list(row) + [0.0] * (2 * n) for row in a]
    for k in range(n):
        valid = (k % c) != c - 1
        unit = [-1.0 if q == k else 0.0 for q in range(n)]
        rows.append(dx[k] + unit + zero if valid else [0.0] * (3 * n))
    for k in range(n):
        valid = (k // c) != r - 1
        unit = [-1.0 if q == k else 0.0 for q in range(n)]
        rows.append(dy[k] + zero + unit if valid else [0.0] * (3 * n))
    for k in range(n):
        rows.append(zero + dx[k] + zero)
    for k in range(n):
        rows.append(zero + zero + dy[k])
    for k in range(n):
        rows.append(
            zero + [root * value for value in dy[k]] + [root * value for value in dx[k]]
        )
    return rows


def _dense_apply(k: Matrix, z: list[float]) -> list[float]:
    """Return ``K z`` for a dense matrix."""
    return [sum(a * b for a, b in zip(row, z, strict=True)) for row in k]


def _tgv2d_terms(
    u: list[float], m: int, n: int, y: tuple[float, ...], a1: float, a0: float
) -> tuple[float, float]:
    """Return ``(F, data term)`` of the TGV^2 objective from ``u = K z``."""
    fit = 0.5 * sum((p - q) ** 2 for p, q in zip(u[:m], y, strict=True))
    first = sum(math.hypot(u[m + i], u[m + n + i]) for i in range(n))
    second = sum(
        math.hypot(u[m + 2 * n + i], u[m + 3 * n + i], u[m + 4 * n + i])
        for i in range(n)
    )
    return fit + a1 * first + a0 * second, fit


def _smoothed_oracle(
    k: Matrix,
    m: int,
    n: int,
    y: tuple[float, ...],
    a1: float,
    a0: float,
    schedule: tuple[float, ...],
) -> list[float]:
    """Minimise the eps-smoothed TGV^2 objective by damped Newton (dense).

    ``F_eps = 1/2||u_d - y||^2 + a1 sum sqrt(|b1_i|^2 + eps^2) + a0 sum
    sqrt(|b2_i|^2 + eps^2)`` with ``u = K z``; ``F <= F_eps <= F + (a1 + a0) n
    eps``. Independent of PDHG: a different algorithm on the same K.
    """
    rows_total = len(k)
    dim = len(k[0])
    groups: list[tuple[list[int], float]] = [
        ([m + i, m + n + i], a1) for i in range(n)
    ] + [([m + 2 * n + i, m + 3 * n + i, m + 4 * n + i], a0) for i in range(n)]

    def value(z: list[float], eps: float) -> float:
        u = _dense_apply(k, z)
        total = 0.5 * sum((p - q) ** 2 for p, q in zip(u[:m], y, strict=True))
        for idx, weight in groups:
            total += weight * math.sqrt(sum(u[t] ** 2 for t in idx) + eps * eps)
        return total

    z = [0.0] * dim
    for eps in schedule:
        for _ in range(200):
            u = _dense_apply(k, z)
            g = [0.0] * rows_total
            h = [[0.0] * rows_total for _ in range(rows_total)]
            for t in range(m):
                g[t] = u[t] - y[t]
                h[t][t] = 1.0
            for idx, weight in groups:
                s = math.sqrt(sum(u[t] ** 2 for t in idx) + eps * eps)
                for t in idx:
                    g[t] = weight * u[t] / s
                    for q in idx:
                        diag = 1.0 / s if t == q else 0.0
                        h[t][q] = weight * (diag - u[t] * u[q] / s**3)
            grad = [sum(k[t][j] * g[t] for t in range(rows_total)) for j in range(dim)]
            if math.sqrt(sum(v * v for v in grad)) < 1e-11:
                break
            hk = [
                [sum(h[t][q] * k[q][j] for q in range(rows_total)) for j in range(dim)]
                for t in range(rows_total)
            ]
            hess = tuple(
                tuple(
                    sum(k[t][i] * hk[t][j] for t in range(rows_total))
                    + (1e-12 if i == j else 0.0)
                    for j in range(dim)
                )
                for i in range(dim)
            )
            step = solve_spd(hess, tuple(-v for v in grad))
            base = value(z, eps)
            slope = sum(a * b for a, b in zip(grad, step, strict=True))
            t_len = 1.0
            accepted = False
            while t_len > 1e-12:
                trial = [a + t_len * b for a, b in zip(z, step, strict=True)]
                if value(trial, eps) <= base + 1e-4 * t_len * slope:
                    accepted = True
                    break
                t_len *= 0.5
            if not accepted:
                break
            z = trial
    return z


def _flat_horizontal(x: tuple[float, ...], rows: int, cols: int, tol: float) -> int:
    """Count horizontal neighbours whose values differ by less than ``tol``."""
    return sum(
        1
        for i in range(rows)
        for j in range(cols - 1)
        if abs(x[i * cols + j + 1] - x[i * cols + j]) < tol
    )


def _eye(n: int) -> tuple[tuple[float, ...], ...]:
    """Return the dense ``n x n`` identity."""
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))


class TGV2DTests(unittest.TestCase):
    """Second-order TGV on a 2D grid."""

    def test_gradient_helpers_match_the_library_operator(self) -> None:
        """The inlined Neumann differences equal GridGradientOperator and adjoint."""
        rng = random.Random(4)
        r, c, hy, hx = 3, 4, 0.5, 2.0
        op = GridGradientOperator(r, c, spacing=(hy, hx))
        u = tuple(rng.gauss(0, 1) for _ in range(r * c))
        w = tuple(rng.gauss(0, 1) for _ in range(2 * r * c))
        ours = sparse_module._grid_gradient(u, r, c, hy, hx)
        for a, b in zip(ours, op.matvec(u), strict=True):
            self.assertAlmostEqual(a, b, places=14)
        back = sparse_module._grid_gradient_transpose(w, r, c, hy, hx)
        for a, b in zip(back, op.rmatvec(w), strict=True):
            self.assertAlmostEqual(a, b, places=14)

    def test_norm_bound_is_certified(self) -> None:
        """The step bound dominates the dense spectral norm of K (2 x 3 grid)."""
        r, c = 2, 3
        n = r * c
        eye = _eye(n)
        for hy, hx in ((1.0, 1.0), (0.5, 2.0), (2.0, 0.25)):
            k = _dense_k_tgv2d(eye, r, c, hy, hx)
            gram = tuple(
                tuple(sum(row[i] * row[j] for row in k) for j in range(3 * n))
                for i in range(3 * n)
            )
            top = math.sqrt(symmetric_eigen(gram)[0][-1])
            result = tgv2_least_squares_2d(
                eye, (0.0,) * n, 1.0, 1.0, r, c, spacing=(hy, hx), max_iterations=3
            )
            grad_sq = 4.0 / hx**2 + 4.0 / hy**2
            self.assertAlmostEqual(
                result.norm_bound, math.sqrt(n + 2.0 * grad_sq + 1.0), places=12
            )
            self.assertLessEqual(top, result.norm_bound)
            self.assertLess(result.optimization.step_product, 1.0)
            self.assertIn("certified", result.norm_bound_kind)
        # Square unit grid: ||G||^2 <= 8/h^2, so ||K||^2 <= ||A||_F^2 + 17.
        square = tgv2_least_squares_2d(
            eye, (0.0,) * n, 1.0, 1.0, r, c, max_iterations=1
        )
        self.assertAlmostEqual(square.norm_bound**2, n + 17.0, places=12)

    def test_affine_image_has_zero_penalty(self) -> None:
        """An affine image is reproduced: TGV^2 vanishes on ramps (v constant)."""
        r, c = 4, 5
        y = tuple(0.5 + 0.3 * i + 0.7 * j for i in range(r) for j in range(c))
        res = tgv2_least_squares_2d(
            IdentityOperator(r * c),
            y,
            1.0,
            1.0,
            r,
            c,
            tolerance_absolute=1e-8,
            tolerance_relative=1e-7,
        )
        self.assertTrue(res.converged, res.stop_reason)
        self.assertLess(res.objective, 1e-6)
        self.assertLess(
            max(abs(a - b) for a, b in zip(res.solution, y, strict=True)), 1e-4
        )
        v1, v2 = res.auxiliary[: r * c], res.auxiliary[r * c :]
        self.assertLess(max(abs(v - 0.7) for v in v1), 1e-3)
        self.assertLess(max(abs(v - 0.3) for v in v2), 1e-3)
        flat = tgv2_least_squares_2d(IdentityOperator(6), (2.0,) * 6, 1.0, 1.0, 2, 3)
        self.assertLess(flat.objective, 1e-7)

    def test_objective_matches_the_dense_definition(self) -> None:
        """Reported objective and terms equal the dense K evaluation at (x, v)."""
        r, c, hy, hx = 2, 3, 0.7, 1.3
        n = r * c
        rng = random.Random(6)
        y = tuple(rng.gauss(0, 1) for _ in range(n))
        eye = _eye(n)
        res = tgv2_least_squares_2d(
            eye, y, 0.4, 0.3, r, c, spacing=(hy, hx), max_iterations=50
        )
        k = _dense_k_tgv2d(eye, r, c, hy, hx)
        u = _dense_apply(k, list(res.solution) + list(res.auxiliary))
        total, fit = _tgv2d_terms(u, n, n, y, 0.4, 0.3)
        self.assertAlmostEqual(res.objective, total, places=12)
        self.assertAlmostEqual(res.data_misfit, fit, places=12)
        self.assertAlmostEqual(
            res.objective,
            res.data_misfit + res.first_order_term + res.second_order_term,
        )

    def test_matches_a_smoothed_newton_oracle(self) -> None:
        """PDHG optimum equals the optimum of the eps-smoothed problem (dense Newton).

        F <= F_eps <= F + (a1 + a0) n eps, so for the oracle point
        0 <= F(z_eps) - F* <= (a1 + a0) n eps; the PDHG point has
        F(z_p) >= F*. Hence F(z_p) - F(z_eps) >= -(a1 + a0) n eps and, with
        PDHG converged, F(z_p) - F(z_eps) <= tol. The data term is
        1-strongly convex in x, so ||x_p - x_eps||^2 <= 2 (F(z_p) - F* +
        F(z_eps) - F*).
        """
        r, c = 2, 3
        n = r * c
        a1, a0 = 0.4, 0.25
        rng = random.Random(21)
        y = tuple(
            rng.gauss(0.0, 1.0) + 0.6 * (j - 1) for _ in range(r) for j in range(c)
        )
        eye = _eye(n)
        k = _dense_k_tgv2d(eye, r, c, 1.0, 1.0)
        eps = 1e-5
        z_eps = _smoothed_oracle(k, n, n, y, a1, a0, (1e-1, 1e-3, eps))
        f_eps, _ = _tgv2d_terms(_dense_apply(k, z_eps), n, n, y, a1, a0)
        res = tgv2_least_squares_2d(
            eye,
            y,
            a1,
            a0,
            r,
            c,
            tolerance_absolute=1e-10,
            tolerance_relative=1e-9,
            max_iterations=60000,
        )
        self.assertTrue(res.converged, res.stop_reason)
        bound = (a1 + a0) * n * eps
        self.assertGreaterEqual(res.objective - f_eps, -bound - 1e-9)
        self.assertLessEqual(res.objective - f_eps, 1e-6)
        dist = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(res.solution, z_eps[:n], strict=True))
        )
        self.assertLessEqual(dist, math.sqrt(2.0 * (1e-6 + bound)) + 1e-9)

    def test_upper_bound_by_isotropic_tv_objective(self) -> None:
        """With v = 0 feasible, min TGV objective <= the isotropic TV optimum."""
        r, c = 4, 4
        rng = random.Random(9)
        y = tuple(0.4 * j + 0.2 * rng.gauss(0, 1) for _ in range(r) for j in range(c))
        eye = IdentityOperator(r * c)
        tv = total_variation_regularized_least_squares(
            eye,
            y,
            0.3,
            gradient=GridGradientOperator(r, c),
            tolerance_absolute=1e-9,
            tolerance_relative=1e-8,
            max_iterations=40000,
        )
        self.assertTrue(tv.converged)
        res = tgv2_least_squares_2d(
            eye,
            y,
            0.3,
            0.8,
            r,
            c,
            tolerance_absolute=1e-8,
            tolerance_relative=1e-7,
            max_iterations=40000,
        )
        self.assertTrue(res.converged, res.stop_reason)
        self.assertLessEqual(res.objective, tv.objective + 1e-6)

    def _staircase(
        self,
        rows: int,
        cols: int,
        seed: int,
        noise: float,
        alphas: tuple[float, ...] = (0.1, 0.2),
    ) -> tuple[float, float, int, int]:
        """Return (best TV rmse, TGV rmse, TV flat count, TGV flat count)."""
        truth = tuple(
            2.0 * (j / (cols - 1))
            if j / (cols - 1) < 0.5
            else 2.0 * (1.0 - j / (cols - 1))
            for _ in range(rows)
            for j in range(cols)
        )
        rng = random.Random(seed)
        y = tuple(v + noise * rng.gauss(0, 1) for v in truth)
        n = rows * cols

        def rmse(x: tuple[float, ...]) -> float:
            return math.sqrt(
                sum((a - b) ** 2 for a, b in zip(x, truth, strict=True)) / n
            )

        eye = IdentityOperator(n)
        best_error, best_flat = math.inf, 0
        for alpha in alphas:
            tv = total_variation_regularized_least_squares(
                eye,
                y,
                alpha,
                gradient=GridGradientOperator(rows, cols),
                tolerance_absolute=1e-6,
                tolerance_relative=1e-5,
                max_iterations=30000,
            )
            self.assertTrue(tv.converged)
            error = rmse(tv.solution)
            if error < best_error:
                best_error = error
                best_flat = _flat_horizontal(tv.solution, rows, cols, 0.02)
        tgv = tgv2_least_squares_2d(
            eye,
            y,
            0.3,
            0.6,
            rows,
            cols,
            tolerance_absolute=1e-4,
            tolerance_relative=1e-3,
            max_iterations=40000,
        )
        self.assertTrue(tgv.converged, tgv.stop_reason)
        return (
            best_error,
            rmse(tgv.solution),
            best_flat,
            _flat_horizontal(tgv.solution, rows, cols, 0.02),
        )

    def test_tgv_avoids_staircasing_2d_fast(self) -> None:
        """3 x 14 image, gentle tent: TV (best alpha) staircases, TGV does not.

        Measured in development (seed 3, noise 0.3, tolerance 1e-4/1e-3): TV
        flat 12 of 39 increments, rmse 0.170; TGV flat 0, rmse 0.143.
        """
        tv_error, tgv_error, tv_flat, tgv_flat = self._staircase(3, 14, 3, 0.3)
        self.assertLess(tgv_error, 0.95 * tv_error)
        self.assertGreaterEqual(tv_flat, 8)
        self.assertLessEqual(tgv_flat, 2)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_tgv_avoids_staircasing_2d_seeds(self) -> None:
        """Six seeds on 4 x 20: TGV beats oracle-tuned TV and has no plateaus."""
        ratios = []
        for seed in range(6):
            tv_error, tgv_error, tv_flat, tgv_flat = self._staircase(
                4, 20, 50 + seed, 0.3, (0.1, 0.2, 0.4)
            )
            ratios.append(round(tgv_error / tv_error, 3))
            self.assertGreaterEqual(tv_flat, 10)
            self.assertLessEqual(tgv_flat, 4)
        print(f"2D TGV/TV rmse ratios: {ratios}")
        self.assertTrue(all(ratio < 1.0 for ratio in ratios))

    def test_invalid_input(self) -> None:
        """Bad weights, grids, spacings, data and covariance factors are rejected."""
        eye = IdentityOperator(6)
        y = (1.0,) * 6
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 0.0, 1.0, 2, 3)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, -1.0, 2, 3)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2, 2)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 1, 6)
        with self.assertRaises(TypeError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2.0, 3)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, True, 3)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2, 3, spacing=(0.0, 1.0))
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(
                eye,
                y,
                1.0,
                1.0,
                2,
                3,
                spacing=(1.0, 1.0, 1.0),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, (NAN,) * 6, 1.0, 1.0, 2, 3)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2, 3, initial=(NAN,) * 6)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2, 3, step_safety=1.0)
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2, 3, forward_norm_bound=0.0)
        with self.assertRaises(TypeError):
            tgv2_least_squares_2d(eye, y, 1.0, 1.0, 2, 3, noise="identity")
        with self.assertRaises(ValueError):
            tgv2_least_squares_2d(
                eye, y, 1.0, 1.0, 2, 3, noise=DiagonalCovariance((1.0, 2.0))
            )

    def test_budget_is_reported(self) -> None:
        """A tiny iteration budget is never reported as converged."""
        rng = random.Random(1)
        y = tuple(rng.gauss(0, 1) for _ in range(12))
        res = tgv2_least_squares_2d(
            IdentityOperator(12), y, 0.5, 0.5, 3, 4, max_iterations=5
        )
        self.assertFalse(res.converged)
        self.assertEqual(res.stop_reason, "max_iterations")


class IRLSIsotropicTVTests(unittest.TestCase):
    """IRLS for isotropic total variation on a 2D grid."""

    def _problem(self) -> tuple[tuple[float, ...], int, int]:
        """Return a noisy 5 x 5 two-region image."""
        rng = random.Random(12)
        r, c = 5, 5
        y = tuple(
            (1.0 if j >= 2 else 0.0) + 0.15 * rng.gauss(0, 1)
            for _ in range(r)
            for j in range(c)
        )
        return y, r, c

    def test_objective_gap_against_the_isotropic_tv_solver(self) -> None:
        """F(x_eps) lies in [F*, F* + alpha p eps]; the gap shrinks with eps."""
        y, r, c = self._problem()
        alpha = 0.3
        eye = IdentityOperator(r * c)
        tv = total_variation_regularized_least_squares(
            eye,
            y,
            alpha,
            gradient=GridGradientOperator(r, c),
            isotropic=True,
            tolerance_absolute=1e-9,
            tolerance_relative=1e-8,
            max_iterations=60000,
        )
        self.assertTrue(tv.converged)
        gaps = []
        for eps in (1e-2, 1e-3):
            res = irls_regularized_least_squares(
                eye,
                y,
                alpha,
                penalty="tv2d",
                grid=(r, c),
                epsilon=eps,
                inner_tolerance=1e-8,
            )
            self.assertTrue(res.converged, res.stop_reason)
            self.assertEqual(res.penalty_name, "tv2d")
            self.assertEqual(res.components, 2)
            self.assertAlmostEqual(res.gap_bound, alpha * r * c * eps, places=14)
            gap = res.objective - tv.objective
            self.assertGreater(gap, -1e-7)
            self.assertLess(gap, res.gap_bound)
            self.assertLess(res.max_objective_increase, 1e-9)
            self.assertGreaterEqual(res.smoothed_objective, res.objective)
            gaps.append(gap)
        self.assertLess(gaps[1], gaps[0])

    def test_component_grouping_matches_tv2d_and_differs_from_anisotropic(
        self,
    ) -> None:
        """components=2 with an explicit gradient equals tv2d; components=1 differs."""
        y, r, c = self._problem()
        eye = IdentityOperator(r * c)
        grad = GridGradientOperator(r, c)
        common = {"epsilon": 1e-2, "inner_tolerance": 1e-8}
        a = irls_regularized_least_squares(
            eye, y, 0.3, penalty="tv2d", grid=(r, c), **common
        )
        b = irls_regularized_least_squares(
            eye, y, 0.3, penalty_operator=grad, components=2, **common
        )
        self.assertEqual(a.solution, b.solution)
        aniso = irls_regularized_least_squares(
            eye, y, 0.3, penalty_operator=grad, **common
        )
        self.assertEqual(aniso.components, 1)
        self.assertGreater(
            max(abs(p - q) for p, q in zip(a.solution, aniso.solution, strict=True)),
            1e-4,
        )

    def test_constant_image_is_a_fixed_point(self) -> None:
        """A constant image has zero gradient: the minimiser is the image itself."""
        res = irls_regularized_least_squares(
            IdentityOperator(6),
            (2.5,) * 6,
            0.7,
            penalty="tv2d",
            grid=(2, 3),
            epsilon=1e-3,
            inner_tolerance=1e-8,
        )
        self.assertTrue(res.converged)
        self.assertLess(max(abs(v - 2.5) for v in res.solution), 1e-6)

    def test_invalid_input(self) -> None:
        """Grid, spacing and component arguments are validated."""
        eye = IdentityOperator(6)
        y = (1.0,) * 6
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(eye, y, 0.1, penalty="tv2d")
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(eye, y, 0.1, penalty="tv2d", grid=(2, 2))
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(eye, y, 0.1, penalty="tv", grid=(2, 3))
        with self.assertRaises(TypeError):
            irls_regularized_least_squares(
                eye,
                y,
                0.1,
                penalty="tv2d",
                grid=(2.0, 3),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(
                eye, y, 0.1, penalty="tv2d", grid=(2, 3), spacing=(0.0, 1.0)
            )
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(eye, y, 0.1, penalty="l1", components=2)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(eye, y, 0.1, components=0)
        with self.assertRaises(TypeError):
            irls_regularized_least_squares(eye, y, 0.1, components=True)
        with self.assertRaises(ValueError):
            irls_regularized_least_squares(
                eye,
                y,
                0.1,
                penalty_operator=GridGradientOperator(2, 3),
                components=5,
            )


def _whitened_pair(
    a: tuple[tuple[float, ...], ...], y: tuple[float, ...], variances: tuple[float, ...]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return ``(D^-1/2 A, D^-1/2 y)`` for ``D = diag(variances)`` by hand."""
    root = [math.sqrt(v) for v in variances]
    return (
        tuple(tuple(x / s for x in row) for row, s in zip(a, root, strict=True)),
        tuple(v / s for v, s in zip(y, root, strict=True)),
    )


class WhitenedDataTermTests(unittest.TestCase):
    """``noise=`` replaces the data term by ``1/2 ||W (A x - y)||^2``."""

    def setUp(self) -> None:
        """Build one random problem and heteroscedastic variances."""
        self.a, _, self.y = _random_problem(8, 16, 14, 0.05)
        self.variances = tuple(0.2 + 0.35 * (i % 5) for i in range(16))
        self.noise = DiagonalCovariance(self.variances)
        self.aw, self.yw = _whitened_pair(self.a, self.y, self.variances)

    def _assert_close(
        self, p: tuple[float, ...], q: tuple[float, ...], tol: float
    ) -> None:
        """Assert two vectors agree entrywise within ``tol``."""
        self.assertLessEqual(max(abs(u - v) for u, v in zip(p, q, strict=True)), tol)

    def test_irls_equals_the_manually_whitened_problem(self) -> None:
        """IRLS with a covariance factor equals the hand-whitened problem."""
        for penalty in ("l1", "tv"):
            common = {"epsilon": 1e-2, "inner_tolerance": 1e-8}
            with_noise = irls_regularized_least_squares(
                self.a, self.y, 0.05, penalty=penalty, noise=self.noise, **common
            )
            manual = irls_regularized_least_squares(
                self.aw, self.yw, 0.05, penalty=penalty, **common
            )
            self.assertTrue(with_noise.converged and manual.converged)
            self._assert_close(with_noise.solution, manual.solution, 1e-7)
            self.assertAlmostEqual(with_noise.objective, manual.objective, places=9)
            plain = irls_regularized_least_squares(
                self.a, self.y, 0.05, penalty=penalty, **common
            )
            self.assertGreater(
                max(
                    abs(u - v)
                    for u, v in zip(plain.solution, with_noise.solution, strict=True)
                ),
                1e-3,
            )

    def test_irls_2d_with_noise(self) -> None:
        """The isotropic grid penalty also accepts a covariance factor."""
        rng = random.Random(2)
        y = tuple(rng.gauss(0, 1) for _ in range(9))
        var = tuple(0.5 + 0.25 * i for i in range(9))
        eye = _eye(9)
        aw, yw = _whitened_pair(eye, y, var)
        one = irls_regularized_least_squares(
            eye,
            y,
            0.3,
            penalty="tv2d",
            grid=(3, 3),
            noise=DiagonalCovariance(var),
            epsilon=1e-2,
            inner_tolerance=1e-8,
        )
        two = irls_regularized_least_squares(
            aw,
            yw,
            0.3,
            penalty="tv2d",
            grid=(3, 3),
            epsilon=1e-2,
            inner_tolerance=1e-8,
        )
        self._assert_close(one.solution, two.solution, 1e-7)

    def test_huber_equals_the_manually_whitened_problem(self) -> None:
        """Huber-TV with a covariance factor equals the hand-whitened problem."""
        kwargs = {"tolerance_absolute": 1e-11, "tolerance_relative": 1e-10}
        with_noise = huber_total_variation_least_squares(
            self.a, self.y, 0.1, 0.05, noise=self.noise, **kwargs
        )
        manual = huber_total_variation_least_squares(
            self.aw, self.yw, 0.1, 0.05, **kwargs
        )
        self.assertTrue(with_noise.converged and manual.converged)
        self._assert_close(with_noise.solution, manual.solution, 1e-6)
        self.assertAlmostEqual(with_noise.objective, manual.objective, places=8)

    def test_tgv_1d_equals_the_manually_whitened_problem(self) -> None:
        """1D TGV^2 with a covariance factor equals the hand-whitened problem."""
        kwargs = {"tolerance_absolute": 1e-8, "tolerance_relative": 1e-7}
        with_noise = tgv2_least_squares_1d(
            self.a, self.y, 0.1, 0.2, noise=self.noise, **kwargs
        )
        manual = tgv2_least_squares_1d(self.aw, self.yw, 0.1, 0.2, **kwargs)
        self.assertTrue(with_noise.converged and manual.converged)
        self._assert_close(with_noise.solution, manual.solution, 1e-5)
        self.assertAlmostEqual(with_noise.objective, manual.objective, places=7)

    def test_tgv_2d_equals_the_manually_whitened_problem(self) -> None:
        """2D TGV^2 with a covariance factor equals the hand-whitened problem."""
        r, c = 2, 3
        rng = random.Random(15)
        a = tuple(
            tuple(rng.gauss(0.0, 1.0) / math.sqrt(8) for _ in range(r * c))
            for _ in range(8)
        )
        y = tuple(rng.gauss(0.0, 1.0) for _ in range(8))
        var = tuple(0.3 + 0.2 * i for i in range(8))
        aw, yw = _whitened_pair(a, y, var)
        kwargs = {"tolerance_absolute": 1e-6, "tolerance_relative": 1e-5}
        with_noise = tgv2_least_squares_2d(
            a, y, 0.2, 0.3, r, c, noise=DiagonalCovariance(var), **kwargs
        )
        manual = tgv2_least_squares_2d(aw, yw, 0.2, 0.3, r, c, **kwargs)
        self.assertTrue(with_noise.converged and manual.converged)
        self._assert_close(with_noise.solution, manual.solution, 1e-5)
        self.assertAlmostEqual(with_noise.objective, manual.objective, places=7)

    def test_nonconvex_equals_the_manually_whitened_problem(self) -> None:
        """MCP/SCAD with a covariance factor equals the hand-whitened problem."""
        for penalty in ("mcp", "scad"):
            with_noise = nonconvex_sparse_least_squares(
                self.a, self.y, 0.2, penalty=penalty, noise=self.noise, tolerance=1e-7
            )
            manual = nonconvex_sparse_least_squares(
                self.aw, self.yw, 0.2, penalty=penalty, tolerance=1e-7
            )
            self.assertTrue(with_noise.converged and manual.converged)
            self._assert_close(with_noise.solution, manual.solution, 1e-7)
            self.assertAlmostEqual(with_noise.objective, manual.objective, places=9)

    def test_none_and_unit_covariance_reproduce_the_plain_solvers(self) -> None:
        """noise=None is the default path; unit variance agrees to rounding.

        Fixed iteration caps keep the run short and make the comparison a
        statement about identical iterations, not about stopping rules.
        """
        unit = ScaledIdentityCovariance(len(self.y), 1.0)
        a, y = self.a, self.y
        solvers = (
            lambda **k: irls_regularized_least_squares(
                a, y, 0.05, epsilon=1e-2, max_iterations=15, **k
            ),
            lambda **k: huber_total_variation_least_squares(
                a, y, 0.1, 0.05, max_iterations=100, **k
            ),
            lambda **k: tgv2_least_squares_1d(a, y, 0.1, 0.2, max_iterations=100, **k),
            lambda **k: nonconvex_sparse_least_squares(
                a, y, 0.05, max_iterations=100, **k
            ),
        )
        for solve in solvers:
            plain = solve()
            none = solve(noise=None)
            one = solve(noise=unit)
            self.assertEqual(plain.solution, none.solution)
            self.assertEqual(plain.objective, none.objective)
            self._assert_close(plain.solution, one.solution, 1e-6)

    def test_invalid_noise(self) -> None:
        """A non-factor or wrongly sized covariance is rejected by every solver."""
        a, y = self.a, self.y
        wrong = DiagonalCovariance((1.0,) * 5)
        calls = (
            lambda n: irls_regularized_least_squares(a, y, 0.1, noise=n),
            lambda n: huber_total_variation_least_squares(a, y, 0.1, 0.1, noise=n),
            lambda n: tgv2_least_squares_1d(a, y, 0.1, 0.1, noise=n),
            lambda n: nonconvex_sparse_least_squares(a, y, 0.1, noise=n),
        )
        for call in calls:
            with self.assertRaises(TypeError):
                call("identity")
            with self.assertRaises(TypeError):
                call(((1.0,) * 16,) * 16)
            with self.assertRaises(ValueError):
                call(wrong)


if __name__ == "__main__":
    unittest.main()
