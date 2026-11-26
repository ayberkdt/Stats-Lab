"""Tests for bounded, robust and variable-projection nonlinear least squares.

Oracles: exhaustive enumeration of the bound-activity patterns of a linear
box-constrained least-squares problem (dense normal equations from
``probstat.linalg.solve_spd``), a golden-section profile search over the one
free variable of a nonlinear bounded problem, the closed-form clamp of a
scalar bounded fit, bisection on the Huber score for the robust location
problem, central differences of the reduced residual for the variable
projection Jacobian, and the existing joint Gauss-Newton solver.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import unittest

from probstat.inverse.gauss_newton import CallbackForward, gauss_newton_krylov
from probstat.inverse.nonlinear_extended import (
    BoundedGaussNewtonResult,
    bounded_gauss_newton,
    robust_gauss_newton,
    robust_loss,
    robust_weight,
    variable_projection,
    variable_projection_reduced_system,
)
from probstat.linalg import (
    ScaledIdentityCovariance,
    dot,
    matvec,
    norm,
    solve_spd,
    transpose,
)

TIMES = tuple(0.25 * i for i in range(21))


def linear_model(matrix) -> CallbackForward:
    """Return ``G(x) = A x`` with analytic ``jvp`` and ``vjp``."""
    at = transpose(matrix)
    return CallbackForward(
        len(matrix[0]),
        len(matrix),
        lambda x: matvec(matrix, x),
        lambda x, v: matvec(matrix, v),
        lambda x, w: matvec(at, w),
    )


def decay_model(times=TIMES) -> CallbackForward:
    """Return ``G_i(a, b) = a exp(-b t_i)`` with analytic derivatives."""

    def value(x):
        return tuple(x[0] * math.exp(-x[1] * t) for t in times)

    def jvp(x, v):
        return tuple(math.exp(-x[1] * t) * (v[0] - x[0] * t * v[1]) for t in times)

    def vjp(x, w):
        pairs = list(zip(w, times, strict=True))
        return (
            sum(wi * math.exp(-x[1] * t) for wi, t in pairs),
            -x[0] * sum(wi * t * math.exp(-x[1] * t) for wi, t in pairs),
        )

    return CallbackForward(2, len(times), value, jvp, vjp)


def box_ls_exhaustive(matrix, y, lower, upper):
    """Solve ``min ||A x - y||`` on a box by enumerating all activity patterns.

    For each assignment of every variable to free / lower / upper, the free
    block solves its normal equations; the candidate must be feasible and
    satisfy the KKT sign conditions. The strictly convex problem has a unique
    KKT point, which is the global minimizer.
    """
    n = len(matrix[0])
    best = None
    for pattern in itertools.product("flu", repeat=n):
        x = [0.0] * n
        for i, s in enumerate(pattern):
            if s == "l":
                x[i] = lower[i]
            elif s == "u":
                x[i] = upper[i]
        free = [i for i, s in enumerate(pattern) if s == "f"]
        if free:
            sub = tuple(tuple(matrix[r][i] for i in free) for r in range(len(matrix)))
            fixed = [
                y[r] - sum(matrix[r][i] * x[i] for i in range(n) if i not in free)
                for r in range(len(matrix))
            ]
            normal = tuple(
                tuple(dot(ca, cb) for cb in transpose(sub)) for ca in transpose(sub)
            )
            rhs = tuple(dot(col, tuple(fixed)) for col in transpose(sub))
            try:
                sol = solve_spd(normal, rhs)
            except (ValueError, ArithmeticError):
                continue
            for j, i in enumerate(free):
                x[i] = sol[j]
        if any(x[i] < lower[i] - 1e-12 or x[i] > upper[i] + 1e-12 for i in range(n)):
            continue
        res = tuple(a - b for a, b in zip(matvec(matrix, tuple(x)), y, strict=True))
        grad = matvec(transpose(matrix), res)
        ok = True
        for i, s in enumerate(pattern):
            if s == "f" and abs(grad[i]) > 1e-9:
                ok = False
            if s == "l" and grad[i] < -1e-9:
                ok = False
            if s == "u" and grad[i] > 1e-9:
                ok = False
        if ok:
            obj = 0.5 * dot(res, res)
            if best is None or obj < best[0]:
                best = (obj, tuple(x))
    return best


A3 = ((1.0, 0.5, 0.2), (0.3, 1.0, 0.1), (0.2, 0.4, 1.0), (1.0, 1.0, 1.0))
Y3 = (3.0, -1.0, 0.5, 2.0)


class BoundedClosedFormTests(unittest.TestCase):
    """Closed forms and exhaustive oracles for the bounded solver."""

    def test_scalar_fit_is_the_clamp(self) -> None:
        """``G(x) = a x`` on ``[l, u]``: the minimizer is ``clamp(y/a)``."""
        a = 2.0
        model = CallbackForward(
            1,
            1,
            lambda x: (a * x[0],),
            lambda x, v: (a * v[0],),
            lambda x, w: (a * w[0],),
        )
        for y, expected in ((1.0, 0.5), (10.0, 1.0), (-4.0, 0.0)):
            result = bounded_gauss_newton(model, (y,), (0.0,), (1.0,), (0.3,))
            self.assertTrue(result.converged, result.stop_reason)
            self.assertAlmostEqual(result.estimate[0], expected, delta=1e-12)
            self.assertAlmostEqual(result.kkt_violation, 0.0, delta=1e-10)
        # y/a above the upper bound: the upper bound is active, multiplier > 0.
        result = bounded_gauss_newton(model, (10.0,), (0.0,), (1.0,), (0.3,))
        self.assertEqual(result.active_upper, (0,))
        self.assertEqual(result.active_lower, ())
        self.assertGreater(result.multipliers[0], 0.0)

    def test_unconstrained_minimizer_violates_bounds(self) -> None:
        """The box solution equals the exhaustive KKT oracle, not the clamp."""
        model = linear_model(A3)
        lower, upper = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        free_fit = gauss_newton_krylov(model, Y3, initial=(0.0, 0.0, 0.0))
        self.assertTrue(
            any(v < 0.0 or v > 1.0 for v in free_fit.estimate), free_fit.estimate
        )
        result = bounded_gauss_newton(model, Y3, lower, upper, (0.5, 0.5, 0.5))
        self.assertTrue(result.converged, result.stop_reason)
        oracle = box_ls_exhaustive(A3, Y3, lower, upper)
        assert oracle is not None
        for a, b in zip(result.estimate, oracle[1], strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)
        self.assertAlmostEqual(result.objective, oracle[0], delta=1e-12)
        # the naive clamp of the unconstrained fit is strictly worse
        clamp = tuple(min(max(v, 0.0), 1.0) for v in free_fit.estimate)
        resid = tuple(a - b for a, b in zip(matvec(A3, clamp), Y3, strict=True))
        self.assertGreater(0.5 * dot(resid, resid), result.objective + 1e-3)
        # KKT report: multipliers nonnegative, partition covers all variables
        self.assertLessEqual(result.projected_gradient_norm, 1e-7)
        self.assertGreaterEqual(min(result.multipliers), -1e-9)
        every = set(result.active_lower) | set(result.active_upper) | set(result.free)
        self.assertEqual(every, {0, 1, 2})
        self.assertIsInstance(result, BoundedGaussNewtonResult)

    def test_random_linear_problems_match_exhaustive(self) -> None:
        """Random 3-variable boxes against the exhaustive oracle (fast tier)."""
        self._sweep(12)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_random_linear_problems_slow_tier(self) -> None:
        """200 random boxes against the exhaustive oracle."""
        self._sweep(200)

    def _sweep(self, count: int) -> None:
        gen = random.Random(20260611)
        agree = 0
        for _ in range(count):
            matrix = tuple(
                tuple(gen.uniform(-1.0, 1.0) for _ in range(3)) for _ in range(5)
            )
            y = tuple(gen.uniform(-3.0, 3.0) for _ in range(5))
            lower = tuple(gen.uniform(-1.0, 0.0) for _ in range(3))
            upper = tuple(gen.uniform(0.1, 1.0) for _ in range(3))
            oracle = box_ls_exhaustive(matrix, y, lower, upper)
            if oracle is None:
                continue
            start = tuple(0.5 * (a + b) for a, b in zip(lower, upper, strict=True))
            result = bounded_gauss_newton(
                linear_model(matrix), y, lower, upper, start, max_iterations=300
            )
            self.assertTrue(result.converged, result.stop_reason)
            self.assertAlmostEqual(result.objective, oracle[0], delta=1e-8)
            agree += 1
        self.assertGreaterEqual(agree, count - 2)


class BoundedNonlinearTests(unittest.TestCase):
    """Nonlinear bounded fits, start on a bound and failure reporting."""

    def _data(self):
        truth = (2.0, 0.7)
        clean = decay_model().value(truth)
        gen = random.Random(5)
        return tuple(c + 0.01 * gen.gauss(0.0, 1.0) for c in clean)

    def test_decay_with_active_amplitude_matches_profile_search(self) -> None:
        """Amplitude capped at 1.5: golden-section search over ``b`` is the oracle."""
        y = self._data()
        model = decay_model()
        result = bounded_gauss_newton(
            model, y, (0.0, 0.0), (1.5, 3.0), (1.0, 0.3), gradient_tolerance=1e-10
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertEqual(result.active_upper, (0,))
        self.assertEqual(result.estimate[0], 1.5)

        def profile(b: float) -> float:
            res = tuple(a - c for a, c in zip(model.value((1.5, b)), y, strict=True))
            return 0.5 * dot(res, res)

        lo, hi = 0.0, 3.0
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(200):
            c = hi - phi * (hi - lo)
            d = lo + phi * (hi - lo)
            if profile(c) < profile(d):
                hi = d
            else:
                lo = c
        self.assertAlmostEqual(result.estimate[1], 0.5 * (lo + hi), delta=1e-6)
        self.assertAlmostEqual(result.objective, profile(0.5 * (lo + hi)), delta=1e-10)
        self.assertGreater(result.multipliers[0], 0.0)

    def test_inactive_bounds_reproduce_gauss_newton(self) -> None:
        """Wide bounds: the bounded solver reaches the unconstrained optimum."""
        y = self._data()
        free_fit = gauss_newton_krylov(
            decay_model(), y, initial=(1.0, 0.3), gradient_tolerance=1e-12
        )
        result = bounded_gauss_newton(
            decay_model(),
            y,
            (-10.0, -10.0),
            (10.0, 10.0),
            (1.0, 0.3),
            gradient_tolerance=1e-12,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.active_lower + result.active_upper, ())
        for a, b in zip(result.estimate, free_fit.estimate, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-7)

    def test_start_on_the_bound_leaves_it_when_gradient_points_inward(self) -> None:
        """A start at ``a = 0`` with a descent direction into the box moves."""
        y = self._data()
        result = bounded_gauss_newton(
            decay_model(), y, (0.0, 0.0), (5.0, 3.0), (0.0, 0.5)
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertGreater(result.estimate[0], 1.0)

    def test_nonunique_product_model_reports_kkt(self) -> None:
        """``G = a b`` (rank-deficient Jacobian at the origin) stays feasible."""
        model = CallbackForward(
            2,
            1,
            lambda x: (x[0] * x[1],),
            lambda x, v: (x[1] * v[0] + x[0] * v[1],),
            lambda x, w: (x[1] * w[0], x[0] * w[0]),
        )
        result = bounded_gauss_newton(
            model, (2.0,), (0.0, 0.0), (1.0, 1.0), (0.5, 0.5), max_iterations=500
        )
        self.assertTrue(result.converged, result.stop_reason)
        # max of a b on the box is 1 at the corner: both bounds active
        self.assertEqual(result.active_upper, (0, 1))
        self.assertAlmostEqual(result.estimate[0] * result.estimate[1], 1.0, delta=1e-9)

    def test_trial_with_nonfinite_forward_is_rejected_not_clipped(self) -> None:
        """A trial in a failed-forward region is recorded as ``nan`` and shrunk."""

        def value(x):
            return (math.nan,) if x[0] > 0.9 else (x[0],)

        model = CallbackForward(1, 1, value, lambda x, v: (v[0],), lambda x, w: (w[0],))
        result = bounded_gauss_newton(model, (5.0,), (0.0,), (2.0,), (0.1,))
        rejected = [v for it in result.history for v in it.rejected_objectives]
        self.assertTrue(any(math.isnan(v) for v in rejected))
        self.assertLess(result.estimate[0], 0.9 + 1e-12)
        # the iterate stalls below the failure region: this is not convergence
        self.assertFalse(result.converged)
        self.assertIn(result.stop_reason, ("line_search_failure", "step"))

    def test_budget_nonfinite_derivative_and_zero_jacobian(self) -> None:
        """Budget, non-finite derivative and a useless direction are reported."""
        y = self._data()
        capped = bounded_gauss_newton(
            decay_model(), y, (0.0, 0.0), (5.0, 3.0), (1.0, 0.3), max_iterations=1
        )
        self.assertEqual(capped.stop_reason, "max_iterations")
        self.assertFalse(capped.converged)
        bad_derivative = CallbackForward(
            1, 1, lambda x: (x[0],), lambda x, v: (v[0],), lambda x, w: (math.nan,)
        )
        result = bounded_gauss_newton(bad_derivative, (1.0,), (0.0,), (2.0,), (0.5,))
        self.assertEqual(result.stop_reason, "nonfinite")
        self.assertFalse(result.converged)
        # derivatives of the wrong sign: the direction increases Phi
        wrong = CallbackForward(
            1, 1, lambda x: (x[0],), lambda x, v: (-v[0],), lambda x, w: (-w[0],)
        )
        result = bounded_gauss_newton(wrong, (1.0,), (0.0,), (2.0,), (0.5,))
        self.assertEqual(result.stop_reason, "line_search_failure")
        self.assertFalse(result.converged)

    def test_validation(self) -> None:
        """Invalid bounds, infeasible start and bad options are rejected."""
        model = decay_model()
        y = self._data()
        with self.assertRaises(ValueError):
            bounded_gauss_newton(model, y, (0.0, 0.0), (1.0, 1.0), (2.0, 0.5))
        with self.assertRaises(ValueError):
            bounded_gauss_newton(model, y, (1.0, 0.0), (1.0, 1.0), (1.0, 0.5))
        with self.assertRaises(ValueError):
            bounded_gauss_newton(model, y, (math.nan, 0.0), (1.0, 1.0), (0.5, 0.5))
        with self.assertRaises(ValueError):
            bounded_gauss_newton(model, y, (0.0,), (1.0, 1.0), (0.5, 0.5))
        with self.assertRaises(TypeError):
            bounded_gauss_newton(model, y, (0.0, 0.0), (1.0, 1.0), "ab")
        with self.assertRaises(ValueError):
            bounded_gauss_newton(
                model, y, (0.0, 0.0), (1.0, 1.0), (0.5, 0.5), armijo=1.5
            )
        with self.assertRaises(TypeError):
            bounded_gauss_newton(object(), y, (0.0, 0.0), (1.0, 1.0), (0.5, 0.5))
        nonfinite = CallbackForward(
            1, 1, lambda x: (math.nan,), lambda x, v: v, lambda x, w: w
        )
        with self.assertRaises(ValueError):
            bounded_gauss_newton(nonfinite, (1.0,), (0.0,), (1.0,), (0.5,))
        # infinite bounds are allowed and behave as unconstrained
        free = bounded_gauss_newton(
            linear_model(A3), Y3, (-math.inf,) * 3, (math.inf,) * 3, (0.0, 0.0, 0.0)
        )
        self.assertTrue(free.converged)

    def test_matrix_free_counts_for_200_parameters(self) -> None:
        """200 box-constrained parameters: derivative actions, no Jacobian."""
        n = 200
        c = 0.1
        gen = random.Random(11)
        truth = tuple(gen.uniform(-1.0, 1.0) for _ in range(n))

        def value(x):
            return tuple(x[i] + c * x[i] * x[(i + 1) % n] for i in range(n))

        def jvp(x, v):
            return tuple(
                v[i] + c * (v[i] * x[(i + 1) % n] + x[i] * v[(i + 1) % n])
                for i in range(n)
            )

        def vjp(x, w):
            return tuple(
                w[j] * (1.0 + c * x[(j + 1) % n]) + w[j - 1] * c * x[j - 1]
                for j in range(n)
            )

        model = CallbackForward(n, n, value, jvp, vjp)
        y = value(truth)
        result = bounded_gauss_newton(
            model, y, (-0.5,) * n, (0.5,) * n, (0.0,) * n, gradient_tolerance=1e-8
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertGreater(len(result.active_lower) + len(result.active_upper), 0)
        # n = 200 columns would need 200 evaluations per Jacobian
        self.assertLess(result.counts.jvp, n * result.iterations)
        self.assertLessEqual(result.kkt_violation, 1e-6)
        print(
            f"\n[bounded n=200] iterations {result.iterations}, counts "
            f"{result.counts}, active {len(result.active_lower)}+"
            f"{len(result.active_upper)}"
        )


def huber_location(values, c):
    """Return the Huber location estimate by bisection on the score."""
    lo, hi = min(values), max(values)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        score = sum(max(-c, min(c, v - mid)) for v in values)
        if score > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def location_model(m: int) -> CallbackForward:
    """Return ``G(x) = (x, ..., x)`` of length ``m``."""
    return CallbackForward(
        1, m, lambda x: (x[0],) * m, lambda x, v: (v[0],) * m, lambda x, w: (sum(w),)
    )


class RobustTests(unittest.TestCase):
    """Huber and Cauchy iteratively reweighted Gauss-Newton."""

    def setUp(self) -> None:
        """Build clean data, noise and contaminated data with three outliers."""
        gen = random.Random(7)
        self.truth = (2.0, 0.7)
        clean = decay_model().value(self.truth)
        self.sigma = 0.02
        y = [c + self.sigma * gen.gauss(0.0, 1.0) for c in clean]
        self.outliers = {3: 4.0, 10: -3.0, 17: 5.0}
        self.clean_y = tuple(y)
        for i, shift in self.outliers.items():
            y[i] += shift
        self.y = tuple(y)
        self.noise = ScaledIdentityCovariance(len(TIMES), self.sigma**2)

    def test_loss_and_weight_definitions(self) -> None:
        """``rho`` is continuous at the threshold and ``rho' = w e``."""
        for loss in ("huber", "cauchy"):
            c = 1.3
            for e in (-4.0, -1.3, -0.4, 0.2, 1.3, 2.9):
                h = 1e-6
                slope = (robust_loss(loss, c, e + h) - robust_loss(loss, c, e - h)) / (
                    2.0 * h
                )
                self.assertAlmostEqual(slope, robust_weight(loss, c, e) * e, delta=1e-6)
            self.assertEqual(robust_weight(loss, c, 0.0), 1.0)
        self.assertAlmostEqual(
            robust_loss("huber", 1.0, 1.0 - 1e-12),
            robust_loss("huber", 1.0, 1.0 + 1e-12),
            delta=1e-9,
        )
        with self.assertRaises(ValueError):
            robust_loss("tukey", 1.0, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            robust_weight("huber", 0.0, 1.0)

    def test_huber_location_matches_bisection_oracle(self) -> None:
        """Scalar location with outliers: IRLS-GN equals the score root."""
        gen = random.Random(3)
        values = [gen.gauss(1.0, 1.0) for _ in range(30)] + [15.0, 18.0, 22.0]
        result = robust_gauss_newton(
            location_model(len(values)),
            tuple(values),
            loss="huber",
            threshold=1.345,
            initial=(0.0,),
            gradient_tolerance=1e-12,
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertAlmostEqual(
            result.estimate[0], huber_location(values, 1.345), delta=1e-8
        )
        plain = sum(values) / len(values)
        self.assertGreater(plain - result.estimate[0], 1.0)

    def test_huge_threshold_reduces_to_plain_gauss_newton(self) -> None:
        """With every ``|e| <= c`` all weights are one: the plain GN optimum."""
        plain = gauss_newton_krylov(
            decay_model(),
            self.y,
            initial=(1.0, 0.3),
            noise=self.noise,
            gradient_tolerance=1e-9,
        )
        huge = robust_gauss_newton(
            decay_model(),
            self.y,
            loss="huber",
            threshold=1e12,
            noise=self.noise,
            initial=(1.0, 0.3),
            gradient_tolerance=1e-9,
        )
        self.assertTrue(huge.converged, huge.stop_reason)
        self.assertEqual(huge.downweighted, 0)
        self.assertEqual(set(huge.weights), {1.0})
        for a, b in zip(huge.estimate, plain.estimate, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6)
        self.assertAlmostEqual(
            huge.objective, 0.5 * plain.whitened_residual_norm**2, delta=1e-6
        )

    def test_gross_outliers_resisted_with_measured_numbers(self) -> None:
        """Huber and Cauchy stay near the truth while plain GN is pulled away."""
        start = (1.0, 0.3)
        plain = gauss_newton_krylov(
            decay_model(), self.y, initial=start, noise=self.noise
        )
        clean = gauss_newton_krylov(
            decay_model(), self.clean_y, initial=start, noise=self.noise
        )
        errors = {"plain GN": math.dist(plain.estimate, self.truth)}
        reference = math.dist(clean.estimate, self.truth)
        for loss in ("huber", "cauchy"):
            result = robust_gauss_newton(
                decay_model(),
                self.y,
                loss=loss,
                noise=self.noise,
                initial=start,
            )
            self.assertTrue(result.converged, (loss, result.stop_reason))
            errors[loss] = math.dist(result.estimate, self.truth)
            # the three gross outliers receive the smallest weights
            order = sorted(range(len(TIMES)), key=lambda i: result.weights[i])
            self.assertEqual(set(order[:3]), set(self.outliers), loss)
            self.assertLess(errors[loss], 0.1 * errors["plain GN"], loss)
            self.assertLess(errors[loss], 0.02, loss)
        self.assertGreater(errors["plain GN"], 0.3)
        print(
            f"\n[outliers] param error plain GN {errors['plain GN']:.4f}, "
            f"Huber {errors['huber']:.4f}, Cauchy {errors['cauchy']:.4f}; "
            f"fit to the outlier-free data {reference:.4f}"
        )

    def test_breakdown_with_majority_outliers(self) -> None:
        """Failure case: 60% contamination drags the Huber estimate away."""
        gen = random.Random(9)
        values = [gen.gauss(0.0, 0.1) for _ in range(12)] + [
            10.0 + gen.gauss(0.0, 0.1) for _ in range(18)
        ]
        result = robust_gauss_newton(
            location_model(len(values)),
            tuple(values),
            loss="huber",
            threshold=1.345,
            initial=(0.0,),
        )
        self.assertTrue(result.converged)
        # the estimate sits near the contaminating cluster, not the clean one
        self.assertGreater(result.estimate[0], 5.0)

    def test_failures_and_validation(self) -> None:
        """Budget, failed forward map, nonfinite derivative, bad options."""
        capped = robust_gauss_newton(
            decay_model(),
            self.y,
            noise=self.noise,
            initial=(1.0, 0.3),
            max_iterations=1,
        )
        self.assertEqual(capped.stop_reason, "max_iterations")
        self.assertFalse(capped.converged)
        bad = CallbackForward(
            1, 1, lambda x: (x[0],), lambda x, v: v, lambda x, w: (math.nan,)
        )
        self.assertEqual(
            robust_gauss_newton(bad, (1.0,), initial=(0.0,)).stop_reason, "nonfinite"
        )
        # derivatives of the wrong sign: the "descent" direction increases Phi
        wrong = CallbackForward(
            1, 1, lambda x: (x[0],), lambda x, v: (-v[0],), lambda x, w: (-w[0],)
        )
        self.assertEqual(
            robust_gauss_newton(wrong, (1.0,), initial=(0.0,)).stop_reason,
            "line_search_failure",
        )
        nan_model = CallbackForward(
            1, 1, lambda x: (math.nan,), lambda x, v: v, lambda x, w: w
        )
        with self.assertRaises(ValueError):
            robust_gauss_newton(nan_model, (1.0,), initial=(0.0,))
        with self.assertRaises(ValueError):
            robust_gauss_newton(decay_model(), self.y, loss="tukey")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            robust_gauss_newton(decay_model(), self.y, threshold=0.0)
        with self.assertRaises(ValueError):
            robust_gauss_newton(decay_model(), self.y[:-1])
        with self.assertRaises(TypeError):
            robust_gauss_newton(object(), self.y)  # type: ignore[arg-type]

        # a rejected trial is recorded: step into a failed-forward region
        def value(x):
            return (math.nan,) if x[0] > 0.9 else (x[0],)

        guarded = CallbackForward(
            1, 1, value, lambda x, v: (v[0],), lambda x, w: (w[0],)
        )
        result = robust_gauss_newton(guarded, (5.0,), initial=(0.1,))
        rejected = [v for it in result.history for v in it.rejected_objectives]
        self.assertTrue(any(math.isnan(v) for v in rejected))
        self.assertFalse(result.converged)


def exp_basis(times):
    """Return ``Phi(alpha) = [exp(-a_j t)]`` and its derivatives."""

    def basis(a):
        try:
            return tuple(tuple(math.exp(-aj * t) for aj in a) for t in times)
        except OverflowError:
            return tuple((math.nan,) * len(a) for _ in times)

    def derivative(a):
        out = []
        for j in range(len(a)):
            out.append(
                tuple(
                    tuple(
                        -t * math.exp(-aj * t) if i == j else 0.0
                        for i, aj in enumerate(a)
                    )
                    for t in times
                )
            )
        return tuple(out)

    return basis, derivative


class VariableProjectionTests(unittest.TestCase):
    """Golub-Pereyra variable projection for a sum of exponentials."""

    T = tuple(0.2 * i for i in range(30))

    def _data(self, noise: float):
        gen = random.Random(3)
        return tuple(
            2.0 * math.exp(-0.5 * t)
            + 1.0 * math.exp(-2.0 * t)
            + noise * gen.gauss(0.0, 1.0)
            for t in self.T
        )

    def test_exact_small_case_recovers_coefficients(self) -> None:
        """Noise-free data: ``alpha`` and the linear ``c`` are recovered."""
        basis, deriv = exp_basis(self.T)
        result = variable_projection(
            basis,
            self._data(0.0),
            (0.8, 1.5),
            basis_derivative=deriv,
            gradient_tolerance=1e-12,
        )
        self.assertTrue(result.converged, result.stop_reason)
        alpha = sorted(result.alpha)
        self.assertAlmostEqual(alpha[0], 0.5, delta=1e-7)
        self.assertAlmostEqual(alpha[1], 2.0, delta=1e-6)
        by_alpha = dict(zip(result.alpha, result.coefficients, strict=True))
        self.assertAlmostEqual(by_alpha[min(result.alpha)], 2.0, delta=1e-6)
        self.assertAlmostEqual(by_alpha[max(result.alpha)], 1.0, delta=1e-6)
        self.assertLess(result.residual_norm, 1e-6)

    def test_reduced_jacobian_matches_finite_differences(self) -> None:
        """Full Jacobian = central differences of ``r``; Kaufman misses a term."""
        basis, deriv = exp_basis(self.T)
        for noise, alpha in ((0.0, (0.5, 2.0)), (0.05, (0.7, 1.7))):
            y = self._data(noise)
            full = variable_projection_reduced_system(
                basis, y, alpha, basis_derivative=deriv
            )
            kauf = variable_projection_reduced_system(
                basis, y, alpha, basis_derivative=deriv, jacobian="kaufman"
            )
            fd_cols = []
            for j in range(2):
                h = 1e-6
                up = list(alpha)
                dn = list(alpha)
                up[j] += h
                dn[j] -= h
                r_up = variable_projection_reduced_system(
                    basis, y, tuple(up), basis_derivative=deriv
                ).residual
                r_dn = variable_projection_reduced_system(
                    basis, y, tuple(dn), basis_derivative=deriv
                ).residual
                fd_cols.append(
                    tuple((a - b) / (2 * h) for a, b in zip(r_up, r_dn, strict=True))
                )
            scale = max(abs(v) for row in full.jacobian for v in row)
            err_full = max(
                abs(full.jacobian[i][j] - fd_cols[j][i])
                for i in range(len(self.T))
                for j in range(2)
            )
            err_kauf = max(
                abs(kauf.jacobian[i][j] - fd_cols[j][i])
                for i in range(len(self.T))
                for j in range(2)
            )
            self.assertLess(err_full, 1e-6 * scale, noise)
            if noise == 0.0:
                self.assertLess(err_kauf, 1e-3 * scale)
            else:
                self.assertGreater(err_kauf, 1e-3 * scale)
            print(
                f"\n[varpro Jacobian, noise {noise}] max error: full {err_full:.2e}, "
                f"Kaufman {err_kauf:.2e} (scale {scale:.2f})"
            )

    def test_central_difference_matches_analytic(self) -> None:
        """Explicit finite-difference derivatives reach the same optimum."""
        basis, deriv = exp_basis(self.T)
        y = self._data(0.01)
        exact = variable_projection(basis, y, (0.8, 1.5), basis_derivative=deriv)
        fd = variable_projection(basis, y, (0.8, 1.5), derivative="central_difference")
        self.assertTrue(exact.converged and fd.converged, (exact, fd))
        for a, b in zip(exact.alpha, fd.alpha, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-5)
        self.assertEqual(fd.derivative_evaluations, 0)
        self.assertGreater(fd.basis_evaluations, exact.basis_evaluations)

    def test_agrees_with_joint_gauss_newton_and_measured_comparison(self) -> None:
        """Same optimum as the joint solve; report iterations and basins."""
        basis, deriv = exp_basis(self.T)
        y = self._data(0.01)
        t_values = self.T

        def value(x):
            try:
                return tuple(
                    sum(x[2 + j] * math.exp(-x[j] * t) for j in range(2))
                    for t in t_values
                )
            except OverflowError:
                return (math.nan,) * len(t_values)

        def jvp(x, v):
            return tuple(
                sum(
                    v[2 + j] * math.exp(-x[j] * t)
                    - x[2 + j] * t * math.exp(-x[j] * t) * v[j]
                    for j in range(2)
                )
                for t in t_values
            )

        def vjp(x, w):
            out = [0.0] * 4
            for wi, t in zip(w, t_values, strict=True):
                for j in range(2):
                    out[j] += wi * (-x[2 + j] * t * math.exp(-x[j] * t))
                    out[2 + j] += wi * math.exp(-x[j] * t)
            return tuple(out)

        joint = CallbackForward(4, len(t_values), value, jvp, vjp)
        optimum = variable_projection(basis, y, (0.8, 1.5), basis_derivative=deriv)
        stats = {"vp": [0, 0], "kaufman": [0, 0], "joint": [0, 0]}
        for a1, a2 in itertools.product((0.1, 0.8, 3.0), (0.2, 1.2, 5.0)):
            if a1 == a2:
                continue
            runs = {
                "vp": variable_projection(basis, y, (a1, a2), basis_derivative=deriv),
                "kaufman": variable_projection(
                    basis, y, (a1, a2), basis_derivative=deriv, jacobian="kaufman"
                ),
            }
            c0 = variable_projection_reduced_system(
                basis, y, (a1, a2), basis_derivative=deriv
            ).coefficients
            gj = gauss_newton_krylov(
                joint, y, initial=(a1, a2, *c0), max_iterations=200
            )
            for name, res in runs.items():
                if res.converged and abs(res.objective - optimum.objective) < 1e-9:
                    stats[name][0] += 1
                    stats[name][1] += res.iterations
            if gj.converged and abs(gj.objective - optimum.objective) < 1e-9:
                stats["joint"][0] += 1
                stats["joint"][1] += gj.iterations
        # variable projection reaches the optimum from every start in the grid
        self.assertEqual(stats["vp"][0], 9 - 0, stats)
        self.assertGreaterEqual(stats["vp"][0], stats["joint"][0])
        print(
            "\n[varpro vs joint GN, 9 starts] (successes, total iterations): "
            + ", ".join(f"{k} {tuple(v)}" for k, v in stats.items())
        )

    def test_rank_deficiency_is_reported_not_repaired(self) -> None:
        """Identical columns at the start raise; a useless parameter stops."""
        basis, deriv = exp_basis(self.T)
        y = self._data(0.01)
        with self.assertRaises(ValueError):
            variable_projection(basis, y, (1.0, 1.0), basis_derivative=deriv)

        def basis_one(a):
            return tuple((math.exp(-a[0] * t), 1.0) for t in self.T)

        def deriv_one(a):
            zero = tuple((0.0, 0.0) for _ in self.T)
            return (tuple((-t * math.exp(-a[0] * t), 0.0) for t in self.T), zero)

        result = variable_projection(
            basis_one, y, (0.8, 0.3), basis_derivative=deriv_one
        )
        self.assertEqual(result.stop_reason, "rank_deficient")
        self.assertFalse(result.converged)

    def test_rank_floor_rejects_trials(self) -> None:
        """A rank floor between the start and the optimum rejects trial points."""
        basis, deriv = exp_basis(self.T)
        y = self._data(0.0)

        def ratio(alpha):
            return variable_projection_reduced_system(
                basis, y, alpha, basis_derivative=deriv
            ).singular_ratio

        start, optimum = ratio((0.5, 5.0)), ratio((0.5, 2.0))
        floor = math.sqrt(start * optimum)
        self.assertLess(optimum, floor)
        self.assertLess(floor, start)
        result = variable_projection(
            basis, y, (0.5, 5.0), basis_derivative=deriv, rank_tolerance=floor
        )
        self.assertFalse(result.converged)
        self.assertGreater(result.rank_rejections, 0)
        # backtracking can only stop short of the floor: stagnation, not success
        self.assertIn(result.stop_reason, ("step", "line_search_failure"))
        self.assertGreater(result.min_singular_ratio, floor)

    def test_option_and_shape_validation(self) -> None:
        """Contradictory options and malformed callbacks are rejected."""
        basis, deriv = exp_basis(self.T)
        y = self._data(0.01)
        with self.assertRaises(ValueError):
            variable_projection(basis, y, (0.8, 1.5))
        with self.assertRaises(ValueError):
            variable_projection(
                basis,
                y,
                (0.8, 1.5),
                basis_derivative=deriv,
                derivative="central_difference",
            )
        with self.assertRaises(ValueError):
            variable_projection(
                basis,
                y,
                (0.8, 1.5),
                basis_derivative=deriv,
                jacobian="gauss",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            variable_projection(basis, y[:-1], (0.8, 1.5), basis_derivative=deriv)
        with self.assertRaises(TypeError):
            variable_projection(basis, y, "ab", basis_derivative=deriv)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            variable_projection(basis, y, (), basis_derivative=deriv)
        with self.assertRaises(ValueError):
            variable_projection(
                basis, y, (0.8, 1.5), basis_derivative=deriv, max_iterations=0
            )
        capped = variable_projection(
            basis, y, (0.8, 1.5), basis_derivative=deriv, max_iterations=1
        )
        self.assertEqual(capped.stop_reason, "max_iterations")
        self.assertFalse(capped.converged)
        self.assertGreater(norm(capped.alpha), 0.0)


if __name__ == "__main__":
    unittest.main()
