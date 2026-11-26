"""Tests for matrix-free Gauss-Newton and IRGNM inversion (roadmap 09).

The linear case is checked against a stacked Householder QR solve (an
independent dense oracle); the nonlinear cases use exponential decay with a
known optimum, ill-scaled and non-identifiable (product) parameters, a mildly
nonlinear ill-posed problem for the discrepancy-stopped iteration, and a
300-parameter matrix-free model whose evaluation counts must scale with Krylov
iterations, not with ``n``.
"""

from __future__ import annotations

import math
import random
import unittest
from typing import Any

from probstat.inverse.gauss_newton import (
    CallbackForward,
    DifferentiableForward,
    ModelCounts,
    check_derivatives,
    gauss_newton_krylov,
    iteratively_regularized_gauss_newton,
)
from probstat.inverse.test_problems import second_derivative_problem
from probstat.linalg import (
    Convolution1DOperator,
    DenseOperator,
    DiagonalCovariance,
    ScaledIdentityCovariance,
    householder_qr,
    matvec,
    transpose,
)

T_GRID = tuple(0.25 * i for i in range(8))


def _norm(values: Any) -> float:
    """Return the Euclidean norm."""
    return math.sqrt(math.fsum(v * v for v in values))


def _diff(a: Any, b: Any) -> list[float]:
    """Return the entrywise difference ``a - b``."""
    return [x - y for x, y in zip(a, b, strict=True)]


def exponential_model(
    t: tuple[float, ...] = T_GRID, *, derivative: str = "analytic", **kwargs: Any
) -> CallbackForward:
    """Return ``G(a, b)_i = a exp(-b t_i)`` with analytic or FD derivatives."""

    def value(x: Any) -> list[float]:
        return [x[0] * math.exp(-x[1] * ti) for ti in t]

    def jvp(x: Any, v: Any) -> list[float]:
        return [
            v[0] * math.exp(-x[1] * ti) - x[0] * ti * math.exp(-x[1] * ti) * v[1]
            for ti in t
        ]

    def vjp(x: Any, w: Any) -> list[float]:
        da = math.fsum(wi * math.exp(-x[1] * ti) for wi, ti in zip(w, t, strict=True))
        db = math.fsum(
            -wi * x[0] * ti * math.exp(-x[1] * ti) for wi, ti in zip(w, t, strict=True)
        )
        return [da, db]

    if derivative == "analytic":
        return CallbackForward(2, len(t), value, jvp, vjp)
    return CallbackForward(2, len(t), value, derivative="central_difference", **kwargs)


class LinearModelTests(unittest.TestCase):
    """Linear ``G = A x`` must reproduce the stacked-QR Tikhonov solution."""

    def setUp(self) -> None:
        """Build a small full-rank linear problem with a nonzero reference."""
        self.a = (
            (1.0, 0.5, -0.2),
            (0.3, 2.0, 0.1),
            (-0.4, 0.2, 1.5),
            (0.9, -0.7, 0.6),
            (0.1, 0.1, 0.3),
        )
        self.y = (1.0, -2.0, 0.5, 0.25, 3.0)
        self.sd = (0.1, 0.2, 0.5, 1.0, 0.25)
        self.l_matrix = ((1.0, -1.0, 0.0), (0.0, 1.0, -1.0))
        self.x_ref = (0.5, -1.0, 2.0)
        self.lam = 0.7
        at = transpose(self.a)
        self.model = CallbackForward(
            3,
            5,
            lambda x: matvec(self.a, x),
            lambda x, v: matvec(self.a, v),
            lambda x, w: matvec(at, w),
        )

    def oracle(self) -> tuple[float, ...]:
        """Solve ``min ||[W A; lam L] x - [W y; lam L x_ref]||`` by QR."""
        stacked = [
            tuple(a / s for a in row) for row, s in zip(self.a, self.sd, strict=True)
        ] + [tuple(self.lam * c for c in row) for row in self.l_matrix]
        lx = matvec(self.l_matrix, self.x_ref)
        rhs = [yi / s for yi, s in zip(self.y, self.sd, strict=True)] + [
            self.lam * c for c in lx
        ]
        return tuple(householder_qr(tuple(stacked)).solve(tuple(rhs)))

    def test_matches_stacked_qr_in_two_iterations(self) -> None:
        """One step solves the quadratic; the second confirms stationarity."""
        expected = self.oracle()
        for solver in ("lsqr", "cgls"):
            result = gauss_newton_krylov(
                self.model,
                self.y,
                noise=DiagonalCovariance([s * s for s in self.sd]),
                regularization=self.lam,
                penalty=DenseOperator(self.l_matrix),
                reference=self.x_ref,
                inner_solver=solver,  # type: ignore[arg-type]
                inner_tolerance=1e-14,
                inner_max_iterations=30,
            )
            with self.subTest(solver=solver):
                self.assertTrue(result.converged, result.stop_reason)
                self.assertLessEqual(result.iterations, 2)
                error = _norm(_diff(result.estimate, expected))
                self.assertLess(error, 1e-8 * _norm(expected))

    def test_reference_is_not_the_start(self) -> None:
        """Changing ``initial`` does not change a strictly convex problem's answer."""
        kwargs: dict[str, Any] = {
            "noise": DiagonalCovariance([s * s for s in self.sd]),
            "regularization": self.lam,
            "penalty": DenseOperator(self.l_matrix),
            "reference": self.x_ref,
            "inner_tolerance": 1e-14,
        }
        first = gauss_newton_krylov(self.model, self.y, **kwargs)
        second = gauss_newton_krylov(
            self.model, self.y, initial=(5.0, -3.0, 1.0), **kwargs
        )
        self.assertLess(_norm(_diff(first.estimate, second.estimate)), 1e-7)

    def test_objective_terms_and_history(self) -> None:
        """Reported misfit and penalty equal the closed-form definitions."""
        result = gauss_newton_krylov(
            self.model,
            self.y,
            regularization=self.lam,
            reference=self.x_ref,
            inner_tolerance=1e-14,
        )
        x = result.estimate
        r = _diff(matvec(self.a, x), self.y)
        self.assertAlmostEqual(result.misfit, 0.5 * math.fsum(v * v for v in r))
        off = _diff(x, self.x_ref)
        self.assertAlmostEqual(
            result.penalty, 0.5 * self.lam**2 * math.fsum(v * v for v in off)
        )
        self.assertAlmostEqual(result.objective, result.misfit + result.penalty)
        self.assertEqual(result.history[-1].step_length, 0.0)
        self.assertGreaterEqual(result.counts.value, 2)


class ExponentialDecayTests(unittest.TestCase):
    """``y = a exp(-b t)`` with known optimum and analytic derivatives."""

    truth = (2.0, 1.3)

    def data(self) -> tuple[float, ...]:
        """Return noise-free data."""
        return tuple(self.truth[0] * math.exp(-self.truth[1] * t) for t in T_GRID)

    def test_taylor_remainder_is_second_order(self) -> None:
        """The remainder decays with observed order two before roundoff."""
        model = exponential_model()
        check = check_derivatives(model, (1.5, 1.0), (0.3, -0.7), (1.0, -1.0) * 4)
        self.assertTrue(check.taylor_passed)
        self.assertTrue(check.passed)
        self.assertEqual(len(check.orders), len(check.steps) - 1)
        usable = [o for o in check.orders if not math.isnan(o)]
        self.assertTrue(any(1.9 < o < 2.1 for o in usable), check.orders)
        self.assertLess(check.dot_relative_error, 1e-12)

    def test_wrong_jacobian_is_flagged(self) -> None:
        """A JVP that is off by 10 percent shows order one and fails."""
        good = exponential_model()
        bad = CallbackForward(
            2,
            len(T_GRID),
            good.value,
            lambda x, v: [1.1 * c for c in good.jvp(x, v)],
            good.vjp,
        )
        check = check_derivatives(bad, (1.5, 1.0), (0.3, -0.7), (1.0,) * 8)
        self.assertFalse(check.taylor_passed)
        self.assertFalse(check.passed)

    def test_wrong_adjoint_is_flagged(self) -> None:
        """A sign-flipped VJP fails the dot test."""
        good = exponential_model()
        bad = CallbackForward(
            2,
            len(T_GRID),
            good.value,
            good.jvp,
            lambda x, w: [-c for c in good.vjp(x, w)],
        )
        check = check_derivatives(bad, (1.5, 1.0), (0.3, -0.7), (1.0,) * 8)
        self.assertFalse(check.adjoint_passed)

    def test_dot_test_on_random_vectors(self) -> None:
        """``w'(Jv) = (J'w)'v`` to roundoff on seeded random vectors."""
        rng = random.Random(7)
        model = exponential_model()
        x = (1.2, 0.8)
        v = tuple(rng.gauss(0, 1) for _ in range(2))
        w = tuple(rng.gauss(0, 1) for _ in range(8))
        lhs = math.fsum(a * b for a, b in zip(w, model.jvp(x, v), strict=True))
        rhs = math.fsum(a * b for a, b in zip(model.vjp(x, w), v, strict=True))
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_recovers_known_parameters(self) -> None:
        """Noise-free data are fitted to the stopping tolerance."""
        result = gauss_newton_krylov(
            exponential_model(),
            self.data(),
            initial=(1.5, 1.0),
            inner_tolerance=1e-12,
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(abs(result.estimate[0] - 2.0), 1e-6)
        self.assertLess(abs(result.estimate[1] - 1.3), 1e-6)
        objectives = [h.objective for h in result.history]
        for before, after in zip(objectives, objectives[1:], strict=False):
            self.assertLessEqual(after, before * (1 + 1e-12))

    def test_central_difference_matches_analytic(self) -> None:
        """FD JVP and dense FD VJP agree with the analytic actions."""
        analytic = exponential_model()
        fd = exponential_model(derivative="central_difference", allow_dense_fd_vjp=True)
        x, v, w = (1.5, 1.0), (0.3, -0.7), tuple(0.1 * (i + 1) for i in range(8))
        for a, b in zip(analytic.jvp(x, v), fd.jvp(x, v), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-7)
        for a, b in zip(analytic.vjp(x, w), fd.vjp(x, w), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-7)
        counts = fd.counts
        self.assertEqual((counts.jvp, counts.vjp), (1, 1))
        self.assertEqual(counts.derivative_value, 2 + 2 * 2)
        self.assertEqual(counts.value, 0)
        check = check_derivatives(fd, x, v, w)
        self.assertTrue(check.passed, (check.orders, check.dot_relative_error))

    def test_central_difference_solve_agrees(self) -> None:
        """The inversion with FD derivatives matches the analytic one."""
        fd = exponential_model(derivative="central_difference", allow_dense_fd_vjp=True)
        result = gauss_newton_krylov(
            fd, self.data(), initial=(1.5, 1.0), inner_tolerance=1e-12
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(abs(result.estimate[0] - 2.0), 1e-4)
        self.assertLess(abs(result.estimate[1] - 1.3), 1e-4)
        self.assertGreater(result.counts.derivative_value, 0)

    def test_fd_vjp_refused_without_permission(self) -> None:
        """The dense finite-difference VJP must be requested explicitly."""
        fd = exponential_model(derivative="central_difference")
        with self.assertRaises(ValueError):
            fd.vjp((1.0, 1.0), (1.0,) * 8)
        with self.assertRaises(ValueError):
            gauss_newton_krylov(fd, self.data(), initial=(1.5, 1.0))

    def test_ill_scaled_parameters(self) -> None:
        """Parameters seven orders apart in scale are recovered (lambda = 0)."""
        times = tuple(500.0 * i for i in range(9))
        a_true, b_true = 1.0e4, 1.0e-3
        model = exponential_model(times)
        data = tuple(a_true * math.exp(-b_true * t) for t in times)
        result = gauss_newton_krylov(
            model,
            data,
            initial=(5.0e3, 2.0e-3),
            inner_tolerance=1e-13,
            inner_max_iterations=20,
            gradient_tolerance=1e-10,
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(abs(result.estimate[0] - a_true) / a_true, 1e-6)
        self.assertLess(abs(result.estimate[1] - b_true) / b_true, 1e-6)

    def test_max_iterations_is_not_convergence(self) -> None:
        """An exhausted iteration budget is reported as failure."""
        result = gauss_newton_krylov(
            exponential_model(), self.data(), initial=(0.5, 3.0), max_iterations=1
        )
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)

    def test_discrepancy_stop_in_gauss_newton(self) -> None:
        """A requested discrepancy level stops the fixed-lambda iteration."""
        rng = random.Random(3)
        sigma = 0.01
        data = tuple(d + rng.gauss(0, sigma) for d in self.data())
        level = 1.5 * math.sqrt(len(data))
        result = gauss_newton_krylov(
            exponential_model(),
            data,
            noise=ScaledIdentityCovariance(len(data), sigma * sigma),
            initial=(1.5, 1.0),
            discrepancy=level,
        )
        self.assertEqual(result.stop_reason, "discrepancy")
        self.assertLessEqual(result.whitened_residual_norm, level)

    def test_adaptive_forcing_converges(self) -> None:
        """A loosened forcing term still reaches the solution."""
        result = gauss_newton_krylov(
            exponential_model(),
            self.data(),
            initial=(1.5, 1.0),
            forcing="adaptive",
            inner_tolerance=1e-12,
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(abs(result.estimate[1] - 1.3), 1e-5)


class NonuniqueAndFailureTests(unittest.TestCase):
    """Rank deficiency, failed forward evaluations and invalid input."""

    def product_model(self) -> CallbackForward:
        """Return ``G(a, b) = a b`` (non-identifiable)."""
        return CallbackForward(
            2,
            1,
            lambda x: [x[0] * x[1]],
            lambda x, v: [x[1] * v[0] + x[0] * v[1]],
            lambda x, w: [x[1] * w[0], x[0] * w[0]],
        )

    def test_product_model_with_regularization(self) -> None:
        """Rank-one Jacobian: no crash, a stationary point is reported."""
        result = gauss_newton_krylov(
            self.product_model(), (6.0,), regularization=0.1, initial=(1.0, 1.0)
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(result.gradient_norm, 1e-6)
        self.assertAlmostEqual(result.estimate[0], result.estimate[1], places=8)
        self.assertLess(result.objective, 0.5 * 25.0)

    def test_product_model_unregularized_is_reported(self) -> None:
        """With lambda = 0 the iterate fits ``a b = 6``; the outcome is reported."""
        result = gauss_newton_krylov(self.product_model(), (6.0,), initial=(1.0, 2.0))
        self.assertIn(result.stop_reason, {"gradient", "step", "discrepancy"})
        self.assertAlmostEqual(result.estimate[0] * result.estimate[1], 6.0, places=6)

    def test_nonfinite_trial_is_rejected(self) -> None:
        """A NaN forward value at a trial point is a rejected trial."""
        model = CallbackForward(
            1,
            1,
            lambda x: [x[0] if x[0] <= 2.0 else math.nan],
            lambda x, v: [v[0]],
            lambda x, w: [w[0]],
        )
        result = gauss_newton_krylov(model, (3.0,), initial=(0.0,), max_iterations=5)
        first = result.history[0]
        self.assertTrue(math.isnan(first.rejected_objectives[0]))
        self.assertEqual(first.step_length, 0.5)
        self.assertLessEqual(result.estimate[0], 2.0)
        self.assertFalse(result.converged)
        self.assertGreaterEqual(result.counts.value, 3)

    def test_nonfinite_initial_point_raises(self) -> None:
        """A start outside the forward map's domain is invalid input."""
        model = CallbackForward(
            1, 1, lambda x: [math.nan], lambda x, v: [0.0], lambda x, w: [0.0]
        )
        with self.assertRaises(ValueError):
            gauss_newton_krylov(model, (1.0,))

    def test_nonfinite_derivative_is_reported(self) -> None:
        """A NaN Jacobian action ends with reason ``nonfinite``."""
        model = CallbackForward(
            1, 1, lambda x: [x[0]], lambda x, v: [math.nan], lambda x, w: [w[0]]
        )
        result = gauss_newton_krylov(model, (1.0,), initial=(0.0,))
        self.assertEqual(result.stop_reason, "nonfinite")
        self.assertFalse(result.converged)

    def test_invalid_inputs(self) -> None:
        """Wrong shapes, ranges and modes are rejected."""
        data = ExponentialDecayTests().data()
        good = exponential_model()
        with self.assertRaises(ValueError):
            gauss_newton_krylov(good, data[:3])
        with self.assertRaises(ValueError):
            gauss_newton_krylov(good, data, noise=DiagonalCovariance([1.0, 1.0]))
        with self.assertRaises(TypeError):
            gauss_newton_krylov(good, data, noise=[1.0] * 8)
        with self.assertRaises(ValueError):
            gauss_newton_krylov(good, data, regularization=-1.0)
        with self.assertRaises(ValueError):
            gauss_newton_krylov(
                good,
                data,
                inner_solver="lanczos",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            gauss_newton_krylov(good, data, initial=(1.0,))
        with self.assertRaises(ValueError):
            gauss_newton_krylov(good, data, reference=(1.0, math.inf))
        with self.assertRaises(ValueError):
            gauss_newton_krylov(good, data, max_iterations=0)
        with self.assertRaises(TypeError):
            gauss_newton_krylov(object(), data)  # type: ignore[arg-type]

    def test_forward_declaration_is_explicit(self) -> None:
        """No implicit derivative fallback is available."""

        def zero(x: Any) -> list[float]:
            return [0.0, 0.0]

        def zero2(x: Any, v: Any) -> list[float]:
            return [0.0, 0.0]

        with self.assertRaises(ValueError):
            CallbackForward(2, 2, zero)
        with self.assertRaises(ValueError):
            CallbackForward(2, 2, zero, zero2)
        with self.assertRaises(ValueError):
            CallbackForward(2, 2, zero, zero2, zero2, derivative="central_difference")
        with self.assertRaises(ValueError):
            CallbackForward(2, 2, zero, allow_dense_fd_vjp=True)
        with self.assertRaises(ValueError):
            CallbackForward(
                2,
                2,
                zero,
                derivative="adjoint",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            CallbackForward(0, 2, zero, derivative="central_difference")
        jac = CallbackForward(
            2, 1, lambda x: [x[0] * x[1]], jacobian=lambda x: [[x[1], x[0]]]
        )
        self.assertEqual(jac.jvp((2.0, 3.0), (1.0, 1.0)), (5.0,))
        self.assertEqual(jac.vjp((2.0, 3.0), (2.0,)), (6.0, 4.0))
        self.assertIsInstance(jac, DifferentiableForward)
        self.assertIsInstance(jac.counts, ModelCounts)

    def test_check_derivatives_input_errors(self) -> None:
        """Zero direction, bad steps and wrong lengths are rejected."""
        model = exponential_model()
        x, v, w = (1.0, 1.0), (1.0, 0.0), (1.0,) * 8
        with self.assertRaises(ValueError):
            check_derivatives(model, x, (0.0, 0.0), w)
        with self.assertRaises(ValueError):
            check_derivatives(model, x, v, w, steps=(1e-3,))
        with self.assertRaises(ValueError):
            check_derivatives(model, x, v, w, steps=(1e-3, -1e-4))
        with self.assertRaises(ValueError):
            check_derivatives(model, x, v, (1.0,) * 3)


class IRGNMTests(unittest.TestCase):
    """IRGNM on a mildly nonlinear ill-posed problem."""

    def setUp(self) -> None:
        """Build ``G(x) = exp(s A x)`` with ``A`` the deriv2 matrix."""
        problem = second_derivative_problem(16)
        self.matrix = problem.operator
        self.matrix_t = transpose(self.matrix)
        self.scale = 5.0
        self.truth = problem.exact_solution
        s = self.scale

        def value(x: Any) -> list[float]:
            return [math.exp(s * c) for c in matvec(self.matrix, x)]

        def jvp(x: Any, v: Any) -> list[float]:
            ax = matvec(self.matrix, x)
            av = matvec(self.matrix, v)
            return [math.exp(s * a) * s * b for a, b in zip(ax, av, strict=True)]

        def vjp(x: Any, w: Any) -> list[float]:
            ax = matvec(self.matrix, x)
            scaled = [math.exp(s * a) * wi for a, wi in zip(ax, w, strict=True)]
            return [s * c for c in matvec(self.matrix_t, scaled)]

        self.model = CallbackForward(16, 16, value, jvp, vjp)
        rng = random.Random(11)
        self.sigma = 1e-3
        clean = value(tuple(self.truth))
        self.data = tuple(c + rng.gauss(0, self.sigma) for c in clean)
        self.noise = ScaledIdentityCovariance(16, self.sigma**2)
        # Whitened noise norm is about sqrt(m); tau = 1.5.
        self.level = 1.5 * math.sqrt(16)

    def solve(self, **overrides: Any) -> Any:
        """Run IRGNM with the standard setup and overrides."""
        kwargs: dict[str, Any] = {
            "noise": self.noise,
            "initial_regularization": 10.0,
            "decrease": 0.5,
            "discrepancy": self.level,
            "max_iterations": 60,
        }
        kwargs.update(overrides)
        return iteratively_regularized_gauss_newton(self.model, self.data, **kwargs)

    def test_stops_by_discrepancy(self) -> None:
        """The first iterate below ``tau delta`` is returned."""
        result = self.solve()
        self.assertEqual(result.stop_reason, "discrepancy", result.history[-3:])
        self.assertTrue(result.converged)
        self.assertLessEqual(result.whitened_residual_norm, self.level)
        self.assertLessEqual(result.residual_norm, self.level * self.sigma * 1.0000001)
        self.assertGreater(result.history[-2].whitened_residual_norm, self.level)
        lams = [h.regularization for h in result.history]
        for a, b in zip(lams, lams[1:], strict=False):
            self.assertAlmostEqual(b, 0.5 * a, delta=1e-12 * a)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertEqual(result.history[-1].step_norm, 0.0)
        self.assertLess(result.iterations, 60)

    def test_early_stopping_beats_overfitting(self) -> None:
        """Iterating on past the discrepancy index does not reduce the error."""
        stopped = self.solve()
        late = self.solve(discrepancy=1e-12, max_iterations=8)
        self.assertFalse(late.converged)
        self.assertEqual(late.stop_reason, "max_iterations")
        err_stop = _norm(_diff(stopped.estimate, self.truth))
        err_late = _norm(_diff(late.estimate, self.truth))
        self.assertLess(err_stop, err_late)

    def test_reference_dependence(self) -> None:
        """The reference ``x_ref`` changes the estimate."""
        zero = self.solve()
        shifted = self.solve(reference=self.truth)
        self.assertGreater(_norm(_diff(zero.estimate, shifted.estimate)), 1e-3)

    def test_cgls_inner_solver_also_stops(self) -> None:
        """CGLS as the inner solver reaches the discrepancy too."""
        result = self.solve(inner_solver="cgls")
        self.assertEqual(result.stop_reason, "discrepancy")
        self.assertEqual(result.inner_solver, "cgls")

    def test_regularization_floor_and_budget(self) -> None:
        """Floor and iteration cap are reported, not converged."""
        floor = self.solve(discrepancy=1e-9, minimum_regularization=1.0)
        self.assertEqual(floor.stop_reason, "regularization_floor")
        self.assertFalse(floor.converged)
        capped = self.solve(discrepancy=1e-9, max_iterations=2)
        self.assertEqual(capped.stop_reason, "max_iterations")
        self.assertFalse(capped.converged)

    def test_invalid_inputs(self) -> None:
        """The discrepancy is required and ranges are checked."""
        with self.assertRaises(TypeError):
            iteratively_regularized_gauss_newton(  # type: ignore[call-arg]
                self.model, self.data, initial_regularization=1.0
            )
        for bad in (
            {"decrease": 1.0},
            {"decrease": 0.0},
            {"initial_regularization": 0.0},
            {"discrepancy": -1.0},
            {"max_iterations": 0},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.solve(**bad)


class MatrixFreeTests(unittest.TestCase):
    """300 parameters: work scales with Krylov iterations, not with ``n``."""

    def test_counts_do_not_scale_with_n(self) -> None:
        """Elementwise cubic composed with a periodic convolution."""
        n = 300
        conv = Convolution1DOperator((0.2, 0.6, 0.2), n, boundary="periodic")

        def phi(x: Any) -> list[float]:
            return [c + 0.1 * c**3 for c in x]

        def dphi(x: Any) -> list[float]:
            return [1.0 + 0.3 * c * c for c in x]

        model = CallbackForward(
            n,
            n,
            lambda x: conv.matvec(phi(x)),
            lambda x, v: conv.matvec([d * c for d, c in zip(dphi(x), v, strict=True)]),
            lambda x, w: [d * c for d, c in zip(dphi(x), conv.rmatvec(w), strict=True)],
        )
        truth = tuple(math.sin(0.05 * i) + 0.5 * math.cos(0.11 * i) for i in range(n))
        data = tuple(model.value(truth))
        before = model.counts
        self.assertEqual((before.jvp, before.vjp), (0, 0))
        check = check_derivatives(
            model, truth, tuple(math.cos(0.3 * i) for i in range(n)), (1.0,) * n
        )
        self.assertTrue(check.passed, (check.orders, check.dot_relative_error))
        result = gauss_newton_krylov(
            model,
            data,
            regularization=1e-6,
            inner_tolerance=1e-12,
            inner_max_iterations=60,
            gradient_tolerance=1e-9,
        )
        self.assertTrue(result.converged, result.stop_reason)
        error = max(abs(a - b) for a, b in zip(result.estimate, truth, strict=True))
        self.assertLess(error, 1e-4)
        solves = sum(1 for h in result.history if h.inner_stop_reason != "none")
        total = result.total_inner_iterations
        self.assertGreaterEqual(result.counts.jvp, total)
        self.assertLessEqual(result.counts.jvp, total + 2 * solves)
        self.assertLess(result.counts.jvp, n)
        self.assertLess(result.counts.vjp, n)
        self.assertEqual(result.counts.derivative_value, 0)


if __name__ == "__main__":
    unittest.main()
