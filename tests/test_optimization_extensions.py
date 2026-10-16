"""BFGS stopping rules and Wolfe search, finite differences, bounds, multi-start.

Oracles are closed-form minimizers and derivatives (quadratics, Rosenbrock,
``exp``, ``sqrt``), constrained minima derived by hand, and the defining
inequalities of the strong Wolfe conditions.
"""

from __future__ import annotations

import math
import unittest

from probstat.foundations import RandomStream
from probstat.optimization import (
    BoxTransform,
    central_gradient,
    central_jacobian,
    forward_gradient,
    minimize_bfgs,
    minimize_nelder_mead,
    multi_start,
)
from probstat.optimization.solvers import _strong_wolfe


def quadratic(constant: float):
    """Return ``(x-1)^2 + 10 (y+2)^2 + constant`` and its gradient."""

    def value(x):
        """Return the shifted quadratic."""
        return (x[0] - 1.0) ** 2 + 10.0 * (x[1] + 2.0) ** 2 + constant

    def gradient(x):
        """Return the exact gradient."""
        return (2.0 * (x[0] - 1.0), 20.0 * (x[1] + 2.0))

    return value, gradient


def rosenbrock(x):
    """Return the Rosenbrock function."""
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_gradient(x):
    """Return the Rosenbrock gradient."""
    b = x[1] - x[0] ** 2
    return (-2.0 * (1.0 - x[0]) - 400.0 * x[0] * b, 200.0 * b)


class StoppingRuleTests(unittest.TestCase):
    """Defect 0.7: the stopping rule must not scale with ``|f|``."""

    def test_additive_constant_does_not_stop_early(self) -> None:
        """With ``f + 1e9`` the minimizer still reaches ``(1, -2)``."""
        for constant in (0.0, 1e9, -1e12):
            value, gradient = quadratic(constant)
            with self.subTest(constant=constant):
                result = minimize_bfgs(value, (5.0, 5.0), gradient=gradient)
                self.assertTrue(result.converged, result.message)
                self.assertAlmostEqual(result.x[0], 1.0, 8)
                self.assertAlmostEqual(result.x[1], -2.0, 8)
                self.assertLess(result.gradient_norm, 1e-8)

    def test_finite_difference_resolution_is_reported(self) -> None:
        """Without a gradient the constant limits accuracy, and that is said."""
        value, _ = quadratic(1e9)
        result = minimize_bfgs(value, (5.0, 5.0))
        self.assertGreater(result.iterations, 0)
        self.assertLess(math.dist(result.x, (1.0, -2.0)), 1e-2)
        if result.converged:
            self.assertIn("finite-difference resolution", result.message)

    def test_stall_reports_not_converged(self) -> None:
        """An inconsistent gradient or a noisy objective is a stall."""
        flat = minimize_bfgs(lambda x: 5.0, (0.0, 0.0), gradient=lambda x: (1.0, 0.0))
        self.assertFalse(flat.converged)
        self.assertIn("line search", flat.message)

        def noisy(x):
            """Return a quadratic with a 1e-3 discontinuous ripple."""
            ripple = (x[0] * 1e7) % 1.0
            return (x[0] - 3.0) ** 2 + x[1] ** 2 + 1e-3 * ripple

        result = minimize_bfgs(noisy, (0.0, 1.0))
        self.assertFalse(result.converged)

    def test_iteration_limit_is_not_converged(self) -> None:
        """Hitting ``max_iterations`` is reported honestly."""
        result = minimize_bfgs(rosenbrock, (-1.2, 1.0), max_iterations=3)
        self.assertFalse(result.converged)
        self.assertEqual(result.message, "iteration limit reached")

    def test_rosenbrock_and_ill_scaled_quadratic(self) -> None:
        """Standard benchmarks converge accurately with the scaled start."""
        for gradient in (None, rosenbrock_gradient):
            result = minimize_bfgs(rosenbrock, (-1.2, 1.0), gradient=gradient)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(result.x[0], 1.0, 6)
            self.assertAlmostEqual(result.x[1], 1.0, 6)
        scales = (1.0, 1e2, 1e4)

        def ill(x):
            """Return a quadratic with condition number 1e4."""
            return math.fsum(s * (v - 1.0) ** 2 for s, v in zip(scales, x, strict=True))

        def ill_gradient(x):
            """Return its gradient."""
            return tuple(2.0 * s * (v - 1.0) for s, v in zip(scales, x, strict=True))

        result = minimize_bfgs(ill, (0.0, 0.0, 0.0), gradient=ill_gradient)
        self.assertTrue(result.converged)
        self.assertLess(result.iterations, 25)
        for value in result.x:
            self.assertAlmostEqual(value, 1.0, 9)


class WolfeSearchTests(unittest.TestCase):
    """The accepted step satisfies the strong Wolfe conditions."""

    def test_strong_wolfe_conditions_hold(self) -> None:
        """Sufficient decrease ``c1 = 1e-4`` and curvature ``c2 = 0.9``."""
        cases = (
            (
                lambda x: (x[0] - 3.0) ** 4 + x[0],
                lambda x: (4.0 * (x[0] - 3.0) ** 3 + 1.0,),
            ),
            (lambda x: math.exp(x[0]) - 5.0 * x[0], lambda x: (math.exp(x[0]) - 5.0,)),
        )
        for function, gradient in cases:
            start = (0.0,)
            value, g = function(start), gradient(start)
            direction = (-g[0],)
            slope = g[0] * direction[0]
            for initial in (1e-3, 1.0, 50.0):
                found = _strong_wolfe(
                    function, gradient, start, value, g, direction, slope, initial
                )
                self.assertIsNotNone(found)
                assert found is not None
                alpha, point, new_value, new_g = found
                self.assertLessEqual(new_value, value + 1e-4 * alpha * slope)
                self.assertLessEqual(abs(new_g[0] * direction[0]), -0.9 * slope + 1e-12)


class FiniteDifferenceTests(unittest.TestCase):
    """Defect 0.8: relative steps, one-sided fallback, step override."""

    def test_small_scale_parameters(self) -> None:
        """A parameter on a 1e-6 scale is differentiated accurately."""
        exact = math.exp(2.0) / 1e-6

        def scaled(x):
            """Return ``exp(x / 1e-6)``."""
            return math.exp(x[0] / 1e-6)

        central, calls = central_gradient(scaled, (2e-6,), typical_scale=1e-6)
        self.assertEqual(calls, 2)
        self.assertLess(abs(central[0] / exact - 1.0), 1e-9)
        forward, _ = forward_gradient(scaled, (2e-6,), typical_scale=1e-6)
        self.assertLess(abs(forward[0] / exact - 1.0), 1e-6)
        result = minimize_bfgs(lambda x: (x[0] - 3e-6) ** 2 * 1e12, (1e-6,))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.x[0] / 3e-6, 1.0, 6)

    def test_steps_are_relative_to_the_variable(self) -> None:
        """At ``x = 1e5`` the step is ``eps^(1/3) |x|``; legacy steps are kept."""
        exact = 3.0 * 1e10
        cubic = lambda x: x[0] ** 3  # noqa: E731
        gradient, _ = central_gradient(cubic, (1e5,))
        self.assertLess(abs(gradient[0] / exact - 1.0), 1e-9)
        h = 1e-6 * 1e5
        legacy, _ = central_gradient(cubic, (1e5,), relative_step=1e-6)
        manual = (cubic((1e5 + h,)) - cubic((1e5 - h,))) / (2.0 * h)
        self.assertAlmostEqual(legacy[0] / manual, 1.0, 9)
        stepped, _ = central_gradient(cubic, (2.0,), step=1e-3)
        self.assertAlmostEqual(stepped[0], 12.0 + 1e-6, 9)
        with self.assertRaises(ValueError):
            central_gradient(cubic, (2.0,), step=0.0)

    def test_one_sided_fallback_at_domain_boundary(self) -> None:
        """``sqrt`` and ``log`` next to zero use the side that is defined."""

        def boundary(x):
            """Return ``sqrt(x0) + log(x1)``; undefined for negative inputs."""
            return math.sqrt(x[0]) + math.log(x[1])

        point = (1e-7, 1e-7)
        gradient, _ = central_gradient(boundary, point, typical_scale=1e-7)
        self.assertLess(abs(gradient[0] / (0.5 / math.sqrt(1e-7)) - 1.0), 1e-3)
        self.assertLess(abs(gradient[1] / 1e7 - 1.0), 1e-3)
        jacobian, _ = central_jacobian(
            lambda x: (math.sqrt(x[0]), math.log(x[1])), point, typical_scale=1e-7
        )
        # The scalar sum carries rounding of log(x1) ~ -16 into the x0 quotient;
        # the vector function differentiates sqrt alone and is far more accurate.
        self.assertLess(abs(jacobian[0][0] / (0.5 / math.sqrt(1e-7)) - 1.0), 1e-9)
        self.assertAlmostEqual(jacobian[1][1], gradient[1], 0)
        with self.assertRaises(ValueError):
            central_gradient(lambda x: math.log(-abs(x[0]) - 1.0), (0.0,))


class BoundsAndMultiStartTests(unittest.TestCase):
    """Defect P1.11: box bounds and seeded multi-start."""

    def test_bfgs_and_nelder_mead_respect_bounds(self) -> None:
        """Constrained Rosenbrock minimum ``(1/2, 1/4)`` with ``f = 1/4``."""
        visited: list[tuple[float, ...]] = []

        def recorded(x):
            """Record the evaluation point."""
            visited.append(tuple(x))
            return rosenbrock(x)

        bounds = [(None, 0.5), (0.0, 3.0)]
        for method in (minimize_bfgs, minimize_nelder_mead):
            visited.clear()
            result = method(recorded, (0.2, 0.2), bounds=bounds)
            with self.subTest(method=method.__name__):
                self.assertTrue(result.converged, result.message)
                self.assertAlmostEqual(result.x[0], 0.5, 6)
                self.assertAlmostEqual(result.x[1], 0.25, 5)
                self.assertTrue(
                    all(p[0] <= 0.5 and 0.0 <= p[1] <= 3.0 for p in visited)
                )
        lower = minimize_bfgs(
            lambda x: (x[0] + 2.0) ** 2,
            (5.0,),
            bounds=[(1.0, None)],
            gradient=lambda x: (2.0 * (x[0] + 2.0),),
        )
        self.assertAlmostEqual(lower.x[0], 1.0, 6)
        on_bound = minimize_bfgs(
            lambda x: (x[0] - 0.3) ** 2, (0.0,), bounds=[(0.0, 1.0)]
        )
        self.assertAlmostEqual(on_bound.x[0], 0.3, 6)

    def test_box_transform_round_trip(self) -> None:
        """``to_outer(to_inner(x)) = x`` for interior points of each bound type."""
        box = BoxTransform.from_bounds(
            [(-1.0, 2.0), (3.0, None), (None, -4.0), (None, None)], 4
        )
        point = (0.7, 5.5, -9.0, 1.25)
        back = box.to_outer(box.to_inner(point))
        for a, b in zip(point, back, strict=True):
            self.assertAlmostEqual(a, b, 12)
        with self.assertRaises(ValueError):
            BoxTransform.from_bounds([(1.0, 1.0)], 1)
        with self.assertRaises(ValueError):
            box.to_inner((3.0, 5.5, -9.0, 0.0))

    def test_multi_start_finds_the_global_minimum(self) -> None:
        """Two-well function: the deeper well near ``x = -1.04`` is found."""

        def wells(x):
            """Return ``(x^2 - 1)^2 + 0.3 x``."""
            return (x[0] ** 2 - 1.0) ** 2 + 0.3 * x[0]

        result = multi_start(wells, [(-2.0, 2.0)], starts=6, seed=3)
        self.assertLess(result.best.x[0], 0.0)
        self.assertAlmostEqual(
            4.0 * result.best.x[0] ** 3 - 4.0 * result.best.x[0] + 0.3, 0.0, 6
        )
        self.assertEqual(len(result.runs), 6)
        self.assertGreaterEqual(result.converged_runs, 1)
        again = multi_start(wells, [(-2.0, 2.0)], starts=6, seed=RandomStream(3))
        self.assertEqual(result.starts, again.starts)
        simplex = multi_start(
            wells, [(-2.0, 2.0)], starts=8, seed=3, method="nelder-mead"
        )
        self.assertLess(simplex.best.x[0], 0.0)
        with self.assertRaises(ValueError):
            multi_start(wells, [(None, 2.0)], starts=2, seed=1)


if __name__ == "__main__":
    unittest.main()
