"""Tests for forward-mode automatic differentiation (ADR-0065)."""

from __future__ import annotations

import cmath
import math
import unittest

from probstat.numerics import (
    AutodiffUnsupportedError,
    Dual,
    GradientEvaluator,
    admath,
    derivative,
    gradient,
    hessian,
    jacobian,
    value_and_gradient,
    value_gradient_and_hessian,
)

M = admath


def kitchen_sink(v):
    """Use nearly every admath function (the mpmath oracle mirrors it)."""
    x, y, z = v
    return (
        M.exp(x) * M.sin(y)
        + M.log1p(x * x) * M.cos(z)
        + M.sqrt(1 + y * y) / (2 + M.tanh(z))
        + M.atan(x * y) * M.erf(z)
        + M.lgamma(x + 3) * M.expm1(y / 4)
        + M.erfc(x - y) * M.hypot(y, z)
        + (x + 2) ** (y + 1)
        + M.log(2 + z * z) * M.atan2(y, 2 + x)
        + M.asin(x / 3) * M.acos(z / 4)
        + M.sinh(x / 2) * M.cosh(y / 3)
        + M.tan(z / 5)
        + M.log10(3 + x)
        + M.log2(4 + y)
    )


# mpmath 1.4 at 50 digits, mpmath.diff of the same expression at
# (0.3, -0.7, 1.1) (scratch script oracle_ad.py of workstream W8).
SINK_VALUE = 3.1577982041359346021
SINK_GRADIENT = (
    -0.52472702713205756797,
    3.7032970583986813399,
    -0.088177563611480138102,
)
SINK_HESSIAN = (
    (0.86711570672267545593, 1.6522367806563808719, -1.0704803351864018295),
    (1.6522367806563808719, 2.7586369841835339254, 0.80051974788529709381),
    (-1.0704803351864018295, 0.80051974788529709381, 0.22430287788736382103),
)


class OracleTests(unittest.TestCase):
    """Derivatives agree with independent references."""

    def test_gradient_and_hessian_match_mpmath(self) -> None:
        """Value, gradient and Hessian equal 50-digit mpmath.diff results."""
        value, grad, hess = value_gradient_and_hessian(kitchen_sink, (0.3, -0.7, 1.1))
        self.assertAlmostEqual(value, SINK_VALUE, delta=1e-14)
        for got, want in zip(grad, SINK_GRADIENT, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-13)
        for row, want_row in zip(hess, SINK_HESSIAN, strict=True):
            for got, want in zip(row, want_row, strict=True):
                self.assertAlmostEqual(got, want, delta=1e-12)
        first = value_and_gradient(kitchen_sink, (0.3, -0.7, 1.1))
        self.assertEqual(first[1], grad)
        self.assertEqual(hessian(kitchen_sink, (0.3, -0.7, 1.1)), hess)

    def test_gradient_matches_complex_step(self) -> None:
        """For an analytic function, Im f(x + i h) / h with h = 1e-30 agrees."""

        def real_version(v):
            """Dual-capable function."""
            x, y = v
            return M.exp(x * y) * M.sin(x) / (1 + y * y) + M.log(2 + M.cos(y)) * x**3

        def complex_version(x, y):
            """Evaluate the same function on complex numbers."""
            return (
                cmath.exp(x * y) * cmath.sin(x) / (1 + y * y)
                + cmath.log(2 + cmath.cos(y)) * x**3
            )

        point = (0.8, -1.3)
        h = 1e-30
        reference = (
            complex_version(point[0] + 1j * h, point[1]).imag / h,
            complex_version(point[0], point[1] + 1j * h).imag / h,
        )
        for got, want in zip(gradient(real_version, point), reference, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-14 * max(1.0, abs(want)))

    def test_closed_form_derivatives_of_each_function(self) -> None:
        """Each admath function has the textbook derivative."""
        x = 0.37
        cases = (
            (M.exp, math.exp(x)),
            (M.expm1, math.exp(x)),
            (M.log, 1 / x),
            (M.log1p, 1 / (1 + x)),
            (M.log2, 1 / (x * math.log(2))),
            (M.log10, 1 / (x * math.log(10))),
            (M.sqrt, 0.5 / math.sqrt(x)),
            (M.sin, math.cos(x)),
            (M.cos, -math.sin(x)),
            (M.tan, 1 / math.cos(x) ** 2),
            (M.asin, 1 / math.sqrt(1 - x * x)),
            (M.acos, -1 / math.sqrt(1 - x * x)),
            (M.atan, 1 / (1 + x * x)),
            (M.sinh, math.cosh(x)),
            (M.cosh, math.sinh(x)),
            (M.tanh, 1 / math.cosh(x) ** 2),
            (M.erf, 2 / math.sqrt(math.pi) * math.exp(-x * x)),
            (M.erfc, -2 / math.sqrt(math.pi) * math.exp(-x * x)),
            # psi(x) by the reflection-free recurrence psi(x) = psi(x+1) - 1/x
            # checked against Euler's constant: psi(1) = -gamma.
            (M.fabs, 1.0),
        )
        for function, expected in cases:
            with self.subTest(function=function.__name__):
                self.assertAlmostEqual(derivative(function, x), expected, delta=1e-14)
        self.assertAlmostEqual(
            derivative(M.lgamma, 1.0), -0.5772156649015329, delta=1e-13
        )
        # psi'(1) = pi^2 / 6.
        self.assertAlmostEqual(derivative(M.digamma, 1.0), math.pi**2 / 6, delta=1e-12)
        self.assertAlmostEqual(
            derivative(lambda t: M.log(t, 3.0), 2.0), 1 / (2 * math.log(3))
        )
        self.assertAlmostEqual(derivative(lambda t: M.pow(t, 2.5), 2.0), 2.5 * 2.0**1.5)

    def test_jacobian_of_polar_map(self) -> None:
        """The Jacobian of (r cos t, r sin t) has determinant r."""
        r, t = 2.0, 0.6
        rows = jacobian(lambda v: (v[0] * M.cos(v[1]), v[0] * M.sin(v[1])), (r, t))
        self.assertAlmostEqual(rows[0][0], math.cos(t))
        self.assertAlmostEqual(rows[0][1], -r * math.sin(t))
        self.assertAlmostEqual(rows[1][0], math.sin(t))
        self.assertAlmostEqual(rows[1][1], r * math.cos(t))
        det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        self.assertAlmostEqual(det, r)

    def test_hessian_of_quadratic_form_is_exact(self) -> None:
        """A quadratic form ``x' A x / 2`` has Hessian ``A`` exactly."""
        a = ((2.0, -0.5, 0.25), (-0.5, 3.0, 1.0), (0.25, 1.0, 4.0))

        def form(v):
            """Return x' A x / 2."""
            return 0.5 * sum(v[i] * a[i][j] * v[j] for i in range(3) for j in range(3))

        self.assertEqual(hessian(form, (0.4, -1.2, 2.5)), a)


class ArithmeticTests(unittest.TestCase):
    """Operator semantics of Dual."""

    def test_operators_and_reflected_operators(self) -> None:
        """+, -, *, /, ** with floats on either side follow the chain rule."""
        x = Dual(3.0, (1.0,))
        cases = (
            (x + 2, 5.0, 1.0),
            (2 + x, 5.0, 1.0),
            (x - 2, 1.0, 1.0),
            (2 - x, -1.0, -1.0),
            (x * 2, 6.0, 2.0),
            (2 * x, 6.0, 2.0),
            (x / 2, 1.5, 0.5),
            (6 / x, 2.0, -6 / 9),
            (x**2, 9.0, 6.0),
            (2**x, 8.0, 8 * math.log(2)),
            (x**x, 27.0, 27 * (math.log(3) + 1)),
            (-x, -3.0, -1.0),
            (+x, 3.0, 1.0),
            (abs(-x), 3.0, 1.0),
        )
        for result, value, slope in cases:
            self.assertAlmostEqual(result.value, value)
            self.assertAlmostEqual(result.tangent[0], slope)

    def test_negative_base_integer_power_and_zero_cases(self) -> None:
        """(-2)^3 works; 0^y has derivative 0; 0^0.5 slope raises loudly."""
        self.assertAlmostEqual(derivative(lambda t: t**3, -2.0), 12.0)
        self.assertEqual(derivative(lambda t: 0.0**t, 2.0), 0.0)
        self.assertEqual(derivative(lambda t: t**0, 5.0), 0.0)
        with self.assertRaises(ValueError):
            derivative(lambda t: t**0.5, 0.0)
        with self.assertRaises(ValueError):
            derivative(lambda t: t**0.5, -1.0)

    def test_comparisons_and_branches_use_the_value(self) -> None:
        """Branches select a piece; max/min and kinks behave predictably."""
        piecewise = lambda v: v[0] ** 2 if v[0] > 1.0 else 3.0 * v[0]  # noqa: E731
        self.assertEqual(gradient(piecewise, (2.0,)), (4.0,))
        self.assertEqual(gradient(piecewise, (0.5,)), (3.0,))
        self.assertEqual(gradient(lambda v: max(v[0], v[1]), (1.0, 2.0)), (0.0, 1.0))
        self.assertEqual(derivative(abs, 0.0), 0.0)
        self.assertEqual(
            gradient(lambda v: M.hypot(v[0], v[1]), (0.0, 0.0)), (0.0, 0.0)
        )
        d = Dual(1.5, (1.0,))
        self.assertTrue(d == 1.5 and d < 2 and d >= 1.5 and bool(d))
        self.assertEqual(hash(d), hash(1.5))
        self.assertTrue(M.isfinite(d) and not M.isnan(d) and not M.isinf(d))

    def test_fsum_is_exact_and_differentiable(self) -> None:
        """admath.fsum keeps math.fsum accuracy on values and tangents."""
        grad = gradient(lambda v: M.fsum([v[0], 1e100, v[1], -1e100]), (1.0, 2.0))
        self.assertEqual(grad, (1.0, 1.0))
        value, _ = value_and_gradient(lambda v: M.fsum([v[0], 1e100, -1e100]), (1.0,))
        self.assertEqual(value, 1.0)
        self.assertEqual(M.fsum([0.1] * 10), 1.0)

    def test_constant_function_has_zero_gradient(self) -> None:
        """A function that ignores its arguments returns a zero gradient."""
        self.assertEqual(
            value_and_gradient(lambda v: 4.0, (1.0, 2.0)), (4.0, (0.0, 0.0))
        )


class FailureTests(unittest.TestCase):
    """Unsupported code fails loudly instead of losing the derivative."""

    def test_math_functions_and_conversions_raise(self) -> None:
        """math.*, float, int, round, floor, //, % all raise the AD error."""
        d = Dual(0.5, (1.0,))
        operations = (
            lambda: math.exp(d),
            lambda: float(d),
            lambda: int(d),
            lambda: round(d),
            lambda: math.floor(d),
            lambda: math.fsum([d]),
            lambda: d // 1,
            lambda: d % 1,
            lambda: [1, 2][d],
        )
        for operation in operations:
            with self.assertRaises(AutodiffUnsupportedError):
                operation()
        with self.assertRaisesRegex(AutodiffUnsupportedError, "admath"):
            gradient(lambda v: math.exp(v[0]), (1.0,))
        self.assertTrue(issubclass(AutodiffUnsupportedError, TypeError))

    def test_third_derivative_of_lgamma_is_refused(self) -> None:
        """Trigamma of a dual raises; the Hessian of lgamma still works."""
        with self.assertRaises(AutodiffUnsupportedError):
            derivative(M.trigamma, 2.0)
        # d^2/dx^2 lgamma(x) at 1 is pi^2/6.
        self.assertAlmostEqual(
            hessian(lambda v: M.lgamma(v[0]), (1.0,))[0][0], math.pi**2 / 6
        )

    def test_invalid_inputs(self) -> None:
        """Bad points, outputs and constructors are rejected."""
        with self.assertRaises(ValueError):
            gradient(lambda v: v[0], ())
        with self.assertRaises(ValueError):
            gradient(lambda v: v[0], (math.nan,))
        with self.assertRaises(TypeError):
            gradient(lambda v: "x", (1.0,))
        with self.assertRaises(TypeError):
            Dual("a", (1.0,))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Dual(1.0, (True,))
        with self.assertRaises(TypeError):
            admath()
        with self.assertRaises(ValueError):
            Dual(1.0, (1.0,)) + Dual(1.0, (1.0, 0.0))


class GradientEvaluatorTests(unittest.TestCase):
    """The consumer-facing oracle records the method and falls back."""

    @staticmethod
    def central(function, point):
        """Return a plain central difference used as the fallback."""
        h = 1e-6
        out = []
        for i in range(len(point)):
            up = list(point)
            down = list(point)
            up[i] += h
            down[i] -= h
            out.append((function(tuple(up)) - function(tuple(down))) / (2 * h))
        return out

    def test_auto_uses_autodiff_for_dual_code(self) -> None:
        """Dual-capable code is differentiated exactly with one call."""
        calls = []

        def f(v):
            """Count calls."""
            calls.append(1)
            return M.exp(v[0]) * v[1]

        evaluator = GradientEvaluator(f, "auto", finite_difference=self.central)
        value, grad = evaluator.value_and_gradient((0.0, 2.0))
        self.assertEqual((value, grad), (2.0, (2.0, 1.0)))
        self.assertEqual(
            (evaluator.method, evaluator.fallback_reason), ("autodiff", None)
        )
        self.assertEqual(len(calls), 1)

    def test_auto_falls_back_and_records_reason(self) -> None:
        """math-based code falls back to finite differences permanently."""
        evaluator = GradientEvaluator(
            lambda v: math.exp(v[0]), "auto", finite_difference=self.central
        )
        value, grad = evaluator.value_and_gradient((0.0,))
        self.assertEqual(value, 1.0)
        self.assertAlmostEqual(grad[0], 1.0, delta=1e-8)
        self.assertEqual(evaluator.method, "finite-difference")
        self.assertIn("AutodiffUnsupportedError", evaluator.fallback_reason)
        self.assertAlmostEqual(evaluator.gradient((1.0,))[0], math.e, delta=1e-7)

    def test_explicit_modes_and_errors(self) -> None:
        """Autodiff raises, finite-difference never tries duals, callables work."""
        strict = GradientEvaluator(lambda v: math.exp(v[0]), "autodiff")
        with self.assertRaises(AutodiffUnsupportedError):
            strict.value_and_gradient((0.0,))
        fd = GradientEvaluator(
            lambda v: v[0] ** 2, "finite-difference", finite_difference=self.central
        )
        self.assertEqual(fd.method, "finite-difference")
        self.assertAlmostEqual(fd.gradient((3.0,))[0], 6.0, delta=1e-6)
        analytic = GradientEvaluator(lambda v: v[0] ** 2, lambda v: (2 * v[0],))
        self.assertEqual(analytic.value_and_gradient((3.0,)), (9.0, (6.0,)))
        self.assertEqual(analytic.method, "analytic")
        infinite = GradientEvaluator(
            lambda v: -math.inf, "finite-difference", finite_difference=self.central
        )
        self.assertEqual(infinite.value_and_gradient((1.0,)), (-math.inf, None))
        with self.assertRaises(ValueError):
            infinite.gradient((1.0,))
        with self.assertRaises(ValueError):
            GradientEvaluator(lambda v: 0.0, "magic")
        with self.assertRaises(ValueError):
            GradientEvaluator(lambda v: 0.0, "auto")
        # A genuine error that also happens for floats propagates unchanged.
        broken = GradientEvaluator(
            lambda v: v[0] + "x", "auto", finite_difference=self.central
        )
        with self.assertRaises(TypeError) as caught:
            broken.value_and_gradient((1.0,))
        self.assertNotIsInstance(caught.exception, AutodiffUnsupportedError)


if __name__ == "__main__":
    unittest.main()
