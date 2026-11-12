"""Tests for reverse-mode automatic differentiation and its use (ADR-0069)."""

from __future__ import annotations

import math
import random
import sys
import unittest
import warnings

from probstat.distributions import Gamma, Normal, RealSupport, Uniform
from probstat.inference import (
    hamiltonian_monte_carlo,
    laplace_approximation,
    no_u_turn_sampler,
    ppl,
)
from probstat.numerics import (
    GRADIENT_OPTIONS,
    AutodiffUnsupportedError,
    Dual,
    GradientEvaluator,
    NonDifferentiableError,
    ReverseVariable,
    admath,
    digamma,
    gradient,
    hessian,
    hessian_vector_product,
    jacobian,
    reverse_gradient,
    reverse_hessian,
    reverse_jacobian,
    reverse_value_and_gradient,
    reverse_value_gradient_and_hessian,
    trigamma,
    value_and_gradient,
    vector_jacobian_product,
)

M = admath
TIGHT = 1e-14


def close(test: unittest.TestCase, actual, expected, tolerance=TIGHT, scale=1.0):
    """Assert ``|actual - expected| <= tolerance * max(scale, |expected|)``."""
    bound = tolerance * max(scale, abs(expected))
    test.assertLessEqual(abs(actual - expected), bound, f"{actual} vs {expected}")


def kitchen_sink(v):
    """Use nearly every admath function (same expression as test_autodiff)."""
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


# Oracle of tests/test_autodiff.py: mpmath 1.4 at 50 digits, mpmath.diff of
# the same expression at (0.3, -0.7, 1.1).
SINK_POINT = (0.3, -0.7, 1.1)
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

# mpmath 1.3 at 40 digits: mpmath.digamma(2.5), mpmath.polygamma(1, 2.5) and
# mpmath.polygamma(2, 2.5).
DIGAMMA_2_5 = 0.7031566406452431872256903336679110994735
TRIGAMMA_2_5 = 0.4903577561002348649728010554936311232124
TETRAGAMMA_2_5 = -0.2362040516417274030037416685677072781172


def unary_cases():
    """Return ``(name, f, f', f'', x)`` with closed-form derivatives."""
    s = 2.0 / math.sqrt(math.pi)
    return (
        ("exp", M.exp, math.exp, math.exp, 0.7),
        ("expm1", M.expm1, math.exp, math.exp, -0.4),
        ("log", M.log, lambda x: 1 / x, lambda x: -1 / x**2, 1.7),
        ("log1p", M.log1p, lambda x: 1 / (1 + x), lambda x: -1 / (1 + x) ** 2, 0.3),
        (
            "log2",
            M.log2,
            lambda x: 1 / (x * math.log(2)),
            lambda x: -1 / (x * x * math.log(2)),
            3.0,
        ),
        (
            "log10",
            M.log10,
            lambda x: 1 / (x * math.log(10)),
            lambda x: -1 / (x * x * math.log(10)),
            0.6,
        ),
        (
            "sqrt",
            M.sqrt,
            lambda x: 0.5 / math.sqrt(x),
            lambda x: -0.25 * x**-1.5,
            2.3,
        ),
        ("sin", M.sin, math.cos, lambda x: -math.sin(x), 1.1),
        ("cos", M.cos, lambda x: -math.sin(x), lambda x: -math.cos(x), 1.1),
        (
            "tan",
            M.tan,
            lambda x: 1 / math.cos(x) ** 2,
            lambda x: 2 * math.tan(x) / math.cos(x) ** 2,
            0.4,
        ),
        (
            "asin",
            M.asin,
            lambda x: 1 / math.sqrt(1 - x * x),
            lambda x: x * (1 - x * x) ** -1.5,
            0.35,
        ),
        (
            "acos",
            M.acos,
            lambda x: -1 / math.sqrt(1 - x * x),
            lambda x: -x * (1 - x * x) ** -1.5,
            -0.6,
        ),
        (
            "atan",
            M.atan,
            lambda x: 1 / (1 + x * x),
            lambda x: -2 * x / (1 + x * x) ** 2,
            2.0,
        ),
        ("sinh", M.sinh, math.cosh, math.sinh, 0.9),
        ("cosh", M.cosh, math.sinh, math.cosh, -0.9),
        (
            "tanh",
            M.tanh,
            lambda x: 1 - math.tanh(x) ** 2,
            lambda x: -2 * math.tanh(x) * (1 - math.tanh(x) ** 2),
            0.5,
        ),
        (
            "erf",
            M.erf,
            lambda x: s * math.exp(-x * x),
            lambda x: -2 * x * s * math.exp(-x * x),
            0.8,
        ),
        (
            "erfc",
            M.erfc,
            lambda x: -s * math.exp(-x * x),
            lambda x: 2 * x * s * math.exp(-x * x),
            -0.3,
        ),
        ("lgamma", M.lgamma, lambda x: DIGAMMA_2_5, lambda x: TRIGAMMA_2_5, 2.5),
        ("fabs", M.fabs, lambda x: -1.0, lambda x: 0.0, -1.5),
        ("abs", abs, lambda x: 1.0, lambda x: 0.0, 1.5),
        ("square", lambda t: t**2, lambda x: 2 * x, lambda x: 2.0, -1.3),
        ("cube", lambda t: t**3, lambda x: 3 * x * x, lambda x: 6 * x, -1.3),
        (
            "power",
            lambda t: t**2.5,
            lambda x: 2.5 * x**1.5,
            lambda x: 3.75 * x**0.5,
            1.7,
        ),
        (
            "rpow",
            lambda t: 3.0**t,
            lambda x: 3.0**x * math.log(3),
            lambda x: 3.0**x * math.log(3) ** 2,
            0.8,
        ),
        ("reciprocal", lambda t: 2.0 / t, lambda x: -2 / x**2, lambda x: 4 / x**3, 1.9),
        ("negation", lambda t: 5.0 - t, lambda x: -1.0, lambda x: 0.0, 0.2),
    )


class ElementaryRuleTests(unittest.TestCase):
    """Reverse mode reproduces closed-form first and second derivatives."""

    def test_unary_rules_match_closed_forms_and_forward_mode(self) -> None:
        """Each rule: value, f', f'' to 1e-14 relative; equal to forward mode."""
        for name, f, first, second, x in unary_cases():
            with self.subTest(function=name):

                def scalar(v, f=f):
                    """Apply the rule to the only coordinate."""
                    return f(v[0])

                value, grad = reverse_value_and_gradient(scalar, (x,))
                close(self, value, f(x))
                close(self, grad[0], first(x))
                close(self, grad[0], gradient(scalar, (x,))[0])
                curvature = reverse_hessian(scalar, (x,))[0][0]
                close(self, curvature, second(x), 1e-13)
                close(self, curvature, hessian(scalar, (x,))[0][0], 1e-13)

    def test_digamma_and_trigamma(self) -> None:
        """Derivative of digamma is trigamma (mpmath literal)."""
        grad = reverse_gradient(lambda v: M.digamma(v[0]), (2.5,))[0]
        close(self, grad, TRIGAMMA_2_5, 1e-12)
        close(self, digamma(2.5), DIGAMMA_2_5, 1e-12)
        close(self, trigamma(2.5), TRIGAMMA_2_5, 1e-12)
        third = reverse_hessian(lambda v: M.lgamma(v[0]) * 1.0, (2.5,))[0][0]
        close(self, third, TRIGAMMA_2_5, 1e-12)
        self.assertLess(TETRAGAMMA_2_5, 0.0)
        with self.assertRaises(AutodiffUnsupportedError):
            reverse_hessian(lambda v: M.digamma(v[0]), (2.5,))
        with self.assertRaises(AutodiffUnsupportedError):
            reverse_gradient(lambda v: M.trigamma(v[0]), (2.5,))

    def test_binary_and_variadic_rules(self) -> None:
        """atan2, hypot, fsum, pow with two variables, division, log base."""
        point = (0.7, -1.3, 2.1)
        cases = (
            lambda v: M.atan2(v[0], v[1]),
            lambda v: M.atan2(2.0, v[1]) + M.atan2(v[0], 3.0),
            lambda v: M.hypot(v[0], v[1], v[2]),
            lambda v: M.hypot(v[0], 4.0),
            lambda v: M.fsum([v[0], 2.0 * v[1], v[2], v[0], 0.5]),
            lambda v: v[2] ** v[0],
            lambda v: M.pow(v[2], v[0] + 1.0),
            lambda v: v[0] / v[1] - v[1] / (v[2] + 1.0),
            lambda v: M.log(v[2], 3.0) + M.log(v[2] + 1.0, v[2]),
            lambda v: max(v[0], v[1]) * min(v[2], 4.0),
            lambda v: sum(v) * v[0] - v[1] * v[2],
        )
        for index, f in enumerate(cases):
            with self.subTest(case=index):
                forward_value, forward_grad = value_and_gradient(f, point)
                value, grad = reverse_value_and_gradient(f, point)
                close(self, value, forward_value)
                for a, b in zip(grad, forward_grad, strict=True):
                    close(self, a, b, TIGHT, 1.0)
                h_reverse = reverse_hessian(f, point)
                h_forward = hessian(f, point)
                for row_r, row_f in zip(h_reverse, h_forward, strict=True):
                    for a, b in zip(row_r, row_f, strict=True):
                        close(self, a, b, 1e-13, 1.0)

    def test_kitchen_sink_against_mpmath(self) -> None:
        """Value, gradient and Hessian of a 13-term function equal mpmath."""
        value, grad, curvature = reverse_value_gradient_and_hessian(
            kitchen_sink, SINK_POINT
        )
        close(self, value, SINK_VALUE)
        for a, b in zip(grad, SINK_GRADIENT, strict=True):
            close(self, a, b, 1e-14, 1.0)
        for row, expected in zip(curvature, SINK_HESSIAN, strict=True):
            for a, b in zip(row, expected, strict=True):
                close(self, a, b, 1e-13, 1.0)
        forward = gradient(kitchen_sink, SINK_POINT)
        for a, b in zip(grad, forward, strict=True):
            close(self, a, b, 1e-14, 1.0)

    def test_kink_conventions(self) -> None:
        """abs(0) and hypot at the origin have derivative 0 as in forward mode."""
        self.assertEqual(reverse_gradient(lambda v: abs(v[0]), (0.0,)), (0.0,))
        self.assertEqual(
            reverse_gradient(lambda v: M.hypot(v[0], v[1]), (0.0, 0.0)), (0.0, 0.0)
        )
        self.assertEqual(reverse_gradient(lambda v: v[0] ** 0, (0.0,)), (0.0,))
        self.assertEqual(reverse_gradient(lambda v: 0.0**v[0], (2.0,)), (0.0,))
        self.assertEqual(reverse_gradient(lambda v: v[0] ** 1.0, (0.0,)), (1.0,))


def rosenbrock(v):
    """Return the extended Rosenbrock function."""
    total = 0.0
    for i in range(len(v) - 1):
        total = total + 100.0 * (v[i + 1] - v[i] * v[i]) ** 2 + (1.0 - v[i]) ** 2
    return total


def rosenbrock_gradient_and_hessian(x):
    """Return the closed-form gradient and (tridiagonal) Hessian."""
    d = len(x)
    grad = [0.0] * d
    h = [[0.0] * d for _ in range(d)]
    for i in range(d - 1):
        r = x[i + 1] - x[i] * x[i]
        grad[i] += -400.0 * x[i] * r - 2.0 * (1.0 - x[i])
        grad[i + 1] += 200.0 * r
        h[i][i] += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0
        h[i][i + 1] += -400.0 * x[i]
        h[i + 1][i] += -400.0 * x[i]
        h[i + 1][i + 1] += 200.0
    return grad, h


def logistic_data(n=40, d=5, seed=3):
    """Return a fixed design matrix and binary responses."""
    generator = random.Random(seed)
    rows = [[1.0] + [generator.gauss(0.0, 1.0) for _ in range(d - 1)] for _ in range(n)]
    labels = [float(generator.random() < 0.5) for _ in range(n)]
    return rows, labels


def logistic_log_likelihood(rows, labels):
    """Return ``beta -> sum y eta - log(1 + exp(eta))``."""

    def log_likelihood(beta):
        """Bernoulli-logit log-likelihood."""
        total = 0.0
        for row, y in zip(rows, labels, strict=True):
            eta = 0.0
            for x, b in zip(row, beta, strict=True):
                eta = eta + x * b
            total = total + y * eta - M.log1p(M.exp(eta))
        return total

    return log_likelihood


class ClosedFormModelTests(unittest.TestCase):
    """Gradients and Hessians of standard objectives equal their closed forms."""

    def test_rosenbrock(self) -> None:
        """Rosenbrock in 10 dimensions: gradient and Hessian to 1e-14."""
        generator = random.Random(5)
        x = tuple(generator.uniform(-1.5, 1.5) for _ in range(10))
        expected_grad, expected_h = rosenbrock_gradient_and_hessian(x)
        value, grad, h = reverse_value_gradient_and_hessian(rosenbrock, x)
        close(self, value, rosenbrock(x))
        scale = max(abs(g) for g in expected_grad)
        for a, b in zip(grad, expected_grad, strict=True):
            close(self, a, b, 1e-14, scale)
        h_scale = max(abs(e) for row in expected_h for e in row)
        for row, expected in zip(h, expected_h, strict=True):
            for a, b in zip(row, expected, strict=True):
                close(self, a, b, 1e-14, h_scale)

    def test_logistic_regression(self) -> None:
        """Gradient X'(y - p) and Hessian -X' W X of the logit likelihood."""
        rows, labels = logistic_data()
        f = logistic_log_likelihood(rows, labels)
        beta = (0.2, -0.5, 0.3, 0.1, -0.2)
        d = len(beta)
        probabilities = []
        for row in rows:
            eta = sum(x * b for x, b in zip(row, beta, strict=True))
            probabilities.append(1.0 / (1.0 + math.exp(-eta)))
        expected_grad = [
            math.fsum(
                row[j] * (y - p)
                for row, y, p in zip(rows, labels, probabilities, strict=True)
            )
            for j in range(d)
        ]
        expected_h = [
            [
                -math.fsum(
                    row[i] * row[j] * p * (1.0 - p)
                    for row, p in zip(rows, probabilities, strict=True)
                )
                for j in range(d)
            ]
            for i in range(d)
        ]
        _, grad, h = reverse_value_gradient_and_hessian(f, beta)
        for a, b in zip(grad, expected_grad, strict=True):
            close(self, a, b, 1e-13, 1.0)
        for row, expected in zip(h, expected_h, strict=True):
            for a, b in zip(row, expected, strict=True):
                close(self, a, b, 1e-13, 1.0)
        forward_grad = gradient(f, beta)
        for a, b in zip(grad, forward_grad, strict=True):
            close(self, a, b, 1e-13, 1.0)
        direction = (1.0, -2.0, 0.5, 0.0, 3.0)
        product = hessian_vector_product(f, beta, direction)
        for i in range(d):
            expected = math.fsum(expected_h[i][j] * direction[j] for j in range(d))
            close(self, product[i], expected, 1e-13, 1.0)

    def test_jacobian_and_vector_jacobian_product(self) -> None:
        """reverse_jacobian equals forward jacobian; w'J matches the rows."""

        def f(v):
            """Map R^3 to R^4 (one constant output)."""
            x, y, z = v
            return [x * y + M.sin(z), M.exp(x - z), y / (1 + x * x), 2.0]

        point = (0.4, -1.1, 0.9)
        rows = reverse_jacobian(f, point)
        expected = jacobian(f, point)
        for row, row_f in zip(rows, expected, strict=True):
            for a, b in zip(row, row_f, strict=True):
                close(self, a, b, TIGHT, 1.0)
        weights = (0.5, -2.0, 1.5, 7.0)
        product = vector_jacobian_product(f, point, weights)
        for j in range(3):
            target = math.fsum(w * row[j] for w, row in zip(weights, rows, strict=True))
            close(self, product[j], target, TIGHT, 1.0)
        with self.assertRaises(ValueError):
            vector_jacobian_product(f, point, (1.0, 2.0))
        with self.assertRaises(TypeError):
            reverse_jacobian(lambda v: v[0], point)

    def test_constant_function_and_unused_inputs(self) -> None:
        """A constant function has a zero gradient and Hessian."""
        self.assertEqual(reverse_value_and_gradient(lambda v: 3.0, (1.0, 2.0)),
                         (3.0, (0.0, 0.0)))
        self.assertEqual(reverse_gradient(lambda v: v[1] * 2.0, (1.0, 2.0)),
                         (0.0, 2.0))
        self.assertEqual(reverse_hessian(lambda v: 3.0, (1.0,)), ((0.0,),))


class DeepTapeTests(unittest.TestCase):
    """The backward sweep is iterative."""

    def test_150000_operations_without_recursion_error(self) -> None:
        """A chain of 50 000 sin-scale-shift steps (150 000 nodes)."""
        steps = 50_000
        recorded = {}

        def chain(v):
            """Iterate ``x <- 0.5 sin(x) + 0.3``."""
            x = v[0]
            for _ in range(steps):
                x = 0.5 * M.sin(x) + 0.3
            recorded["tape"] = len(x.tape)
            return x

        limit = sys.getrecursionlimit()
        value, grad = reverse_value_and_gradient(chain, (0.2,))
        self.assertEqual(sys.getrecursionlimit(), limit)
        self.assertGreater(recorded["tape"], 100_000)
        x, derivative = 0.2, 1.0
        for _ in range(steps):
            derivative *= 0.5 * math.cos(x)
            x = 0.5 * math.sin(x) + 0.3
        close(self, value, x)
        close(self, grad[0], derivative, 1e-12)

    def test_deep_sum_matches_closed_form(self) -> None:
        """200 000 additions of x_i^2 / 2: gradient equals x exactly."""
        point = tuple(0.001 * i for i in range(1, 201))

        def f(v):
            """Repeat a quadratic 1000 times."""
            total = 0.0
            for _ in range(1000):
                for x in v:
                    total = total + 0.5 * x * x
            return total

        grad = reverse_gradient(f, point)
        for g, x in zip(grad, point, strict=True):
            close(self, g, 1000.0 * x, 1e-12)


class ErrorTests(unittest.TestCase):
    """Unsupported operations and non-differentiable points fail clearly."""

    def test_float_conversions_raise(self) -> None:
        """float(), math.*, int, round, floor, //, % and indexing raise."""
        cases = (
            float,
            math.exp,
            int,
            round,
            math.floor,
            math.ceil,
            math.trunc,
            complex,
            lambda t: t // 2,
            lambda t: 2 // t,
            lambda t: t % 2,
            lambda t: 2 % t,
            lambda t: [1, 2, 3][t],
            lambda t: math.fsum([t, 1.0]),
        )
        for index, operation in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(AutodiffUnsupportedError):
                    reverse_gradient(lambda v, op=operation: op(v[0]), (1.0,))

    def test_non_differentiable_points(self) -> None:
        """Unbounded derivatives raise NonDifferentiableError (a ValueError)."""
        cases = (
            (lambda v: M.sqrt(v[0]), (0.0,)),
            (lambda v: M.asin(v[0]), (1.0,)),
            (lambda v: M.acos(v[0]), (-1.0,)),
            (lambda v: v[0] ** 0.5, (0.0,)),
            (lambda v: v[0] ** -1.0, (0.0,)),
            (lambda v: M.atan2(v[0], v[1]), (0.0, 0.0)),
            (lambda v: v[0] ** v[1], (-2.0, 2.0)),
            (lambda v: v[0] ** v[1], (0.0, 2.0)),
            (lambda v: (-2.0) ** v[0], (2.0,)),
        )
        for index, (f, point) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(NonDifferentiableError):
                    reverse_gradient(f, point)
        self.assertTrue(issubclass(NonDifferentiableError, ValueError))

    def test_mixing_and_construction(self) -> None:
        """Duals, foreign tapes, bad outputs and direct construction fail."""
        with self.assertRaises(TypeError):
            ReverseVariable()
        with self.assertRaises(AutodiffUnsupportedError):
            reverse_gradient(lambda v: v[0] + Dual(1.0, (1.0,)), (1.0,))
        with self.assertRaises(AutodiffUnsupportedError):
            reverse_gradient(lambda v: Dual(1.0, (1.0,)) * v[0], (1.0,))
        outer = {}

        def inner(v):
            """Combine a variable of an enclosing tape with this one."""
            return v[0] * outer["x"]

        def nested(v):
            """Differentiate inside a differentiated function."""
            outer["x"] = v[0]
            return reverse_gradient(inner, (2.0,))[0]

        with self.assertRaises(AutodiffUnsupportedError):
            reverse_gradient(nested, (1.0,))
        with self.assertRaises(TypeError):
            reverse_gradient(lambda v: "text", (1.0,))
        with self.assertRaises(TypeError):
            reverse_gradient(lambda v: v[0] + "text", (1.0,))
        with self.assertRaises(ValueError):
            reverse_gradient(lambda v: v[0], ())
        with self.assertRaises(ValueError):
            hessian_vector_product(lambda v: v[0], (1.0,), (1.0, 2.0))

    def test_comparisons_hash_and_repr(self) -> None:
        """Comparisons use the value; repr names the tape index."""

        def f(v):
            """Exercise comparisons inside the recorded function."""
            x, y = v
            self.assertTrue(x < y and y > x and x <= 1.0 and y >= 2.0)
            self.assertTrue(x == 1.0 and x != y and bool(x))
            self.assertEqual(hash(x), hash(1.0))
            self.assertIn("index=0", repr(x))
            self.assertTrue(M.isfinite(x) and not M.isnan(y) and not M.isinf(y))
            return -x + (+y)

        self.assertEqual(reverse_gradient(f, (1.0, 2.0)), (-1.0, 1.0))


class GradientEvaluatorTests(unittest.TestCase):
    """The shared oracle offers reverse mode with a recorded fallback."""

    @staticmethod
    def central(function, point):
        """Return a simple central-difference gradient."""
        h = 1e-6
        result = []
        for i in range(len(point)):
            up = list(point)
            down = list(point)
            up[i] += h
            down[i] -= h
            result.append((function(tuple(up)) - function(tuple(down))) / (2 * h))
        return result

    def test_options(self) -> None:
        """reverse, auto-reverse (with fallback), forward and finite aliases."""

        def f(v):
            """Smooth test function."""
            return M.exp(v[0]) * v[1] - v[1] ** 3

        point = (0.3, -0.8)
        expected = (math.exp(0.3) * -0.8, math.exp(0.3) - 3 * 0.64)
        self.assertEqual(
            GRADIENT_OPTIONS,
            (
                "auto",
                "auto-reverse",
                "autodiff",
                "forward",
                "reverse",
                "finite-difference",
                "finite",
            ),
        )
        strict = GradientEvaluator(f, "reverse")
        for a, b in zip(strict.gradient(point), expected, strict=True):
            close(self, a, b, TIGHT, 1.0)
        self.assertEqual((strict.method, strict.gradient_evaluations), ("reverse", 1))
        forward = GradientEvaluator(f, "forward")
        self.assertEqual((forward.requested, forward.method), ("autodiff", "autodiff"))
        auto = GradientEvaluator(f, "auto-reverse", finite_difference=self.central)
        auto.gradient(point)
        self.assertEqual((auto.method, auto.fallback_reason), ("reverse", None))
        finite = GradientEvaluator(f, "finite", finite_difference=self.central)
        self.assertEqual(finite.method, "finite-difference")
        self.assertAlmostEqual(finite.gradient(point)[0], expected[0], delta=1e-8)

        def float_only(v):
            """Return a value through math.exp (not traceable)."""
            return math.exp(v[0]) * v[1]

        fallback = GradientEvaluator(
            float_only, "auto-reverse", finite_difference=self.central
        )
        fallback.gradient(point)
        self.assertEqual(fallback.method, "finite-difference")
        self.assertIn("AutodiffUnsupportedError", fallback.fallback_reason)
        with self.assertRaises(AutodiffUnsupportedError):
            GradientEvaluator(float_only, "reverse").gradient(point)
        with self.assertRaises(AutodiffUnsupportedError):
            GradientEvaluator(lambda v: v[0] + "x", "reverse").gradient(point)
        with self.assertRaises(ValueError):
            GradientEvaluator(f, "auto-reverse")
        outside = GradientEvaluator(lambda v: -math.inf + 0.0 * v[0], "reverse")
        self.assertEqual(outside.value_and_gradient(point), (-math.inf, None))
        self.assertEqual(outside.gradient_evaluations, 0)


SCALES = tuple(0.5 + 0.25 * i for i in range(10))


def gaussian_10d(x):
    """Independent normal log-density with scales 0.5 .. 2.75."""
    total = 0.0
    for value, scale in zip(x, SCALES, strict=True):
        z = value / scale
        total = total - 0.5 * z * z
    return total


def quiet(function, *args, **kwargs):
    """Call a sampler with convergence warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def pooled(result):
    """Return all draws of every chain as one flat list."""
    return [
        value for chain in result.mcmc.chains for row in chain.samples for value in row
    ]


class SamplerTests(unittest.TestCase):
    """HMC and NUTS with reverse mode reproduce the forward-mode runs."""

    COMMON = {"draws": 100, "warmup": 100, "chains": 2, "seed": 7}

    def test_nuts_reverse_matches_forward_and_costs_one_call(self) -> None:
        """Same seed: draws agree to 1e-9; one call per gradient."""
        start = [0.1] * 10
        forward = quiet(no_u_turn_sampler, gaussian_10d, start, **self.COMMON)
        reverse = quiet(
            no_u_turn_sampler, gaussian_10d, start, gradient="reverse", **self.COMMON
        )
        alias = quiet(
            no_u_turn_sampler, gaussian_10d, start, gradient="forward", **self.COMMON
        )
        finite = quiet(
            no_u_turn_sampler, gaussian_10d, start, gradient="finite", **self.COMMON
        )
        self.assertEqual(forward.gradient_method, "autodiff")
        self.assertEqual(reverse.gradient_method, "reverse")
        self.assertEqual(finite.gradient_method, "finite-difference")
        self.assertEqual(pooled(alias), pooled(forward))
        for a, b in zip(pooled(reverse), pooled(forward), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)
        for record in reverse.statistics:
            self.assertEqual(
                record.log_density_evaluations, record.gradient_evaluations
            )
        for record in finite.statistics:
            self.assertEqual(
                record.log_density_evaluations, 21 * record.gradient_evaluations
            )
        reverse_calls = sum(r.log_density_evaluations for r in reverse.statistics)
        finite_calls = sum(r.log_density_evaluations for r in finite.statistics)
        self.assertLess(reverse_calls * 10, finite_calls)
        with self.assertRaises(ValueError):
            no_u_turn_sampler(gaussian_10d, start, gradient="backward", **self.COMMON)

    def test_static_hmc_reverse_matches_forward(self) -> None:
        """Static HMC with jitter: identical trajectories up to rounding."""
        options = {"leapfrog_steps": 8, "step_size_jitter": 0.2, **self.COMMON}
        start = [0.3] * 10
        forward = quiet(hamiltonian_monte_carlo, gaussian_10d, start, **options)
        reverse = quiet(
            hamiltonian_monte_carlo, gaussian_10d, start, gradient="reverse", **options
        )
        for a, b in zip(pooled(reverse), pooled(forward), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)
        self.assertEqual(
            reverse.gradient_evaluations,
            sum(r.log_density_evaluations for r in reverse.statistics),
        )

    def test_auto_reverse_falls_back_for_math_densities(self) -> None:
        """A math.* density falls back to finite differences and says why."""

        def density(x):
            """Float-only density."""
            return -0.5 * math.fsum(v * v for v in x)

        result = quiet(
            no_u_turn_sampler,
            density,
            [0.1, 0.2],
            gradient="auto-reverse",
            draws=20,
            warmup=20,
            chains=1,
            seed=2,
        )
        self.assertEqual(result.gradient_method, "finite-difference")
        self.assertIn("AutodiffUnsupportedError", result.gradient_fallback_reason)
        with self.assertRaises(AutodiffUnsupportedError):
            quiet(
                no_u_turn_sampler,
                density,
                [0.1, 0.2],
                gradient="reverse",
                draws=20,
                warmup=20,
                chains=1,
                seed=2,
            )

    def test_laplace_reverse_hessian(self) -> None:
        """Laplace with reverse mode: exact covariance, method recorded."""
        precision = ((2.0, 0.6), (0.6, 1.0))

        def density(x):
            """Correlated Gaussian log-density."""
            a, b = x
            return -0.5 * (
                precision[0][0] * a * a
                + 2.0 * precision[0][1] * a * b
                + precision[1][1] * b * b
            ) + a

        forward = laplace_approximation(density, (0.0, 0.0))
        reverse = laplace_approximation(density, (0.0, 0.0), gradient="reverse")
        self.assertEqual(
            (reverse.gradient_method, reverse.hessian_method), ("reverse", "reverse")
        )
        determinant = 2.0 * 1.0 - 0.36
        exact = ((1.0 / determinant, -0.6 / determinant),
                 (-0.6 / determinant, 2.0 / determinant))
        for row, expected, row_f in zip(
            reverse.covariance, exact, forward.covariance, strict=True
        ):
            for a, b, c in zip(row, expected, row_f, strict=True):
                close(self, a, b, 1e-12, 1.0)
                close(self, a, c, 1e-12, 1.0)
        for a, b in zip(reverse.mode, forward.mode, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8)


class DualNormal:
    """Duck-typed normal law whose log_pdf uses admath (traceable)."""

    support = RealSupport()

    def __init__(self, mu, sigma):
        """Store parameters (floats, duals or reverse-mode variables)."""
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self, x):
        """Return the normal log-density through admath."""
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - M.log(self.sigma) - 0.5 * math.log(2 * math.pi)

    def sample(self, size, *, seed=None):
        """Draw values (prediction only)."""
        generator = random.Random(seed)
        return [generator.gauss(float(self.mu), float(self.sigma)) for _ in range(size)]


DATA = ((1.2, 0.4), (2.1, 1.7), (0.9, -0.3), (1.5, 2.2))


def hierarchical(m):
    """Four group means around a common mean (d = 5), traceable."""
    mu = m.sample("mu", Normal(0.0, 10.0))
    theta = m.sample("theta", [DualNormal(mu, 1.0) for _ in DATA])
    for index, (group, values) in enumerate(zip(theta, DATA, strict=True)):
        m.observe(f"y{index}", [DualNormal(group, 0.8) for _ in values], values)


def library_normal(m):
    """Declare the likelihood with library Normal objects (traceable, ADR-0090)."""
    mu = m.sample("mu", Normal(0.0, 10.0))
    m.observe("y", Normal(mu, 1.0), (1.2, 0.4, 2.1))


def latent_bound(m):
    """Declare a site whose support bound is latent (not traceable)."""
    upper = m.sample("upper", Gamma(4.0, 1.0))
    m.sample("x", Uniform(0.0, upper))


class ProbabilisticProgrammingTests(unittest.TestCase):
    """The PPL uses reverse mode for traceable programs."""

    def test_model_gradient_defaults_to_reverse(self) -> None:
        """ModelIR.gradient equals the closed form and the forward mode."""
        program = ppl.trace(hierarchical)
        u = (0.3, 0.5, 1.0, -0.2, 1.4)
        mu, theta = u[0], u[1:]
        expected = [-mu / 100.0 + sum(t - mu for t in theta)]
        for t, values in zip(theta, DATA, strict=True):
            expected.append(-(t - mu) + sum((y - t) / 0.64 for y in values))
        evaluator = program.gradient_evaluator()
        grad = evaluator.gradient(u)
        self.assertEqual(evaluator.method, "reverse")
        for a, b in zip(grad, expected, strict=True):
            close(self, a, b, 1e-13, 1.0)
        for a, b in zip(program.gradient(u), expected, strict=True):
            close(self, a, b, 1e-13, 1.0)
        forward = program.gradient(u, method="autodiff")
        for a, b in zip(grad, forward, strict=True):
            close(self, a, b, 1e-13, 1.0)

    def test_sample_matches_forward_and_uses_fewer_evaluations(self) -> None:
        """NUTS on the program: reverse equals forward; far fewer calls than FD."""
        common = {"draws": 120, "warmup": 120, "chains": 2, "seed": 11}
        reverse = quiet(ppl.sample, hierarchical, **common)
        forward = quiet(ppl.sample, hierarchical, gradient="autodiff", **common)
        finite = quiet(ppl.sample, hierarchical, gradient="finite-difference", **common)
        self.assertEqual(reverse.gradient_method, "reverse")
        self.assertIsNone(reverse.gradient_fallback_reason)
        self.assertEqual(forward.gradient_method, "autodiff")
        self.assertEqual(finite.gradient_method, "finite-difference")
        for name in ("mu", "theta"):
            for a, b in zip(reverse[name], forward[name], strict=True):
                if isinstance(a, tuple):
                    for x, y in zip(a, b, strict=True):
                        self.assertAlmostEqual(x, y, delta=1e-9)
                else:
                    self.assertAlmostEqual(a, b, delta=1e-9)
        reverse_calls = sum(
            r.log_density_evaluations for r in reverse.engine.statistics
        )
        finite_calls = sum(
            r.log_density_evaluations for r in finite.engine.statistics
        )
        gradients = sum(r.gradient_evaluations for r in reverse.engine.statistics)
        self.assertEqual(reverse_calls, gradients)
        self.assertLess(reverse_calls * 5, finite_calls)
        # Exact posterior mean of mu: group means are N(mu, 1 + 0.8^2 / 2).
        variance = 1.0 + 0.64 / 2.0
        means = [sum(values) / len(values) for values in DATA]
        exact = (sum(means) / variance) / (len(DATA) / variance + 1.0 / 100.0)
        summary = {row.name: row for row in reverse.summary()}
        posterior_mu = summary["mu"]
        self.assertLess(abs(posterior_mu.mean - exact), 4.0 * posterior_mu.mcse_mean)

    def test_laplace_and_advi_use_reverse(self) -> None:
        """Laplace (forward-over-reverse Hessian) and ADVI record reverse mode."""

        def conjugate(m):
            """Declare a normal mean with known variance."""
            mu = m.sample("mu", Normal(0.0, 10.0))
            m.observe("y", [DualNormal(mu, 1.0) for _ in range(5)],
                      (1.2, 0.4, 2.1, 1.7, 0.9))

        laplace = ppl.laplace(conjugate, draws=10, seed=1)
        self.assertEqual(laplace.engine.hessian_method, "reverse")
        self.assertEqual(laplace.gradient_method, "reverse")
        close(self, laplace.engine.covariance[0][0], 1.0 / (0.01 + 5.0), 1e-12)
        advi_fit = quiet(
            ppl.fit_advi, conjugate, draws=10, max_iterations=200, seed=1
        )
        self.assertEqual(advi_fit.gradient_method, "reverse")

    def test_library_distributions_use_reverse(self) -> None:
        """Library Normal accepts tape variables (ADR-0090): exact reverse mode.

        Until ADR-0090 the constructor rejected variables and the run fell
        back to central differences; the gradient now equals the closed form
        to rounding.
        """
        posterior = quiet(
            ppl.sample, library_normal, draws=60, warmup=60, chains=1, seed=4
        )
        self.assertEqual(posterior.gradient_method, "reverse")
        self.assertIsNone(posterior.gradient_fallback_reason)
        program = ppl.trace(library_normal)
        exact = sum(y - 0.3 for y in (1.2, 0.4, 2.1)) - 0.3 / 100.0
        self.assertAlmostEqual(
            program.gradient((0.3,), method="reverse")[0], exact, delta=1e-12
        )

    def test_latent_support_bound_falls_back(self) -> None:
        """A support bound that is a variable is refused: finite differences."""
        program = ppl.trace(latent_bound)
        with self.assertRaises(AutodiffUnsupportedError):
            program.gradient((0.1, 0.2), method="reverse")
        evaluator = program.gradient_evaluator("auto")
        evaluator.gradient((0.1, 0.2))
        self.assertEqual(evaluator.method, "finite-difference")
        self.assertIn("real number", evaluator.fallback_reason)


if __name__ == "__main__":
    unittest.main()
