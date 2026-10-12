"""Tests for adaptive Gauss-Kronrod and general-range double-exponential rules."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Callable
from fractions import Fraction

from probstat.numerics import (
    IntegrationResult,
    exp_sinh,
    gauss_kronrod,
    integrate,
    sinh_sinh,
    tanh_sinh_interval,
)
from probstat.numerics.gauss_kronrod import G7_WEIGHTS, K15_NODES, K15_WEIGHTS

# 33-digit constants of QUADPACK dqk15.f (Netlib), as decimal strings, so the
# moment identities can be checked in exact rational arithmetic.
K15_NODE_TEXT = (
    "0.991455371120812639206854697526329",
    "0.949107912342758524526189684047851",
    "0.864864423359769072789712788640926",
    "0.741531185599394439863864773280788",
    "0.586087235467691130294144838258730",
    "0.405845151377397166906606412076961",
    "0.207784955007898467600689403773245",
    "0",
)
K15_WEIGHT_TEXT = (
    "0.022935322010529224963732008058970",
    "0.063092092629978553290700663189204",
    "0.104790010322250183839876322541518",
    "0.140653259715525918745189590510238",
    "0.169004726639267902826583426598550",
    "0.190350578064785409913256402421014",
    "0.204432940075298892414161999234649",
    "0.209482141084727828012999174891714",
)
G7_WEIGHT_TEXT = (
    "0.129484966168869693270611432679082",
    "0.279705391489276667901467771423780",
    "0.381830050505118944950369775488975",
    "0.417959183673469387755102040816327",
)

Integrand = Callable[[float], float]


def legendre_moment(power: int) -> Fraction:
    """Return ``int_{-1}^{1} x^power dx`` exactly.

    Args:
        power: Non-negative exponent.

    Returns:
        ``2/(power + 1)`` for even powers, zero for odd powers.

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2]. Test helper only.

    """
    return Fraction(2, power + 1) if power % 2 == 0 else Fraction(0)


def kronrod_moment(power: int) -> Fraction:
    """Return the K15 rule applied to ``x^power`` on ``[-1, 1]`` exactly.

    Args:
        power: Non-negative exponent.

    Returns:
        Exact rational value of the rule.

    References:
        [REF-PIESSENS-QUADPACK-1983, routine QK15]. Test helper only.

    """
    nodes = [Fraction(text) for text in K15_NODE_TEXT]
    weights = [Fraction(text) for text in K15_WEIGHT_TEXT]
    total = weights[7] * (1 if power == 0 else 0)
    for node, weight in zip(nodes[:7], weights[:7], strict=True):
        total += weight * (node**power + (-node) ** power)
    return total


def gauss_moment(power: int) -> Fraction:
    """Return the embedded G7 rule applied to ``x^power`` on ``[-1, 1]``.

    Args:
        power: Non-negative exponent.

    Returns:
        Exact rational value of the rule.

    References:
        [REF-PIESSENS-QUADPACK-1983, routine QK15]. Test helper only.

    """
    nodes = [Fraction(text) for text in K15_NODE_TEXT]
    weights = [Fraction(text) for text in G7_WEIGHT_TEXT]
    total = weights[3] * (1 if power == 0 else 0)
    for node, weight in zip(nodes[1:7:2], weights[:3], strict=True):
        total += weight * (node**power + (-node) ** power)
    return total


def square_exp(x: float) -> float:
    """Return ``x^2 exp(-x)`` using ``x * x`` so huge ``x`` gives NaN.

    Args:
        x: Real abscissa.

    Returns:
        Integrand value (NaN for ``x`` near ``1e300``: ``inf * 0``).

    References:
        Test integrand only. No library equation is evaluated.

    """
    return x * x * math.exp(-x)


KNOWN: tuple[tuple[str, Integrand, float, float, float], ...] = (
    ("x^-1/2 on [0,1]", lambda x: x**-0.5, 0.0, 1.0, 2.0),
    ("log x on [0,1]", math.log, 0.0, 1.0, -1.0),
    ("exp(-x) on [0,inf)", lambda x: math.exp(-x), 0.0, math.inf, 1.0),
    ("x^2 exp(-x) on [0,inf)", square_exp, 0.0, math.inf, 2.0),
    (
        "exp(-x^2/2) on R",
        lambda x: math.exp(-0.5 * x * x),
        -math.inf,
        math.inf,
        math.sqrt(2.0 * math.pi),
    ),
    ("1/(1+x^2) on [0,inf)", lambda x: 1.0 / (1.0 + x * x), 0.0, math.inf, math.pi / 2),
    (
        "Gamma(1/2) = int x^-1/2 exp(-x)",
        lambda x: math.exp(-x) / math.sqrt(x),
        0.0,
        math.inf,
        math.sqrt(math.pi),
    ),
    ("exp(x) on (-inf,0]", math.exp, -math.inf, 0.0, 1.0),
)


def peak(x: float) -> float:
    """Return a Lorentzian peak of half-width ``1e-3`` centred at ``0.3``.

    Args:
        x: Real abscissa.

    Returns:
        ``1/((x - 0.3)^2 + 1e-6)``.

    References:
        Test integrand only. No library equation is evaluated.

    """
    return 1.0 / ((x - 0.3) ** 2 + 1.0e-6)


PEAK_EXACT = 1.0e3 * (math.atan(0.7e3) + math.atan(0.3e3))


class GaussKronrodRuleTests(unittest.TestCase):
    """The published G7-K15 constants define the rules they claim to."""

    def test_constants_exact_in_rational_arithmetic(self) -> None:
        """K15 is exact through degree 22 and G7 through degree 13 to 1e-32."""
        k_residual = max(abs(kronrod_moment(p) - legendre_moment(p)) for p in range(23))
        g_residual = max(abs(gauss_moment(p) - legendre_moment(p)) for p in range(14))
        self.assertLess(k_residual, Fraction(1, 10**32))
        self.assertLess(g_residual, Fraction(1, 10**32))
        # Degree 24 and 14 are not integrated exactly: the degrees are sharp.
        self.assertGreater(abs(kronrod_moment(24) - legendre_moment(24)), 1e-10)
        self.assertGreater(abs(gauss_moment(14) - legendre_moment(14)), 1e-6)

    def test_float_constants_match_text(self) -> None:
        """The module constants are the correctly rounded published digits."""
        self.assertEqual(K15_NODES, tuple(float(t) for t in K15_NODE_TEXT))
        self.assertEqual(K15_WEIGHTS, tuple(float(t) for t in K15_WEIGHT_TEXT))
        self.assertEqual(G7_WEIGHTS, tuple(float(t) for t in G7_WEIGHT_TEXT))

    def test_single_panel_exact_for_degree_22_on_random_intervals(self) -> None:
        """One G7-K15 panel integrates random degree-0..22 polynomials."""
        generator = random.Random(20261012)
        for degree in range(23):
            a = generator.uniform(-3.0, 1.0)
            b = a + generator.uniform(0.2, 2.0)
            coefficients = [generator.uniform(-1.0, 1.0) for _ in range(degree + 1)]

            def polynomial(x: float, c: list[float] = coefficients) -> float:
                """Evaluate the random polynomial by Horner's rule.

                References:
                    Test integrand only. No library equation is evaluated.

                """
                total = 0.0
                for coefficient in reversed(c):
                    total = total * x + coefficient
                return total

            exact = math.fsum(
                c * (b ** (k + 1) - a ** (k + 1)) / (k + 1)
                for k, c in enumerate(coefficients)
            )
            scale = math.fsum(
                abs(c) * max(abs(a), abs(b)) ** k * (b - a)
                for k, c in enumerate(coefficients)
            )
            result = gauss_kronrod(polynomial, a, b, limit=1)
            with self.subTest(degree=degree):
                self.assertEqual(result.evaluations, 15)
                self.assertLess(abs(result.value - exact), 2e-14 * scale)


class KnownIntegralTests(unittest.TestCase):
    """Known integrals to near machine precision with bounding estimates."""

    def test_double_exponential_known_integrals(self) -> None:
        """Every case is within 4 ulp-scale of the truth and bounded."""
        for name, function, lower, upper, exact in KNOWN:
            result = integrate(function, lower, upper, method="double-exponential")
            error = abs(result.value - exact)
            with self.subTest(case=name):
                self.assertTrue(result.converged)
                self.assertLessEqual(error, 4.0 * math.ulp(exact))
                self.assertGreaterEqual(result.error_estimate, error)
                self.assertLess(result.error_estimate, 1e-10 * abs(exact))

    def test_gauss_kronrod_known_integrals(self) -> None:
        """Gauss-Kronrod (QAGI on infinite ranges) bounds its true error."""
        for name, function, lower, upper, exact in KNOWN:
            result = gauss_kronrod(function, lower, upper)
            error = abs(result.value - exact)
            with self.subTest(case=name):
                self.assertGreaterEqual(result.error_estimate, error)
                if result.converged:
                    self.assertLessEqual(error, 1e-10 * abs(exact))
                else:
                    # x^-1/2 at x = 0 maps to t = 1 under QAGI; panels next to
                    # t = 1 stop at the bisection floor and say so.
                    self.assertIn("Gamma", name)
                    self.assertLess(error, 1e-7)

    def test_rule_dispatch_on_limits(self) -> None:
        """``integrate`` names the rule chosen from the limits."""
        cases = (
            ((0.0, 1.0), "tanh-sinh"),
            ((0.0, math.inf), "exp-sinh"),
            ((-math.inf, 0.0), "exp-sinh"),
            ((-math.inf, math.inf), "sinh-sinh"),
        )
        for (lower, upper), rule in cases:
            with self.subTest(rule=rule):
                result = integrate(lambda x: math.exp(-x * x), lower, upper)
                self.assertEqual(result.rule, rule)
        forced = integrate(math.exp, 0.0, 1.0, method="gauss-kronrod")
        self.assertEqual(forced.rule, "gauss-kronrod-15")
        self.assertAlmostEqual(forced.value, math.e - 1.0, places=14)

    def test_direct_rule_calls(self) -> None:
        """The three DE rules agree with ``integrate`` and with each other."""
        half_line = exp_sinh(lambda x: math.exp(-2.0 * x), 1.0, math.inf)
        self.assertAlmostEqual(half_line.value, 0.5 * math.exp(-2.0), places=15)
        mirrored = exp_sinh(lambda x: math.exp(2.0 * x), -math.inf, -1.0)
        self.assertAlmostEqual(mirrored.value, half_line.value, places=15)
        whole = sinh_sinh(lambda x: 1.0 / (1.0 + x * x))
        self.assertAlmostEqual(whole.value, math.pi, places=14)
        finite = tanh_sinh_interval(lambda x: math.log(x - 2.0), 2.0, 3.0)
        self.assertTrue(finite.converged)
        self.assertAlmostEqual(finite.value, -1.0, places=14)
        self.assertIsInstance(finite, IntegrationResult)
        self.assertEqual(finite.intervals, ((2.0, 3.0),))


class AdaptivityTests(unittest.TestCase):
    """Refinement follows the difficulty and flags are honest."""

    def test_panels_concentrate_at_sharp_peak(self) -> None:
        """The narrowest panels surround the peak and widths grow away from it."""
        result = gauss_kronrod(peak, 0.0, 1.0)
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.error_estimate, abs(result.value - PEAK_EXACT))
        self.assertLess(abs(result.value - PEAK_EXACT), 1e-10 * PEAK_EXACT)
        by_width = sorted(result.intervals, key=lambda p: p[1] - p[0])
        for left, right in by_width[:4]:
            self.assertLess(abs(0.5 * (left + right) - 0.3), 3e-3)
        for left, right in by_width[:10]:
            self.assertTrue(right > 0.29 and left < 0.31)
        widths = [p[1] - p[0] for p in by_width]
        self.assertGreater(widths[-1] / widths[0], 200.0)
        # Half of [0, 1] holds a tenth of the panels: [0.5, 1] needs only two.
        self.assertEqual(sum(1 for p in result.intervals if p[0] >= 0.5), 2)
        # The panels tile [0, 1] exactly.
        self.assertEqual(result.intervals[0][0], 0.0)
        self.assertEqual(result.intervals[-1][1], 1.0)
        for left, right in zip(result.intervals, result.intervals[1:], strict=False):
            self.assertEqual(left[1], right[0])

    def test_subdivision_limit_reported_honestly(self) -> None:
        """Hitting the panel limit gives ``converged=False`` and a bound."""
        result = gauss_kronrod(peak, 0.0, 1.0, limit=5)
        self.assertFalse(result.converged)
        self.assertEqual(len(result.intervals), 5)
        self.assertEqual(result.evaluations, 15 * 9)
        self.assertGreater(result.error_estimate, 1e-10 * abs(result.value))
        self.assertGreaterEqual(result.error_estimate, abs(result.value - PEAK_EXACT))

    def test_level_limit_reported_honestly(self) -> None:
        """Unconverged tanh-sinh on the peak hands over to Gauss-Kronrod."""
        de = integrate(peak, 0.0, 1.0, method="double-exponential")
        self.assertFalse(de.converged)
        self.assertEqual(de.levels, 9)
        self.assertGreaterEqual(de.error_estimate, abs(de.value - PEAK_EXACT))
        auto = integrate(peak, 0.0, 1.0)
        self.assertEqual(auto.rule, "gauss-kronrod-15")
        self.assertTrue(auto.converged)
        self.assertGreater(auto.evaluations, de.evaluations)

    def test_interior_kink(self) -> None:
        """``|x - 0.3|`` defeats tanh-sinh but not adaptive bisection."""
        exact = 0.5 * (0.09 + 0.49)
        result = integrate(lambda x: abs(x - 0.3), 0.0, 1.0)
        self.assertEqual(result.rule, "gauss-kronrod-15")
        self.assertLess(abs(result.value - exact), 1e-12)
        self.assertGreaterEqual(result.error_estimate, abs(result.value - exact))

    def test_documented_limitations_are_flagged(self) -> None:
        """Oscillatory decay and singularities at non-zero ends stay uncertified."""
        sinc = integrate(lambda x: math.sin(x) / x if x else 1.0, 0.0, math.inf)
        self.assertFalse(sinc.converged)
        upper_singular = tanh_sinh_interval(lambda x: (1.0 - x) ** -0.5, 0.0, 1.0)
        self.assertFalse(upper_singular.converged)
        error = abs(upper_singular.value - 2.0)
        self.assertLess(error, 1e-7)
        self.assertGreaterEqual(upper_singular.error_estimate, error)
        shifted_end = tanh_sinh_interval(lambda x: (x - 2.0) ** -0.5, 2.0, 3.0)
        self.assertFalse(shifted_end.converged)
        self.assertGreaterEqual(
            shifted_end.error_estimate, abs(shifted_end.value - 2.0)
        )
        shifted = tanh_sinh_interval(lambda y: y**-0.5, 0.0, 1.0)
        self.assertTrue(shifted.converged)
        self.assertLessEqual(abs(shifted.value - 2.0), 4.0 * math.ulp(2.0))

    def test_absolute_tolerance_for_zero_integral(self) -> None:
        """An odd integrand needs an absolute target to certify zero."""
        relative = integrate(math.sin, -1.0, 1.0, method="gauss-kronrod")
        self.assertLess(abs(relative.value), 1e-16)
        absolute = integrate(math.sin, -1.0, 1.0, abs_tol=1e-12)
        self.assertTrue(absolute.converged)
        self.assertLess(abs(absolute.value), 1e-15)


class EvaluationPolicyTests(unittest.TestCase):
    """Non-finite values raise except at documented negligible-weight nodes."""

    def test_non_finite_inside_range_raises(self) -> None:
        """A NaN in the interior raises for every rule."""

        def bad(x: float) -> float:
            """Return NaN near ``x = 0.5`` and one elsewhere.

            References:
                Test integrand only. No library equation is evaluated.

            """
            return math.nan if 0.4 < x < 0.6 else 1.0

        for method in ("gauss-kronrod", "double-exponential", "auto"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                integrate(bad, 0.0, 1.0, method=method)
        with self.assertRaises(ValueError):
            sinh_sinh(lambda x: math.inf if abs(x) < 1.0 else 0.0)

    def test_overflow_in_moderate_tail_raises(self) -> None:
        """``cosh`` overflowing at ``x ~ 710`` is inside the safe range."""
        with self.assertRaises(OverflowError):
            exp_sinh(lambda x: math.exp(-x) * math.cosh(0.5 * x), 0.0, math.inf)
        stable = exp_sinh(
            lambda x: 0.5 * (math.exp(-0.5 * x) + math.exp(-1.5 * x)), 0.0, math.inf
        )
        self.assertAlmostEqual(stable.value, 4.0 / 3.0, places=14)

    def test_far_tail_nan_is_skipped(self) -> None:
        """``x * x * exp(-x)`` is NaN at ``x = 1e300`` yet integrates to 2."""
        self.assertTrue(math.isnan(square_exp(1.0e300)))
        result = exp_sinh(square_exp, 0.0, math.inf)
        self.assertTrue(result.converged)
        self.assertLessEqual(abs(result.value - 2.0), 4.0 * math.ulp(2.0))

    def test_endpoint_rounding_is_skipped(self) -> None:
        """Nodes rounding onto a singular end are dropped, not raised."""
        result = tanh_sinh_interval(
            lambda x: 1.0 / math.sqrt(x - 1.0e6), 1.0e6, 1.0e6 + 1.0
        )
        error = abs(result.value - 2.0)
        self.assertLess(error, 1e-4)
        self.assertFalse(result.converged)
        self.assertGreaterEqual(result.error_estimate, error)

    def test_argument_validation(self) -> None:
        """Invalid limits, tolerances, and options are rejected."""
        with self.assertRaises(ValueError):
            integrate(math.exp, 1.0, 0.0)
        with self.assertRaises(ValueError):
            integrate(math.exp, 0.0, math.nan)
        with self.assertRaises(ValueError):
            integrate(math.exp, 0.0, 1.0, method="simpson")
        with self.assertRaises(ValueError):
            gauss_kronrod(math.exp, 0.0, 1.0, rel_tol=1e-16)
        with self.assertRaises(ValueError):
            gauss_kronrod(math.exp, 0.0, 1.0, limit=0)
        with self.assertRaises(ValueError):
            tanh_sinh_interval(math.exp, 0.0, math.inf)
        with self.assertRaises(ValueError):
            exp_sinh(math.exp, 0.0, 1.0)
        with self.assertRaises(ValueError):
            sinh_sinh(math.exp, max_level=2)
        with self.assertRaises(TypeError):
            integrate(math.exp, "0", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            gauss_kronrod(lambda x: "1", 0.0, 1.0)  # type: ignore[arg-type,return-value]


class ReviewRegressionTests(unittest.TestCase):
    """Defects found in review: honest flags, overflow, and rule choice."""

    def test_auto_keeps_double_exponential_when_kronrod_misses_mass(self) -> None:
        """``exp(-(x - 50)^2)`` on the line: GK sees ``3e-198``, DE ``sqrt(pi)``.

        Neither rule converges; the old choice by smaller absolute error
        returned Gauss-Kronrod's ``2.9e-198``. Exact value ``sqrt(pi)``.
        """
        result = integrate(lambda x: math.exp(-((x - 50.0) ** 2)), -math.inf, math.inf)
        self.assertEqual(result.rule, "sinh-sinh")
        self.assertLess(abs(result.value - math.sqrt(math.pi)), 1e-12)
        self.assertFalse(result.converged)

    def test_auto_falls_back_when_double_exponential_raises(self) -> None:
        """``[-1e308, 1e308]`` overflows ``upper - lower``; GK still runs.

        Exact value ``2e307 atan(10)`` for ``1/(1 + (x/1e307)^2)``.
        """
        exact = 2.0e307 * math.atan(10.0)
        result = integrate(lambda x: 1.0 / (1.0 + (x / 1.0e307) ** 2), -1e308, 1e308)
        self.assertEqual(result.rule, "gauss-kronrod-15")
        self.assertTrue(result.converged)
        self.assertLess(abs(result.value - exact), 1e-12 * exact)

    def test_kronrod_width_overflow_is_never_converged_infinity(self) -> None:
        """Half-width ``0.5 (upper - lower)`` overflowed to ``inf``.

        The old code returned ``inf`` with ``converged=True``.
        """
        exact = 2.0e307 * math.atan(10.0)
        wide = gauss_kronrod(lambda x: 1.0 / (1.0 + (x / 1.0e307) ** 2), -1e308, 1e308)
        self.assertTrue(math.isfinite(wide.value))
        self.assertLess(abs(wide.value - exact), 1e-12 * exact)
        gaussian = gauss_kronrod(lambda x: math.exp(-x * x), -1e308, 1e308)
        self.assertTrue(math.isfinite(gaussian.value))
        self.assertTrue(math.isfinite(gaussian.error_estimate))

    def test_qagi_deep_bisection_reports_non_convergence(self) -> None:
        """``x^-1.01`` on ``[1, inf)`` with 5000 panels raised ZeroDivisionError.

        ``t * t`` underflowed for panels near ``t = 0``.
        """
        result = gauss_kronrod(lambda x: x**-1.01, 1.0, math.inf, limit=5000)
        self.assertFalse(result.converged)
        self.assertTrue(math.isfinite(result.value))

    def test_truncation_at_t_max_is_not_converged(self) -> None:
        """Slow algebraic tails beyond the last node were reported converged.

        ``int_1^inf x^-1.025 = 40`` (old estimate ``2.7e-9``, true error
        ``1.26e-6``); ``int_0^1 x^-0.97 = 100/3`` (true relative error
        ``9.3e-10`` above ``rel_tol``).
        """
        tail = exp_sinh(lambda x: x**-1.025, 1.0, math.inf, max_level=12)
        self.assertFalse(tail.converged)
        self.assertGreaterEqual(tail.error_estimate, abs(tail.value - 40.0))
        end = tanh_sinh_interval(lambda x: x**-0.97, 0.0, 1.0)
        self.assertFalse(end.converged)
        self.assertGreaterEqual(end.error_estimate, abs(end.value - 100.0 / 3.0))
        # Faster tails still converge to rounding.
        quick = exp_sinh(lambda x: x**-1.1, 1.0, math.inf)
        self.assertTrue(quick.converged)
        self.assertLess(abs(quick.value - 10.0), 1e-13)

    def test_rounding_term_spares_smooth_integrands_on_shifted_ranges(self) -> None:
        """A constant on ``[1e8, 1e8 + 1]`` was non-converged (estimate 5.5e-7).

        The node-rounding term now scales with the log-log slope of ``|f|``
        near the end, so ``(x - 2)^-1/2`` on ``[2, 3]`` stays flagged.
        """
        constant = tanh_sinh_interval(lambda x: 1.0, 1.0e8, 1.0e8 + 1.0)
        self.assertTrue(constant.converged)
        self.assertEqual(constant.value, 1.0)
        line = tanh_sinh_interval(lambda x: x, 1.0e8, 1.0e8 + 1.0)
        self.assertTrue(line.converged)
        self.assertLess(abs(line.value - 100000000.5), 1e-6)
        singular = tanh_sinh_interval(lambda x: (x - 2.0) ** -0.5, 2.0, 3.0)
        self.assertFalse(singular.converged)
        self.assertGreaterEqual(singular.error_estimate, abs(singular.value - 2.0))


if __name__ == "__main__":
    unittest.main()
