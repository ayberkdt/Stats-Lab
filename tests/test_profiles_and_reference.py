"""Tests for performance/data profiles and the multiple-precision references.

Reference values come from exact identities evaluated independently of the
algorithms under test (factorials, binomial sums, ``ln`` of exact rationals,
digit strings of ``pi``, ``e``, ``ln 2``, and Euler's constant) or from a
published table: ``erf(1)`` is OEIS A099286. ULP budgets for the float
kernels are those recorded in ADR-0077, tightened by ADR-0089.
"""

from __future__ import annotations

import functools
import math
import sys
import unittest
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import problems, report  # noqa: E402
from probstat.distributions import Normal  # noqa: E402
from probstat.distributions import _special as special  # noqa: E402
from probstat.numerics import (  # noqa: E402
    compare_to_reference,
    data_profile,
    exact_regularized_beta,
    first_converged_evaluation,
    performance_profile,
    performance_ratios,
    reference_erf,
    reference_erfc,
    reference_exp,
    reference_log,
    reference_log_gamma,
    reference_normal_cdf,
    reference_normal_quantile,
    reference_pi,
    reference_regularized_beta,
    reference_regularized_gamma_p,
    reference_regularized_gamma_q,
    ulp_error,
)
from probstat.numerics import reference as backend  # noqa: E402

PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
)
E = Decimal("2.71828182845904523536028747135266249775724709369995")
LN2 = Decimal("0.6931471805599453094172321214581765680755")
EULER_GAMMA = Decimal("0.57721566490153286060651209008240243104215933593992")
#: OEIS A099286, decimal expansion of erf(1) (105 digits).
ERF_ONE = Decimal(
    "0.842700792949714869341220635082609259296066997966302908459937897834717"
    "254096010841261983325348144888454158"
)


def close(value: Decimal, expected: Decimal, digits: int) -> bool:
    """Return whether two decimals agree to ``digits`` significant digits."""
    with localcontext() as context:
        context.prec = digits + 20
        if expected == 0:
            return abs(value) <= Decimal(10) ** -digits
        return abs(value - expected) <= abs(expected) * Decimal(10) ** (1 - digits)


def precise(test: Callable[[Any], None]) -> Callable[[Any], None]:
    """Run a test inside a 120-digit decimal context for its own arithmetic."""

    @functools.wraps(test)
    def wrapper(self: Any) -> None:
        """Call the test with the raised precision."""
        with localcontext() as context:
            context.prec = 120
            test(self)

    return wrapper


def grid(low: float, high: float, count: int) -> list[float]:
    """Return ``count`` equally spaced points on ``[low, high]``."""
    return [low + (high - low) * i / (count - 1) for i in range(count)]


class PerformanceProfileTests(unittest.TestCase):
    """Exact profile values on a hand-made cost table."""

    costs = {
        "A": [1.0, 2.0, 4.0, None],
        "B": [2.0, 2.0, 1.0, 8.0],
        "C": [None, math.inf, math.nan, None],
    }

    def test_ratios_ties_and_failures(self) -> None:
        """Ratios divide by the best cost; ties share 1; failures are inf."""
        ratios = performance_ratios(self.costs)
        self.assertEqual(ratios["A"], (1.0, 1.0, 4.0, math.inf))
        self.assertEqual(ratios["B"], (2.0, 1.0, 1.0, 1.0))
        self.assertTrue(all(r == math.inf for r in ratios["C"]))
        unsolved = performance_ratios({"x": [None, 1.0], "y": [None, 3.0]})
        self.assertEqual(unsolved["x"][0], math.inf)
        self.assertEqual(unsolved["y"], (math.inf, 3.0))

    def test_profile_values_wins_and_quantiles(self) -> None:
        """The profile is the exact step function; rho(1) is the win fraction."""
        profile = performance_profile(self.costs)
        self.assertEqual(profile.problems, 4)
        self.assertEqual(profile.max_ratio, 4.0)
        expected = {
            "A": {0.5: 0.0, 1.0: 0.5, 3.99: 0.5, 4.0: 0.75, 1e9: 0.75},
            "B": {0.5: 0.0, 1.0: 0.75, 1.99: 0.75, 2.0: 1.0},
            "C": {1.0: 0.0, 1e300: 0.0},
        }
        for solver, values in expected.items():
            for tau, share in values.items():
                self.assertEqual(profile.value(solver, tau), share, (solver, tau))
        for solver, row in self.costs.items():
            wins = sum(
                1
                for p, cost in enumerate(row)
                if cost is not None
                and math.isfinite(cost)
                and cost
                == min(
                    c[p] for c in self.costs.values()
                    if c[p] is not None and math.isfinite(c[p])
                )
            )  # fmt: skip
            self.assertEqual(profile.wins(solver), wins / 4)
        self.assertEqual(profile.solved_fraction("A"), 0.75)
        self.assertEqual(profile.solved_fraction("C"), 0.0)
        self.assertEqual(profile.tau_at("A", 0.5), 1.0)
        self.assertEqual(profile.tau_at("A", 0.75), 4.0)
        self.assertEqual(profile.tau_at("A", 1.0), math.inf)
        self.assertEqual(profile.tau_at("B", 1.0), 2.0)
        self.assertEqual(profile.breakpoints("A"), ((1.0, 0.5), (4.0, 0.75)))
        self.assertEqual(profile.breakpoints("C"), ())

    def test_area_is_the_exact_log2_integral(self) -> None:
        """The area equals (1/(n L)) sum (L - log2 r) and rho(1) at tau_max 1."""
        profile = performance_profile(self.costs)
        self.assertAlmostEqual(profile.area("A"), (2 + 2 + 0) / (4 * 2), places=15)
        self.assertAlmostEqual(profile.area("B"), (2 + 2 + 2 + 1) / 8, places=15)
        self.assertEqual(profile.area("C"), 0.0)
        self.assertAlmostEqual(profile.area("A", 16.0), (4 + 4 + 2) / 16, places=15)
        self.assertEqual(profile.area("B", 1.0), profile.wins("B"))
        tied = performance_profile({"x": [3.0, 5.0], "y": [3.0, 5.0]})
        self.assertEqual(tied.area("x"), 1.0)
        self.assertEqual(tied.wins("y"), 1.0)

    def test_invalid_tables_and_arguments_raise(self) -> None:
        """Empty, ragged, non-positive, and non-real inputs are rejected."""
        for bad in ({}, {"a": []}, {"a": [1.0], "b": [1.0, 2.0]}):
            with self.assertRaises(ValueError):
                performance_profile(bad)
        for value in (0.0, -1.0, -math.inf):
            with self.assertRaises(ValueError):
                performance_profile({"a": [value]})
        for value in (True, "1", object()):
            with self.assertRaises(TypeError):
                performance_profile({"a": [value]})  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            performance_profile({"a": "12"})  # type: ignore[dict-item]
        profile = performance_profile(self.costs)
        for fraction in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                profile.tau_at("A", fraction)
        with self.assertRaises(TypeError):
            profile.tau_at("A", True)
        with self.assertRaises(ValueError):
            profile.area("A", 0.5)
        with self.assertRaises(KeyError):
            profile.value("Z", 1.0)


class DataProfileTests(unittest.TestCase):
    """Data profiles divide by n_p + 1 and never decrease."""

    def test_budgets_values_and_monotonicity(self) -> None:
        """d_s(kappa) counts problems solved within kappa (n_p + 1) evaluations."""
        profile = data_profile(
            {"A": [10, 30, None], "B": [40, 3, 100]}, dimensions=[1, 2, 4]
        )
        self.assertEqual(profile.budgets[0], (5.0, 10.0, math.inf))
        self.assertEqual(profile.budgets[1], (20.0, 1.0, 20.0))
        self.assertEqual(profile.value("A", 4.99), 0.0)
        self.assertEqual(profile.value("A", 5.0), 1 / 3)
        self.assertEqual(profile.value("A", 10.0), 2 / 3)
        self.assertEqual(profile.value("B", 20.0), 1.0)
        self.assertEqual(profile.kappa_at("A", 1.0), math.inf)
        self.assertEqual(profile.kappa_at("B", 0.5), 20.0)
        self.assertEqual(profile.breakpoints("B"), ((1.0, 1 / 3), (20.0, 1.0)))
        for solver in profile.solvers:
            values = [profile.value(solver, 2.0**k) for k in range(-2, 12)]
            self.assertEqual(values, sorted(values))
        raw = data_profile({"A": [10, 30, None]})
        self.assertEqual(raw.budgets[0], (10.0, 30.0, math.inf))

    def test_invalid_dimensions_raise(self) -> None:
        """Dimensions must match the problems and be non-negative integers."""
        with self.assertRaises(ValueError):
            data_profile({"A": [1, 2]}, dimensions=[1])
        with self.assertRaises(ValueError):
            data_profile({"A": [1, 2]}, dimensions=[1, -1])
        with self.assertRaises(TypeError):
            data_profile({"A": [1, 2]}, dimensions=[1, 2.5])  # type: ignore[list-item]

    def test_more_wild_convergence_test(self) -> None:
        """The first point with f0 - f >= (1 - tau)(f0 - fL) is reported."""
        values = [10.0, 8.0, 5.0, 2.0, 1.0]
        self.assertEqual(first_converged_evaluation(values, best=0.0, tolerance=0.5), 3)
        self.assertEqual(first_converged_evaluation(values, best=0.0, tolerance=0.1), 5)
        self.assertIsNone(
            first_converged_evaluation(values, best=0.0, tolerance=0.05)
        )
        for tolerance in (0.0, 1.0):
            with self.assertRaises(ValueError):
                first_converged_evaluation(values, best=0.0, tolerance=tolerance)
        with self.assertRaises(ValueError):
            first_converged_evaluation([], best=0.0, tolerance=0.5)
        with self.assertRaises(ValueError):
            first_converged_evaluation([1.0, math.nan], best=0.0, tolerance=0.5)


class BenchmarkProfileSectionTests(unittest.TestCase):
    """The report turns recorded evaluations and errors into profiles."""

    def summary(self) -> list[dict[str, object]]:
        """Return synthetic propagation summary rows."""
        rows: list[dict[str, object]] = []
        table = {
            "MC": {"mean": (0.2, 100), "variance": (0.01, 100)},
            "QMC": {"mean": (0.01, 50), "variance": (0.02, 50)},
            "PCE": {"mean": (0.001, 200), "variance": (0.001, 200)},
            "Closed form": {"mean": (0.0, 0), "variance": (0.0, 0)},
        }
        for problem in ("ishigami-moments", "exponential-5d"):
            for method, groups in table.items():
                for group, (error, evaluations) in groups.items():
                    rows.append(
                        {
                            "problem": problem,
                            "category": "propagation",
                            "method": method,
                            "kind": "estimator",
                            "group": group,
                            "error": error,
                            "evaluations": evaluations,
                        }
                    )
        return rows

    def test_costs_failures_and_rendering(self) -> None:
        """Inaccurate runs fail; zero-cost methods are listed, not profiled."""
        chosen = [p for p in problems.registry() if p.category == "propagation"]
        profiles = report.category_profiles(chosen, self.summary(), "propagation")
        assert profiles is not None
        self.assertEqual(len(profiles.instances), 4)
        self.assertEqual(profiles.excluded, ("Closed form",))
        performance = profiles.performance
        self.assertEqual(performance.solvers, ("MC", "QMC", "PCE"))
        # MC is inaccurate on the means (0.2 > 5 %), QMC on nothing.
        self.assertEqual(performance.ratios[0], (math.inf, 2.0, math.inf, 2.0))
        self.assertEqual(performance.ratios[1], (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(performance.ratios[2], (4.0, 4.0, 4.0, 4.0))
        self.assertEqual(performance.wins("QMC"), 1.0)
        dimension = {p.key: p.dimension for p in chosen}["ishigami-moments"]
        self.assertEqual(profiles.data.budgets[1][0], 50 / (dimension + 1))
        lines = report.profile_section(profiles, ("light.png", "dark.png"))
        text = "\n".join(lines)
        self.assertIn("### Performance profiles", text)
        self.assertIn('srcset="dark.png"', text)
        self.assertIn("Closed form", text)
        for line in lines:
            if line.startswith("|"):
                self.assertLessEqual(line.count("|") - 1, 7, line)

    def test_profile_figure_light_and_dark(self) -> None:
        """The figure renders headless in both palettes."""
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib is not installed")
        matplotlib.use("Agg")
        from probstat.visualization.profile_plots import plot_performance_profile
        from probstat.visualization.theme import (
            GRAPHITE,
            PORCELAIN,
            chart_theme,
            pyplot,
        )

        profile = performance_profile(PerformanceProfileTests.costs)
        for palette in (PORCELAIN, GRAPHITE):
            with chart_theme(palette):
                axes = plot_performance_profile(profile)
                self.assertEqual(axes.get_xlim()[0], 1.0)
                self.assertEqual(axes.get_ylim(), (0.0, 1.04))
                self.assertEqual(axes.figure.get_facecolor()[:3],
                                 matplotlib.colors.to_rgb(palette.canvas))  # fmt: skip
                pyplot().close(axes.figure)
        with self.assertRaises(ValueError):
            plot_performance_profile(profile, highlight=0)
        with self.assertRaises(TypeError):
            plot_performance_profile(object())  # type: ignore[arg-type]


class ReferenceValueTests(unittest.TestCase):
    """Reference functions reproduce known high-precision constants."""

    @precise
    def test_constants_exp_and_log(self) -> None:
        """pi, e, and ln 2 match their digit strings."""
        self.assertTrue(close(reference_pi(digits=100), PI, 100))
        self.assertTrue(close(reference_exp(1, digits=50), E, 50))
        self.assertTrue(close(reference_log(2, digits=40), LN2, 40))
        self.assertEqual(reference_exp(0, digits=10), 1)

    def test_digits_follow_the_active_context(self) -> None:
        """Without ``digits`` the result has the context's precision."""
        with localcontext() as context:
            context.prec = 25
            value = reference_erf(1)
        self.assertEqual(len(value.as_tuple().digits), 25)
        self.assertTrue(close(value, ERF_ONE, 25))

    @precise
    def test_error_function_against_oeis_and_identities(self) -> None:
        """erf(1) to 100 digits; erfc and incomplete gamma agree independently."""
        self.assertTrue(close(reference_erf(1, digits=100), ERF_ONE, 100))
        self.assertTrue(close(reference_erf(-1, digits=60), -ERF_ONE, 60))
        self.assertTrue(close(reference_erfc(1, digits=60), 1 - ERF_ONE, 60))
        self.assertTrue(close(reference_regularized_gamma_p("0.5", 1, digits=60),
                              ERF_ONE, 60))  # fmt: skip
        self.assertEqual(reference_erf(0, digits=20), 0)
        for x in (2, "3.5", 6, 12, 27):
            argument = Decimal(x)
            erfc = reference_erfc(argument, digits=45)
            gamma_q = reference_regularized_gamma_q("0.5", argument * argument,
                                                    digits=45)  # fmt: skip
            self.assertTrue(close(erfc, gamma_q, 43), x)
            erf = reference_erf(argument, digits=45)
            self.assertTrue(close(erf + erfc, Decimal(1), 44), x)

    @precise
    def test_normal_cdf_and_quantile(self) -> None:
        """Phi(-10) by two algorithms; quantile round trips in both tails."""
        tail = reference_normal_cdf(-10, digits=40)
        gamma_q = reference_regularized_gamma_q("0.5", 50, digits=40) / 2
        self.assertTrue(close(tail, gamma_q, 38))
        self.assertAlmostEqual(float(tail) / (0.5 * math.erfc(10 / math.sqrt(2))),
                               1.0, places=13)  # fmt: skip
        self.assertEqual(reference_normal_cdf(0, digits=30), Decimal("0.5"))
        self.assertTrue(close(reference_normal_cdf(-40, digits=30)
                              + reference_normal_cdf(40, digits=60), Decimal(1),
                              58))  # fmt: skip
        for p in (Fraction(1, 10**300), "1e-30", 0.025, "0.49999999999999999999",
                  0.5, 0.975, 1 - 2**-50):  # fmt: skip
            with localcontext() as context:
                context.prec = 60
                quantile = reference_normal_quantile(p, digits=45)
                if isinstance(p, str):
                    expected = Decimal(p)
                else:
                    ratio = Fraction(p)
                    expected = Decimal(ratio.numerator) / Decimal(ratio.denominator)
                if p == 0.5:
                    self.assertEqual(quantile, 0)
                    continue
                self.assertTrue(close(reference_normal_cdf(quantile, digits=50),
                                      expected, 40), p)  # fmt: skip
        self.assertAlmostEqual(float(reference_normal_quantile(Fraction(39, 40))),
                               1.959963984540054, places=15)  # fmt: skip

    @precise
    def test_log_gamma_identities(self) -> None:
        """Gamma(1/2) = sqrt(pi), half-integer and integer factorial forms."""
        with localcontext() as context:
            context.prec = 80
            half = PI.ln() / 2
            value = Fraction(math.factorial(200), 4**100 * math.factorial(100))
            hundred = (Decimal(value.numerator) / Decimal(value.denominator)).ln()
            hundred += half
            factorial = Decimal(math.factorial(49)).ln()
        self.assertTrue(close(reference_log_gamma("0.5", digits=60), half, 60))
        self.assertTrue(close(reference_log_gamma(100.5, digits=60), hundred, 60))
        self.assertTrue(close(reference_log_gamma(50, digits=60), factorial, 60))
        self.assertEqual(reference_log_gamma(1), 0)
        self.assertEqual(reference_log_gamma(2), 0)
        self.assertTrue(close(reference_log_gamma(3, digits=40), LN2, 40))

    @precise
    def test_log_gamma_keeps_relative_accuracy_near_one(self) -> None:
        """Near 1, log Gamma(1 + e) = -gamma e + (pi^2/12) e^2 + O(e^3)."""
        epsilon = Decimal(2) ** -40
        value = reference_log_gamma(1 + epsilon, digits=40)
        with localcontext() as context:
            context.prec = 60
            curvature = (value + EULER_GAMMA * epsilon) / (epsilon * epsilon)
            self.assertLess(abs(curvature - PI * PI / 12), Decimal("1e-10"))

    @precise
    def test_incomplete_gamma_integer_shapes(self) -> None:
        """Q(n, x) = exp(-x) sum_(k<n) x^k / k! to 40 digits, tails included."""
        for n, x in ((1, "0.5"), (3, 10), (5, 2), (20, 35), (3, 100), (40, 4)):
            with localcontext() as context:
                context.prec = 80
                point = Decimal(x)
                series = sum(point**k / math.factorial(k) for k in range(n))
                exact_q = (-point).exp() * series
                exact_p = 1 - exact_q
            self.assertTrue(close(reference_regularized_gamma_q(n, x, digits=40),
                                  exact_q, 39), (n, x))  # fmt: skip
            if exact_p > Decimal("1e-60"):
                self.assertTrue(close(reference_regularized_gamma_p(n, x, digits=40),
                                      exact_p, 39), (n, x))  # fmt: skip
        self.assertEqual(reference_regularized_gamma_p(2, 0), 0)
        self.assertEqual(reference_regularized_gamma_q(2, 0), 1)

    @precise
    def test_incomplete_gamma_branches_agree(self) -> None:
        """The series complement and the continued fraction coincide."""
        for a, x in (("2.5", "4"), ("10.25", "12"), ("0.3", "1.5"), ("60", "75")):
            shape, point = Decimal(a), Decimal(x)
            series = backend._gamma_series_p(shape, point, 70)
            fraction = backend._gamma_fraction_q(shape, point, 50)
            with localcontext() as context:
                context.prec = 70
                self.assertTrue(close(1 - series, fraction, 45), (a, x))

    @precise
    def test_incomplete_beta_exact_cases(self) -> None:
        """Integer shapes match the binomial sum; closed forms elsewhere."""
        for x, a, b in ((0.3, 2, 3), (0.5, 10, 10), (0.95, 30, 2), (0.01, 3, 40),
                        (Fraction(1, 3), 1, 1), (0.999, 4, 7)):  # fmt: skip
            exact = exact_regularized_beta(x, a, b)
            expected = Decimal(exact.numerator) / Decimal(exact.denominator)
            self.assertTrue(close(reference_regularized_beta(x, a, b, digits=40),
                                  expected, 39), (x, a, b))  # fmt: skip
        self.assertEqual(exact_regularized_beta(Fraction(1, 2), 10, 10), Fraction(1, 2))
        symmetric = reference_regularized_beta("0.5", "7.3", "7.3", digits=40)
        self.assertTrue(close(symmetric, Decimal("0.5"), 40))
        power = reference_exp(Decimal("2.5") * Decimal("0.3").ln(), digits=40)
        self.assertTrue(close(reference_regularized_beta("0.3", "2.5", 1, digits=40),
                              power, 39))  # fmt: skip
        self.assertEqual(reference_regularized_beta(0, 2, 3), 0)
        self.assertEqual(reference_regularized_beta(1, 2, 3), 1)


class ReferenceValidationTests(unittest.TestCase):
    """Invalid inputs raise instead of returning numbers."""

    def test_invalid_arguments(self) -> None:
        """Non-finite, wrongly typed, and out-of-domain inputs are rejected."""
        for value in (math.nan, math.inf, "abc", "NaN", Decimal("Infinity")):
            with self.assertRaises(ValueError):
                reference_erf(value)
        for value in (True, object(), None):
            with self.assertRaises(TypeError):
                reference_erf(value)  # type: ignore[arg-type]
        for digits in (0, -3, 10**6):
            with self.assertRaises(ValueError):
                reference_pi(digits=digits)
        with self.assertRaises(TypeError):
            reference_pi(digits=2.5)  # type: ignore[arg-type]
        for call in (
            lambda: reference_log(0),
            lambda: reference_log_gamma(0),
            lambda: reference_log_gamma(-1.5),
            lambda: reference_regularized_gamma_p(0, 1),
            lambda: reference_regularized_gamma_q(1, -1),
            lambda: reference_regularized_beta(1.5, 1, 1),
            lambda: reference_regularized_beta(0.5, -1, 1),
            lambda: reference_normal_quantile(0),
            lambda: reference_normal_quantile(1),
            lambda: exact_regularized_beta(2, 1, 1),
            lambda: exact_regularized_beta(0.5, 0, 1),
        ):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            exact_regularized_beta(0.5, 1.5, 1)  # type: ignore[arg-type]
        with self.assertRaises(OverflowError):
            reference_exp(10**7)

    def test_ulp_measurement(self) -> None:
        """ULP error is exact on simple cases and the harness validates input."""
        self.assertEqual(ulp_error(1.0, Decimal(1)), 0.0)
        self.assertEqual(ulp_error(1.0 + 2**-52, Decimal(1)), 1.0)
        self.assertEqual(ulp_error(math.nan, Decimal(1)), math.inf)
        with localcontext() as context:
            context.prec = 40
            root = Decimal(2).sqrt()
        self.assertLessEqual(ulp_error(math.sqrt(2.0), root), 0.5)
        with self.assertRaises(ValueError):
            ulp_error(1.0, Decimal("1e400"))
        with self.assertRaises(ValueError):
            compare_to_reference(math.sqrt, lambda x: Decimal(x).sqrt(), [])
        with self.assertRaises(ValueError):
            compare_to_reference(math.sqrt, lambda x: Decimal(x).sqrt(), [2.0],
                                 digits=10)  # fmt: skip
        with self.assertRaises(TypeError):
            compare_to_reference(math.sqrt, math.sqrt, [2.0])  # type: ignore[arg-type]
        result = compare_to_reference(math.sqrt, lambda x: Decimal(x).sqrt(),
                                      grid(0.5, 100.0, 50))  # fmt: skip
        self.assertLessEqual(result.max_ulp, 0.5)
        self.assertTrue(result.within(0.5))
        self.assertLessEqual(result.max_relative_error, 2**-53)


class FloatKernelBudgetTests(unittest.TestCase):
    """Library float kernels stay within the ULP budgets of ADR-0077/ADR-0089."""

    def test_normal_cdf_and_survival(self) -> None:
        """Error <= 8 ULP over the whole range (ADR-0089); <= 4 for |x| <= 2.

        ADR-0077 allowed ``2 (1 + x^2)`` ULP for the rounded argument
        ``x/sqrt 2`` (1228 ULP at ``x = -37``); the double-double argument
        removes that growth.
        """
        normal = Normal()
        points = grid(-37.5, 8.0, 93)
        result = compare_to_reference(normal.cdf, reference_normal_cdf, points)
        for (x,), error in zip(result.points, result.ulp_errors, strict=True):
            self.assertLessEqual(error, 8.0, x)
            if abs(x) <= 2.0:
                self.assertLessEqual(error, 4.0, x)
        survival = compare_to_reference(
            normal.sf, lambda x: reference_normal_cdf(-Decimal(x)), grid(5, 37.5, 14)
        )
        for (x,), error in zip(survival.points, survival.ulp_errors, strict=True):
            self.assertLessEqual(error, 8.0, x)

    def test_normal_quantile_and_inverse_survival(self) -> None:
        """Quantile and inverse survival within 5 ULP down to p = 1e-300."""
        normal = Normal()
        tails = [10.0**-k for k in range(2, 301, 14)]
        points = grid(0.01, 0.99, 25) + tails + [1.0 - 10.0**-k for k in (3, 8, 15)]
        self.assertTrue(
            compare_to_reference(normal.quantile, reference_normal_quantile,
                                 points).within(5.0)  # fmt: skip
        )
        self.assertTrue(
            compare_to_reference(
                normal.isf, lambda q: reference_normal_quantile(1 - Fraction(q)), tails
            ).within(5.0)
        )

    def test_log_gamma(self) -> None:
        """math.lgamma: 32 ULP away from the zeros, 2e-15 absolute near them."""
        away = grid(0.01, 0.8, 20) + grid(3.0, 30.0, 19) + [171.3, 1000.5, 1e15]
        self.assertTrue(
            compare_to_reference(math.lgamma, reference_log_gamma, away).within(32.0)
        )
        near = compare_to_reference(math.lgamma, reference_log_gamma,
                                    grid(0.8, 3.0, 45), digits=30)  # fmt: skip
        for value, exact in zip(near.values, near.references, strict=True):
            self.assertLessEqual(abs(value - float(exact)), 2e-15)
        pairs = [(a, b) for a in (0.5, 1.5, 3.0, 20.0, 150.0)
                 for b in (0.7, 2.5, 40.0, 400.0)]  # fmt: skip
        log_beta = compare_to_reference(
            special.log_beta_function,
            lambda a, b: reference_log_gamma(a) + reference_log_gamma(b)
            - reference_log_gamma(Decimal(a) + Decimal(b)),
            pairs,
        )
        self.assertTrue(log_beta.within(256.0), log_beta.worst_point)

    def test_regularized_gamma(self) -> None:
        """P and Q: 32 ULP for a <= 10 and 16 ULP for a <= 200, tails included.

        ADR-0077 budgets were 256 and 4096 ULP; ADR-0089 forms the prefactor
        exponent in double-double arithmetic and uses Temme's expansion for
        ``a >= 20`` near ``x = a``.
        """
        factors = (0.01, 0.1, 0.3, 0.8, 1.0, 1.05)
        points = [(a, f * a) for a in (0.05, 0.5, 1.0, 2.5, 7.5, 10.0, 30.0, 99.0,
                                       200.0) for f in factors]  # fmt: skip
        points += [(a, s * a + c) for a in (0.5, 2.5, 10.0, 55.0, 200.0)
                   for s, c in ((1.3, 0.5), (2.0, 1.0), (3.0, 5.0))]  # fmt: skip
        for function, reference in (
            (special.regularized_gamma_p, reference_regularized_gamma_p),
            (special.regularized_gamma_q, reference_regularized_gamma_q),
        ):
            result = compare_to_reference(function, reference, points, digits=30)
            for (a, x), error in zip(result.points, result.ulp_errors, strict=True):
                self.assertLessEqual(error, 32.0 if a <= 10.0 else 16.0, (a, x))

    def test_regularized_beta(self) -> None:
        """I_x(a, b): 16 ULP for shapes <= 10 and 64 ULP up to 400 (ADR-0089).

        ADR-0077 budgets were 256 and 2048 ULP (770 ULP measured at
        ``(0.05, 300, 400)``, from the ``lgamma`` prefactor).
        """
        shapes = ((0.5, 0.5), (1, 1), (2, 3), (0.7, 5.5), (5.5, 0.7), (10, 10),
                  (30, 2), (2.5, 40), (150, 90), (300, 400))  # fmt: skip
        points = [(x, a, b) for a, b in shapes
                  for x in (0.001, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999)]  # fmt: skip
        result = compare_to_reference(special.regularized_beta,
                                      reference_regularized_beta, points,
                                      digits=30)  # fmt: skip
        for (x, a, b), error in zip(result.points, result.ulp_errors, strict=True):
            budget = 16.0 if max(a, b) <= 10 else 64.0
            self.assertLessEqual(error, budget, (x, a, b))


if __name__ == "__main__":
    unittest.main()
