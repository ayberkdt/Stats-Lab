"""Sobol estimator choice, quadratic-sum importance factors, and Morris ``mu``."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal, Uniform
from probstat.modeling import (
    FLOOD_INPUT_NAMES,
    FLOOD_NOMINAL,
    FLOOD_NOMINAL_STANDARD_DEVIATIONS,
    ISHIGAMI_FIRST_ORDER,
    ISHIGAMI_TOTAL,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    flood_model,
    ishigami,
    ishigami_model,
)
from probstat.sensitivity import (
    SOBOL_ESTIMATORS,
    estimate_sobol_indices,
    morris_screening,
    quadratic_sum_importance,
    sobol_analysis,
    sobol_analysis_all,
    taylor_importance_factors,
)
from probstat.uq import QuantityOfInterest

ALTERNATIVES = tuple(name for name in SOBOL_ESTIMATORS if name != "saltelli2010")

# Values produced by the code before the estimator option existed; the
# default estimator must keep reproducing them bit for bit.
PINNED_FIRST = (0.3979518225732304, 0.45113650985614406, -0.3431217468129877)
PINNED_TOTAL = (0.46126028564858024, 0.3234279504260432, 0.22235673759866237)
PINNED_SECOND_13 = 0.6412806633180883
PINNED_VARIANCE = 16.45797461147934
PINNED_MODEL_FIRST = (0.3009140676420575, 0.5600646769479116, -0.056672237550094776)
PINNED_MODEL_TOTAL = (0.5005559642102588, 0.4250775680841037, 0.22424928385027879)
PINNED_BOOT_FIRST_0 = (0.20115460057543544, 0.11001251202089947, 0.2988881147194848)
PINNED_REPL_TOTAL_2 = (0.2987283846742283, 0.20580966370533066, 0.39164710564312594)


def _uniform_matrices(rows: int, dimension: int, seed: int):
    """Return two independent uniform ``[-pi, pi]`` matrices."""
    rng = random.Random(seed)
    a = [
        [rng.uniform(-math.pi, math.pi) for _ in range(dimension)] for _ in range(rows)
    ]
    b = [
        [rng.uniform(-math.pi, math.pi) for _ in range(dimension)] for _ in range(rows)
    ]
    return a, b


def _linear_model(weights, law) -> ProbabilisticModel:
    """Return ``y = sum w_i x_i`` with independent inputs of one law."""
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", law) for i in range(len(weights)))
    )

    def response(p):
        """Evaluate the weighted sum."""
        return {"y": math.fsum(w * p[f"x{i}"] for i, w in enumerate(weights))}

    return ProbabilisticModel(inputs, response, (QuantityOfInterest("y", "sum"),))


def _flood_gradient(x):
    """Return the exact gradient of the flood overflow ``S``."""
    q, ks, zv, zm, _hd, _cb, length, width = x
    height = (q / (ks * width * math.sqrt((zm - zv) / length))) ** 0.6
    gap = zm - zv
    return (
        0.6 * height / q,
        -0.6 * height / ks,
        1.0 + 0.3 * height / gap,
        -0.3 * height / gap,
        -1.0,
        -1.0,
        0.3 * height / length,
        -0.6 * height / width,
    )


class DefaultEstimatorUnchangedTests(unittest.TestCase):
    """The default estimator reproduces the numbers pinned before the change."""

    def test_estimate_sobol_indices_default_is_bitwise_unchanged(self) -> None:
        """Plain pick-freeze results equal the pinned values exactly."""
        a, b = _uniform_matrices(16, 3, 7)
        for kwargs in ({}, {"estimator": "saltelli2010"}):
            result = estimate_sobol_indices(lambda x: ishigami(*x), a, b, **kwargs)
            self.assertEqual(result.first_order, PINNED_FIRST)
            self.assertEqual(result.total_effect, PINNED_TOTAL)
            assert result.second_order is not None
            self.assertEqual(result.second_order[0][2], PINNED_SECOND_13)
            self.assertEqual(result.output_variance, PINNED_VARIANCE)
            self.assertEqual(result.evaluations, 128)

    def assert_close(
        self, actual: tuple[float, ...], expected: tuple[float, ...]
    ) -> None:
        """Compare to a relative 1e-12, the libm difference between platforms.

        The model-based values go through ``math.sin`` and the input
        transforms, whose last bits differ between the Windows and Linux C
        libraries; equality would test the platform, not the estimator.

        References:
            [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

        """
        self.assertEqual(len(actual), len(expected))
        for value, pinned in zip(actual, expected, strict=True):
            self.assertTrue(
                math.isclose(value, pinned, rel_tol=1e-12, abs_tol=1e-15),
                f"{value!r} != {pinned!r}",
            )

    def test_model_based_default_is_unchanged(self) -> None:
        """Model-based estimates and both interval methods are unchanged."""
        model = ishigami_model()
        plain = sobol_analysis(model, "y", 64, seed=3)
        self.assert_close(plain.result.first_order, PINNED_MODEL_FIRST)
        self.assert_close(plain.result.total_effect, PINNED_MODEL_TOTAL)
        self.assertEqual(plain.estimator, "saltelli2010")
        boot = sobol_analysis(model, "y", 64, design="mc", seed=3, bootstrap=20)
        first = boot.first_order[0]
        self.assert_close(
            (first.estimate, first.lower, first.upper), PINNED_BOOT_FIRST_0
        )
        repl = sobol_analysis(model, "y", 64, seed=3, bootstrap=1)
        total = repl.total_effect[2]
        self.assert_close(
            (total.estimate, total.lower, total.upper), PINNED_REPL_TOTAL_2
        )


class EstimatorAccuracyTests(unittest.TestCase):
    """Every estimator converges to analytic indices."""

    def test_ishigami_replicate_intervals_cover_analytic_indices(self) -> None:
        """99.9% replicate intervals of every estimator cover the truth."""
        model = ishigami_model()
        slack = 1e-4  # the analytic constants are rounded to four decimals
        for name in SOBOL_ESTIMATORS:
            with self.subTest(estimator=name):
                result = sobol_analysis(
                    model,
                    "y",
                    4096,
                    seed=2026,
                    replicates=8,
                    confidence=0.999,
                    estimator=name,
                )
                self.assertEqual(result.estimator, name)
                self.assertEqual(result.interval_method, "replicates")
                pairs = zip(
                    result.first_order + result.total_effect,
                    ISHIGAMI_FIRST_ORDER + ISHIGAMI_TOTAL,
                    strict=True,
                )
                for interval, truth in pairs:
                    self.assertLessEqual(interval.lower - slack, truth)
                    self.assertGreaterEqual(interval.upper + slack, truth)
                    self.assertLess(interval.upper - interval.lower, 0.1)

    def test_linear_additive_model_closed_form(self) -> None:
        """``S_i = S_Ti = w_i^2 / sum w_j^2`` within replicate error bars."""
        weights = (1.0, 2.0, 3.0)
        truth = tuple(w * w / 14.0 for w in weights)
        model = _linear_model(weights, Uniform(0.0, 1.0))
        for name in SOBOL_ESTIMATORS:
            with self.subTest(estimator=name):
                result = sobol_analysis(
                    model,
                    "y",
                    4096,
                    design="mc",
                    seed=11,
                    replicates=8,
                    confidence=0.999,
                    estimator=name,
                )
                for intervals in (result.first_order, result.total_effect):
                    for interval, expected in zip(intervals, truth, strict=True):
                        self.assertLessEqual(interval.lower, expected)
                        self.assertGreaterEqual(interval.upper, expected)

    def test_linear_model_second_order_indices_vanish(self) -> None:
        """Pair interactions of an additive model are zero for every estimator."""
        rng = random.Random(5)
        a = [[rng.random() for _ in range(3)] for _ in range(2000)]
        b = [[rng.random() for _ in range(3)] for _ in range(2000)]
        for name in SOBOL_ESTIMATORS:
            with self.subTest(estimator=name):
                result = estimate_sobol_indices(
                    lambda x: x[0] + 2.0 * x[1] + 3.0 * x[2], a, b, estimator=name
                )
                self.assertEqual(result.evaluations, 2000 * 5 + 2000 * 3)
                assert result.second_order is not None
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            self.assertLess(abs(result.second_order[i][j]), 0.1)


class EstimatorConsistencyTests(unittest.TestCase):
    """Estimators agree with one another on one design."""

    def test_bootstrap_intervals_of_all_estimators_overlap(self) -> None:
        """Every estimator's bootstrap interval shares a point with the others."""
        model = ishigami_model()
        results = {
            name: sobol_analysis(
                model,
                "y",
                1024,
                design="mc",
                seed=4,
                bootstrap=60,
                confidence=0.99,
                estimator=name,
            )
            for name in SOBOL_ESTIMATORS
        }
        evaluations = {result.evaluations for result in results.values()}
        self.assertEqual(evaluations, {1024 * 5})
        for result in results.values():
            self.assertEqual(result.interval_method, "bootstrap")
        for index in range(3):
            for kind in ("first_order", "total_effect"):
                intervals = [getattr(r, kind)[index] for r in results.values()]
                lower = max(item.lower for item in intervals)
                upper = min(item.upper for item in intervals)
                self.assertLessEqual(lower, upper, f"{kind}[{index}]")

    def test_shift_invariance_and_sensitivity_to_the_output_mean(self) -> None:
        """Centred estimators ignore a constant offset; product forms do not."""
        a, b = _uniform_matrices(512, 3, 9)

        def shifted(x):
            """Return the Ishigami value plus a large constant."""
            return ishigami(*x) + 1.0e4

        for name in ("saltelli2010", "jansen", "martinez", "janon"):
            base = estimate_sobol_indices(lambda x: ishigami(*x), a, b, estimator=name)
            moved = estimate_sobol_indices(shifted, a, b, estimator=name)
            for left, right in zip(
                base.first_order + base.total_effect,
                moved.first_order + moved.total_effect,
                strict=True,
            ):
                self.assertAlmostEqual(left, right, delta=1e-8)
        for name in ("saltelli2002", "sobol2007"):
            base = estimate_sobol_indices(lambda x: ishigami(*x), a, b, estimator=name)
            moved = estimate_sobol_indices(shifted, a, b, estimator=name)
            gap = max(
                abs(left - right)
                for left, right in zip(base.first_order, moved.first_order, strict=True)
            )
            self.assertGreater(gap, 0.05)

    def test_multi_quantity_matches_single_quantity_per_estimator(self) -> None:
        """``sobol_analysis_all`` reproduces ``sobol_analysis`` for one estimator."""
        model = ishigami_model()
        many = sobol_analysis_all(model, 256, seed=8, estimator="janon")
        one = sobol_analysis(model, "y", 256, seed=8, estimator="janon")
        self.assertEqual(many["y"].result.first_order, one.result.first_order)
        self.assertEqual(many["y"].estimator, "janon")


class AsymptoticIntervalTests(unittest.TestCase):
    """Fisher-z (Martinez) and normal (Janon) asymptotic intervals."""

    def test_asymptotic_coverage_on_gaussian_linear_model(self) -> None:
        """Nominal 90% intervals cover ``S_1`` at about the nominal rate."""
        weights = (1.0, 2.0, 0.5)
        truth_first = 1.0 / 5.25
        truth_total = truth_first
        model = _linear_model(weights, Normal(0.0, 1.0))
        for name in ("martinez", "janon"):
            hits = 0
            repetitions = 40
            for seed in range(repetitions):
                result = sobol_analysis(
                    model,
                    "y",
                    256,
                    design="mc",
                    seed=seed,
                    confidence=0.9,
                    estimator=name,
                    asymptotic=True,
                )
                self.assertEqual(result.interval_method, "asymptotic")
                first = result.first_order[0]
                total = result.total_effect[0]
                self.assertLessEqual(first.lower, first.estimate)
                self.assertLessEqual(first.estimate, first.upper)
                self.assertLessEqual(total.lower, total.estimate)
                self.assertLessEqual(total.estimate, total.upper)
                hits += first.lower <= truth_first <= first.upper
                hits += total.lower <= truth_total <= total.upper
            coverage = hits / (2 * repetitions)
            with self.subTest(estimator=name):
                self.assertGreater(coverage, 0.78)
                self.assertLessEqual(coverage, 1.0)

    def test_martinez_interval_is_the_fisher_transform(self) -> None:
        """The first-order bounds are ``tanh(atanh(S) -+ z / sqrt(N - 3))``."""
        result = sobol_analysis(
            ishigami_model(),
            "y",
            1000,
            design="mc",
            seed=1,
            estimator="martinez",
            asymptotic=True,
        )
        z = Normal(0.0, 1.0).quantile(0.975)
        first = result.first_order[1]
        centre = math.atanh(first.estimate)
        self.assertAlmostEqual(first.lower, math.tanh(centre - z / math.sqrt(997)), 12)
        self.assertAlmostEqual(first.upper, math.tanh(centre + z / math.sqrt(997)), 12)

    def test_invalid_settings_are_rejected(self) -> None:
        """Asymptotic intervals need a Monte Carlo design and a fitting estimator."""
        model = ishigami_model()
        with self.assertRaises(ValueError):
            sobol_analysis(model, "y", 64, estimator="martinez", asymptotic=True)
        with self.assertRaises(ValueError):
            sobol_analysis(
                model, "y", 64, design="mc", estimator="jansen", asymptotic=True
            )
        with self.assertRaises(ValueError):
            sobol_analysis(
                model,
                "y",
                64,
                design="mc",
                estimator="janon",
                asymptotic=True,
                bootstrap=10,
            )
        with self.assertRaises(TypeError):
            sobol_analysis(model, "y", 64, design="mc", asymptotic=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sobol_analysis(model, "y", 64, estimator="sobol2002")
        a, b = _uniform_matrices(8, 3, 1)
        with self.assertRaises(ValueError):
            estimate_sobol_indices(lambda x: ishigami(*x), a, b, estimator="nope")


class TaylorImportanceTests(unittest.TestCase):
    """Quadratic-sum importance factors."""

    def setUp(self) -> None:
        """Build the flood model and the nominal point."""
        self.model = flood_model()
        self.point = tuple(FLOOD_NOMINAL[name] for name in FLOOD_INPUT_NAMES)
        self.sigma = tuple(
            FLOOD_NOMINAL_STANDARD_DEVIATIONS[name] for name in FLOOD_INPUT_NAMES
        )

    def test_flood_finite_differences_match_the_exact_gradient(self) -> None:
        """A 1% central difference has the predicted truncation error."""
        fd = taylor_importance_factors(
            self.model,
            "overflow",
            point=FLOOD_NOMINAL,
            standard_deviations=FLOOD_NOMINAL_STANDARD_DEVIATIONS,
        )
        exact = taylor_importance_factors(
            self.model,
            "overflow",
            point=FLOOD_NOMINAL,
            standard_deviations=FLOOD_NOMINAL_STANDARD_DEVIATIONS,
            gradient=_flood_gradient,
        )
        self.assertEqual(fd.evaluations, 16)
        self.assertEqual(exact.evaluations, 0)
        self.assertEqual(fd.gradient_method, "central-difference")
        self.assertEqual(exact.gradient_method, "supplied")
        self.assertEqual(exact.gradient, _flood_gradient(self.point))
        # S is a power law x^p in Q (p = 0.6), Ks and B (p = -0.6), and L
        # (p = 0.3): the central quotient with h = c x equals the derivative
        # times ((1 + c)^p - (1 - c)^p) / (2 c p).
        c = 0.01
        for index, power in ((0, 0.6), (1, -0.6), (6, 0.3), (7, -0.6)):
            factor = ((1 + c) ** power - (1 - c) ** power) / (2 * c * power)
            self.assertAlmostEqual(
                fd.gradient[index] / exact.gradient[index], factor, delta=1e-9
            )
        for index in range(8):
            relative = abs(fd.gradient[index] / exact.gradient[index] - 1.0)
            self.assertLess(relative, 1e-2)
        for left, right in zip(
            fd.importance_factors, exact.importance_factors, strict=True
        ):
            self.assertAlmostEqual(left, right, delta=1e-3)

    def test_flood_importance_factors_and_taylor_variance(self) -> None:
        """Factors sum to one and follow the analytic contributions."""
        result = taylor_importance_factors(
            self.model,
            "overflow",
            point=self.point,
            standard_deviations=self.sigma,
            gradient=_flood_gradient,
        )
        gradient = _flood_gradient(self.point)
        terms = [(g * s) ** 2 for g, s in zip(gradient, self.sigma, strict=True)]
        variance = math.fsum(terms)
        self.assertAlmostEqual(result.taylor_variance, variance, places=12)
        self.assertAlmostEqual(math.fsum(result.importance_factors), 1.0, places=12)
        self.assertAlmostEqual(result.taylor_standard_deviation, math.sqrt(variance))
        self.assertEqual(
            result.ranking(), ("Q", "Hd", "Zv", "Ks", "Cb", "Zm", "B", "L")
        )
        self.assertAlmostEqual(result.as_mapping()["Q"], 0.3484, delta=5e-4)
        self.assertAlmostEqual(result.as_mapping()["Hd"], 0.2963, delta=5e-4)
        self.assertEqual(result.quantity, "overflow")

    def test_default_point_and_spread_come_from_the_input_law(self) -> None:
        """Without arguments the expansion uses input means and deviations."""
        result = taylor_importance_factors(self.model, "overflow")
        self.assertEqual(result.point, tuple(self.model.nominal()))
        self.assertEqual(
            result.standard_deviations,
            tuple(self.model.inputs.standard_deviations()),
        )

    def test_flood_overflow_is_nearly_linear_so_sobol_agrees(self) -> None:
        """On the overflow the Taylor factors track the Sobol indices."""
        taylor = taylor_importance_factors(self.model, "overflow")
        sobol = sobol_analysis(self.model, "overflow", 1024, seed=2026, replicates=8)
        for eta, first in zip(
            taylor.importance_factors, sobol.first_order, strict=True
        ):
            self.assertAlmostEqual(eta, first.estimate, delta=0.03)

    def test_ishigami_shows_that_the_factors_are_local(self) -> None:
        """At the mean of Ishigami only ``x1`` has a slope; ``x2`` still matters."""
        model = ishigami_model()
        taylor = taylor_importance_factors(model, "y")
        self.assertAlmostEqual(taylor.importance_factors[0], 1.0, places=6)
        self.assertLess(taylor.importance_factors[1], 1e-6)
        self.assertEqual(taylor.steps, (0.01, 0.01, 0.01))
        sobol = sobol_analysis(model, "y", 1024, seed=1)
        first = [item.estimate for item in sobol.first_order]
        self.assertGreater(first[1], first[0])
        self.assertGreater(first[1], 0.35)

    def test_plain_function_variant(self) -> None:
        """A linear function gives exact factors; names default to ``x1, ...``."""
        result = quadratic_sum_importance(
            lambda x: 2.0 * x[0] - x[1] + 0.5 * x[2], (1.0, -3.0, 4.0), (1.0, 2.0, 2.0)
        )
        expected = (4.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0)
        for value, target in zip(result.importance_factors, expected, strict=True):
            self.assertAlmostEqual(value, target, places=9)
        self.assertEqual(result.input_names, ("x1", "x2", "x3"))
        self.assertIsNone(result.quantity)
        self.assertEqual(result.steps, (0.01, 0.03, 0.04))
        supplied = quadratic_sum_importance(
            lambda x: 0.0, (1.0, 2.0), (1.0, 1.0), gradient=(3.0, 4.0)
        )
        self.assertAlmostEqual(supplied.taylor_variance, 25.0)
        self.assertEqual(supplied.ranking(), ("x2", "x1"))

    def test_invalid_arguments(self) -> None:
        """Bad shapes, negative spreads, zero variance, and dependence raise."""
        with self.assertRaises(ValueError):
            quadratic_sum_importance(lambda x: x[0], (1.0,), (-1.0,))
        with self.assertRaises(ValueError):
            quadratic_sum_importance(lambda x: x[0], (1.0, 2.0), (1.0,))
        with self.assertRaises(ValueError):
            quadratic_sum_importance(lambda x: 1.0, (1.0,), (1.0,))
        with self.assertRaises(ValueError):
            quadratic_sum_importance(lambda x: x[0], (1.0,), (1.0,), relative_step=0)
        with self.assertRaises(ValueError):
            quadratic_sum_importance(
                lambda x: x[0], (1.0, 2.0), (1.0, 1.0), gradient=(1.0,)
            )
        with self.assertRaises(TypeError):
            quadratic_sum_importance("f", (1.0,), (1.0,))  # type: ignore[arg-type]
        dependent = JointInput.from_pearson(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            ((1.0, 0.5), (0.5, 1.0)),
        )
        model = ProbabilisticModel(
            dependent,
            lambda p: {"y": p["a"] + p["b"]},
            (QuantityOfInterest("y", "sum"),),
        )
        with self.assertRaises(ValueError):
            taylor_importance_factors(model, "y")
        with self.assertRaises(KeyError):
            taylor_importance_factors(self.model, "overflow", point={"Q": 1.0})


class MorrisSignedMeanTests(unittest.TestCase):
    """The signed mean elementary effect accompanies ``mu*`` and ``sigma``."""

    def test_mu_is_reported_and_bounded_by_mu_star(self) -> None:
        """``|mu| <= mu*``, with equality for inputs of one-signed effect."""
        result = morris_screening(flood_model(), "overflow", 10, levels=4, seed=2026)
        self.assertEqual(len(result.mu), 8)
        self.assertEqual(result.evaluations, 10 * 9)
        for mu, mu_star in zip(result.mu, result.mu_star, strict=True):
            self.assertLessEqual(abs(mu), mu_star + 1e-12)
        names = list(result.input_names)
        # S decreases one-for-one in Hd and Cb and increases in Q.
        self.assertAlmostEqual(
            result.mu[names.index("Hd")], -result.mu_star[names.index("Hd")]
        )
        self.assertGreater(result.mu[names.index("Q")], 0.0)
        self.assertLess(result.mu[names.index("Cb")], 0.0)

    def test_non_monotone_input_has_small_signed_mean(self) -> None:
        """Effects of ``x1`` in a symmetric quadratic cancel in ``mu``."""
        inputs = JointInput.independent(
            RandomInput("x1", Uniform(-1.0, 1.0)), RandomInput("x2", Uniform(-1.0, 1.0))
        )
        model = ProbabilisticModel(
            inputs,
            lambda p: {"y": p["x1"] ** 2 + p["x2"]},
            (QuantityOfInterest("y", "quadratic"),),
        )
        result = morris_screening(model, "y", 40, levels=4, seed=3)
        self.assertLess(abs(result.mu[0]), 0.5 * result.mu_star[0])
        self.assertAlmostEqual(result.mu[1], result.mu_star[1])


if __name__ == "__main__":
    unittest.main()
