"""One-at-a-time swings, extended FAST and random balance design.

Oracles: closed forms (linear models, a product of two uniform inputs, the
Ishigami function with its published indices), hand-computed swings and
shares, the library's own Sobol estimator judged within combined standard
errors, and the algebra of the Tissot-Prieur correction. Stochastic checks are
judged on the phase-resampling spread of the estimate plus a stated
truncation allowance of the harmonic series; no seeded digits are pinned.

References:
    [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-TARANTOLA-GATELLI-MARA-2006],
    [REF-TISSOT-PRIEUR-2012], [REF-HOWARD-1988], [REF-ISHIGAMI-HOMMA-1990].
    Checks [EQ-SENS-TORNADO-001], [EQ-SENS-TORNADO-002], [EQ-SENS-FAST-001],
    [EQ-SENS-FAST-002], [EQ-SENS-FAST-003], [EQ-SENS-RBD-001],
    [EQ-SENS-RBD-002].

"""

from __future__ import annotations

import math
import unittest
from collections.abc import Mapping

from probstat.distributions import Normal, StudentT, Uniform
from probstat.modeling import (
    GaussianCopula,
    JointInput,
    ProbabilisticModel,
    RandomInput,
)
from probstat.modeling.benchmarks import (
    ISHIGAMI_FIRST_ORDER,
    ISHIGAMI_TOTAL,
    ishigami_model,
)
from probstat.sensitivity import (
    fast_analysis,
    fast_frequencies,
    fast_indices,
    fast_sample_size,
    rbd_fast_analysis,
    rbd_fast_indices,
    sobol_analysis,
    tornado_analysis,
    tornado_function,
)
from probstat.uq import QuantityOfInterest

LINEAR_WEIGHTS = (1.0, 2.0, 3.0)
LINEAR_SHARES = tuple(w * w / 14.0 for w in LINEAR_WEIGHTS)


def _linear(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``3 a - 2 b + 0.5 c`` for the named inputs.

    References:
        [REF-SALTELLI-2008, Chapter 1]. No equation is evaluated.

    """
    return {"y": 3.0 * parameters["a"] - 2.0 * parameters["b"] + 0.5 * parameters["c"]}


def _linear_model() -> ProbabilisticModel:
    """Return a linear model with normal, uniform and normal inputs.

    References:
        [REF-SALTELLI-2008, Chapter 1]. No equation is evaluated.

    """
    inputs = JointInput.independent(
        RandomInput("a", Normal(1.0, 2.0)),
        RandomInput("b", Uniform(0.0, 4.0)),
        RandomInput("c", Normal(-3.0, 0.5)),
    )
    return ProbabilisticModel(
        inputs, _linear, (QuantityOfInterest("y", "linear response"),), name="linear"
    )


def _weighted(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``x1 + 2 x2 + 3 x3``.

    References:
        [REF-SALTELLI-2008, Chapter 1]. No equation is evaluated.

    """
    return {
        "y": parameters["x1"] + 2.0 * parameters["x2"] + 3.0 * parameters["x3"],
    }


def _uniform_model(response: object, count: int = 3) -> ProbabilisticModel:
    """Return a model of independent ``Uniform(0, 1)`` inputs ``x1 ...``.

    References:
        [REF-SALTELLI-2008, Chapter 4]. No equation is evaluated.

    """
    inputs = JointInput.independent(
        *(RandomInput(f"x{i + 1}", Uniform(0.0, 1.0)) for i in range(count))
    )
    return ProbabilisticModel(
        inputs,
        response,  # type: ignore[arg-type]
        (QuantityOfInterest("y", "response"),),
        name="uniform",
    )


def _product(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``x1 x2`` (``x3`` is a dummy input).

    References:
        [REF-SALTELLI-2008, Chapter 4]. No equation is evaluated.

    """
    return {"y": parameters["x1"] * parameters["x2"]}


def _dummy(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``x1 + 2 x2`` (``x3`` has no effect).

    References:
        [REF-SALTELLI-2008, Chapter 4]. No equation is evaluated.

    """
    return {"y": parameters["x1"] + 2.0 * parameters["x2"]}


def _square(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``x^2 + 0.1 z``.

    References:
        [REF-SALTELLI-2008, Chapter 1]. No equation is evaluated.

    """
    return {"y": parameters["x"] ** 2 + 0.1 * parameters["z"]}


class TornadoTests(unittest.TestCase):
    """Swings, shares, ordering and the non-monotonicity flag.

    References:
        [REF-HOWARD-1988]. Checks [EQ-SENS-TORNADO-001] and
        [EQ-SENS-TORNADO-002].

    """

    def test_linear_model_swings_are_exact(self) -> None:
        """For a linear model the swing is ``c_i (x_high - x_low)``.

        References:
            [EQ-SENS-TORNADO-001].

        """
        model = _linear_model()
        result = tornado_analysis(model, "y")
        weights = {"a": 3.0, "b": -2.0, "c": 0.5}
        laws = {item.name: item.distribution for item in model.inputs.inputs}
        for bar in result.bars:
            expected = weights[bar.name] * (
                laws[bar.name].quantile(0.9) - laws[bar.name].quantile(0.1)
            )
            self.assertAlmostEqual(bar.swing, expected, places=12)
        self.assertEqual(result.evaluations, 7)
        # Normal(1, 2): the median is the mean, so the baseline output is
        # 3 * 1 - 2 * 2 + 0.5 * (-3) = -2.5.
        self.assertAlmostEqual(result.baseline_output, -2.5, places=12)
        self.assertEqual(result.baseline, "median")
        self.assertEqual(result.nonmonotone_inputs(), ())

    def test_hand_computed_ordering_signs_and_shares(self) -> None:
        """Swings 4, 3, -1 give shares 4, 2.25, 0.25 over 6.5.

        References:
            [EQ-SENS-TORNADO-001].

        """

        def function(x: tuple[float, ...]) -> float:
            """Return ``2 x1 + x2 - 0.5 x3``.

            References:
                [EQ-SENS-TORNADO-001]. No equation is evaluated.

            """
            return 2.0 * x[0] + 1.0 * x[1] - 0.5 * x[2]

        result = tornado_function(
            function,
            low=(0.0, 0.0, 0.0),
            base=(1.0, 1.0, 1.0),
            high=(2.0, 3.0, 2.0),
            names=("p", "q", "r"),
        )
        self.assertEqual(result.ranking(), ("p", "q", "r"))
        swings = [bar.swing for bar in result.bars]
        self.assertEqual(swings, [4.0, 3.0, -1.0])
        self.assertEqual([bar.sign for bar in result.bars], [1, 1, -1])
        total = 4.0 + 2.25 + 0.25
        for bar, expected in zip(result.bars, (4.0, 2.25, 0.25), strict=True):
            self.assertAlmostEqual(bar.share, expected / total, places=14)
        self.assertAlmostEqual(math.fsum(bar.share for bar in result.bars), 1.0)
        self.assertEqual(result.baseline_output, 2.5)
        self.assertEqual(result.evaluations, 7)
        self.assertEqual(dict(result.as_mapping()), {"p": 4.0, "q": 3.0, "r": -1.0})

    def test_square_is_flagged_non_monotone(self) -> None:
        """``x^2`` with symmetric ends has swing zero and is flagged.

        References:
            [EQ-SENS-TORNADO-002].

        """
        inputs = JointInput.independent(
            RandomInput("x", Normal(0.0, 1.0)), RandomInput("z", Normal(0.0, 1.0))
        )
        model = ProbabilisticModel(
            inputs, _square, (QuantityOfInterest("y", "square"),), name="square"
        )
        result = tornado_analysis(model, "y")
        by_name = {bar.name: bar for bar in result.bars}
        self.assertAlmostEqual(by_name["x"].swing, 0.0, places=12)
        self.assertTrue(by_name["x"].nonmonotone)
        self.assertFalse(by_name["z"].nonmonotone)
        self.assertEqual(result.nonmonotone_inputs(), ("x",))
        # The linear input ranks first although x drives most of the variance.
        self.assertEqual(result.ranking()[0], "z")

    def test_all_swings_zero_gives_nan_shares(self) -> None:
        """A function without effect has undefined shares, not zero.

        References:
            [EQ-SENS-TORNADO-001].

        """
        result = tornado_function(
            lambda x: 1.0, low=(0.0, 0.0), base=(1.0, 1.0), high=(2.0, 2.0)
        )
        self.assertTrue(all(math.isnan(bar.share) for bar in result.bars))
        self.assertEqual([bar.sign for bar in result.bars], [0, 0])

    def test_baseline_choices_and_explicit_values(self) -> None:
        """Mean, point and mapping baselines; explicit values override quantiles.

        References:
            [EQ-SENS-TORNADO-001].

        """
        model = _linear_model()
        by_mean = tornado_analysis(model, "y", baseline="mean")
        self.assertEqual(by_mean.baseline, "mean")
        self.assertAlmostEqual(by_mean.baseline_output, -2.5, places=12)
        point = tornado_analysis(model, "y", baseline=(0.0, 1.0, 2.0))
        self.assertEqual(point.baseline_input, (0.0, 1.0, 2.0))
        self.assertAlmostEqual(point.baseline_output, 0.0 - 2.0 + 1.0, places=12)
        named = tornado_analysis(model, "y", baseline={"a": 0.0, "b": 1.0, "c": 2.0})
        self.assertEqual(named.baseline_input, (0.0, 1.0, 2.0))
        given = tornado_analysis(model, "y", values={"b": (1.0, 3.0)})
        bar = next(item for item in given.bars if item.name == "b")
        self.assertAlmostEqual(bar.swing, -2.0 * 2.0, places=12)
        self.assertEqual((bar.low_input, bar.high_input), (1.0, 3.0))

    def test_unbracketed_baseline_skips_the_check(self) -> None:
        """A baseline outside the end values is reported as not bracketed.

        References:
            [EQ-SENS-TORNADO-002].

        """
        result = tornado_function(
            lambda x: x[0] ** 2, low=(1.0,), base=(0.0,), high=(2.0,)
        )
        bar = result.bars[0]
        self.assertFalse(bar.bracketed)
        self.assertFalse(bar.nonmonotone)

    def test_dependent_inputs_are_refused(self) -> None:
        """A joint law with a copula is refused with an explanation.

        References:
            [EQ-SENS-TORNADO-001].

        """
        inputs = JointInput(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            copula=GaussianCopula(((1.0, 0.6), (0.6, 1.0))),
        )

        def response(parameters: Mapping[str, float]) -> Mapping[str, float]:
            """Return the sum of the inputs.

            References:
                [EQ-SENS-TORNADO-001]. No equation is evaluated.

            """
            return {"y": parameters["a"] + parameters["b"]}

        model = ProbabilisticModel(inputs, response, (QuantityOfInterest("y", "s"),))
        with self.assertRaisesRegex(ValueError, "independent"):
            tornado_analysis(model, "y")

    def test_argument_validation(self) -> None:
        """Invalid levels, baselines, names and values are rejected.

        References:
            [EQ-SENS-TORNADO-001].

        """
        model = _linear_model()
        with self.assertRaises(ValueError):
            tornado_analysis(model, "y", low=0.9, high=0.1)
        with self.assertRaises(ValueError):
            tornado_analysis(model, "y", low=0.0, high=0.5)
        with self.assertRaises(ValueError):
            tornado_analysis(model, "y", baseline="mode")
        with self.assertRaises(ValueError):
            tornado_analysis(model, "y", baseline=(1.0, 2.0))
        with self.assertRaises(KeyError):
            tornado_analysis(model, "nope")
        with self.assertRaises(KeyError):
            tornado_analysis(model, "y", values={"zz": (0.0, 1.0)})
        with self.assertRaises(TypeError):
            tornado_function(3, (0.0,), (1.0,), (2.0,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            tornado_function(lambda x: x[0], (0.0,), (1.0, 1.0), (2.0,))
        with self.assertRaises(ValueError):
            tornado_function(
                lambda x: x[0], (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), names=("a", "a")
            )
        with self.assertRaises(ValueError):
            tornado_function(lambda x: math.nan, (0.0,), (1.0,), (2.0,))
        with self.assertRaises(ValueError):
            tornado_function(lambda x: x[0], {"a": 0.0}, {"a": 1.0}, {"a": 2.0})
        mapped = tornado_function(
            lambda x: x[0] - x[1],
            {"a": 0.0, "b": 0.0},
            {"a": 1.0, "b": 1.0},
            {"a": 2.0, "b": 3.0},
            names=("a", "b"),
        )
        self.assertEqual(mapped.ranking(), ("b", "a"))


class FastPlanTests(unittest.TestCase):
    """Frequency plan and sample-size rule.

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999]. Checks [EQ-SENS-FAST-003].

    """

    def test_plan_satisfies_the_nyquist_and_distinctness_rules(self) -> None:
        """Frequencies are distinct, bounded by ``omega / (2 M)``, below Nyquist.

        References:
            [EQ-SENS-FAST-003].

        """
        for d in (1, 2, 3, 5, 8):
            for m in (2, 4, 6):
                n = fast_sample_size(d, m)
                self.assertEqual(n % 2, 1)
                omega, others = fast_frequencies(d, n, m)
                self.assertEqual(len(others), d - 1)
                self.assertEqual(len(set(others)), d - 1)
                self.assertLessEqual(max(others, default=0), omega // (2 * m))
                self.assertGreaterEqual(min(others, default=1), 1)
                self.assertLessEqual(2 * m * omega + 1, n)
                # Every harmonic of a low frequency ends below the first
                # harmonic of the high one.
                self.assertLessEqual(m * max(others, default=0), omega // 2)
        self.assertEqual(fast_frequencies(3, 193, 4), (24, (1, 3)))
        self.assertEqual(fast_frequencies(2, 129, 4), (16, (1,)))
        self.assertEqual(fast_sample_size(1, 4), 65)
        self.assertEqual(fast_sample_size(3, 4), 193)
        self.assertEqual(fast_sample_size(5, 2), 4 * 4 * 7 + 1)
        _, five = fast_frequencies(5, fast_sample_size(5, 2), 2)
        self.assertTrue(all(b - a >= 2 for a, b in zip(five, five[1:], strict=False)))

    def test_plan_equals_the_published_frequency_rule(self) -> None:
        """The complementary set is the one of the open reference descriptions.

        The OpenTURNS theory page for extended FAST (which cites Saltelli,
        Tarantola and Chan 1999) and the R ``sensitivity::fast99`` source give
        ``omega = (N - 1) / (2 M)``, a largest complementary frequency
        ``omega / (2 M)`` and the others "distributed uniformly between 1 and"
        it, in R ``floor(seq(1, m, length.out = d - 1))`` for ``m >= d - 1``.
        This plan reproduces that set wherever it accepts the curve length. The
        page's example (eight inputs, ``N = 513``, ``M = 4``) gives
        ``{64, 1, 2, 3, 4, 5, 6, 8}``; this library refuses that short a curve
        because it requires neighbouring low frequencies to be two apart (a
        rule of its own, see ADR-0141), so the example is checked against the
        reference rule written out here and the refusal is asserted.

        References:
            [REF-SALTELLI-TARANTOLA-CHAN-1999]. Checks [EQ-SENS-FAST-003].

        """

        def reference(d: int, n: int, m: int) -> tuple[int, tuple[int, ...]]:
            """Return the frequency set of ``fast99``: high one, then the rest.

            References:
                [EQ-SENS-FAST-003].

            """
            omega = (n - 1) // (2 * m)
            top = omega // (2 * m)
            if top >= d - 1:
                step = (top - 1) / (d - 2) if d > 2 else 0.0
                rest = tuple(math.floor(1 + k * step) for k in range(d - 1))
            else:
                rest = tuple(k % top + 1 for k in range(d - 1))
            return omega, rest

        self.assertEqual(reference(8, 513, 4), (64, (1, 2, 3, 4, 5, 6, 8)))
        with self.assertRaises(ValueError):
            fast_frequencies(8, 513, 4)
        for d in (3, 4, 6, 8):
            for m in (4, 6):
                for extra in (0, 40, 200):
                    n = fast_sample_size(d, m) + extra
                    n += 1 - n % 2
                    self.assertEqual(fast_frequencies(d, n, m), reference(d, n, m))

    def test_rule_violations_are_rejected(self) -> None:
        """Even or too short curves fail with the required size in the text.

        References:
            [EQ-SENS-FAST-003].

        """
        with self.assertRaisesRegex(ValueError, "odd"):
            fast_frequencies(3, 194, 4)
        with self.assertRaisesRegex(ValueError, "193"):
            fast_frequencies(3, 129, 4)
        with self.assertRaises(ValueError):
            fast_sample_size(0, 4)
        with self.assertRaises(TypeError):
            fast_sample_size(3, True)  # type: ignore[arg-type]


class ExtendedFastTests(unittest.TestCase):
    """Extended FAST against closed forms and the Sobol estimator.

    The allowance ``TRUNCATION`` covers the harmonic series cut at ``M``
    terms; the spread covers the random phases.

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973]. Checks
        [EQ-SENS-FAST-001] and [EQ-SENS-FAST-002].

    """

    TRUNCATION = 0.02

    def test_adjacent_low_frequencies_bias_the_ishigami_indices(self) -> None:
        """The pair ``(1, 2)`` errs several times more than the pair ``(1, 3)``.

        The measurement behind the library's spacing rule: with complementary
        frequencies ``(1, 2)`` the largest first-order error of the Ishigami
        indices over twenty seeds (0 to 19, three resamples, ``M = 4``) at
        ``N = 193`` and ``N = 385`` is above 0.10 (0.18 was measured at both
        lengths), with ``(1, 3)`` it stays below 0.05 (0.03 measured at both).
        The two bounds are far from both measured values, so the comparison
        does not depend on platform arithmetic.

        References:
            [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-ISHIGAMI-HOMMA-1990].
            Checks [EQ-SENS-FAST-003].

        """
        from unittest import mock

        from probstat.foundations.random_streams import RandomStream
        from probstat.sensitivity import fast as fast_module

        a, b = 7.0, 0.1

        def ishigami(point: tuple[float, ...]) -> float:
            """Evaluate the Ishigami function on the unit cube.

            References:
                [REF-ISHIGAMI-HOMMA-1990]. No equation is evaluated.

            """
            x = [math.pi * (2.0 * value - 1.0) for value in point]
            return (
                math.sin(x[0])
                + a * math.sin(x[1]) ** 2
                + b * x[2] ** 4 * math.sin(x[0])
            )

        variance = a * a / 8 + b * math.pi**4 / 5 + b * b * math.pi**8 / 18 + 0.5
        exact = (
            0.5 * (1 + b * math.pi**4 / 5) ** 2 / variance,
            (a * a / 8) / variance,
            0.0,
        )

        def worst_error(pair: tuple[int, int], length: int) -> float:
            """Return the largest first-order error over twenty seeds.

            References:
                [EQ-SENS-FAST-001]. No further equation is evaluated.

            """

            def plan(
                dimension: int, samples: int, harmonics: int
            ) -> tuple[int, tuple[int, ...]]:
                """Return the fixed complementary pair for the curve.

                References:
                    [EQ-SENS-FAST-003]. No equation is evaluated.

                """
                del dimension
                return (samples - 1) // (2 * harmonics), pair

            worst = 0.0
            with mock.patch.object(fast_module, "fast_frequencies", plan):
                for seed in range(20):
                    first, _, _ = fast_module._extended_fast(
                        ishigami, 3, length, 4, 3, RandomStream(seed)
                    )
                    means = [sum(row[i] for row in first) / 3 for i in range(3)]
                    worst = max(worst, max(abs(means[i] - exact[i]) for i in range(3)))
            return worst

        for length in (193, 385):
            self.assertGreater(worst_error((1, 2), length), 0.10)
            self.assertLess(worst_error((1, 3), length), 0.05)

    def _close(self, estimate: float, error: float, exact: float, extra: float) -> None:
        """Assert agreement within five standard errors plus an allowance.

        References:
            [EQ-SENS-FAST-002]. No equation is evaluated.

        """
        self.assertLessEqual(abs(estimate - exact), 5.0 * error + extra)

    def test_additive_linear_model(self) -> None:
        """First and total indices of ``x1 + 2 x2 + 3 x3`` are ``w^2 / 14``.

        References:
            [EQ-SENS-FAST-002].

        """
        result = fast_analysis(
            _uniform_model(_weighted),
            "y",
            fast_sample_size(3, 4),
            harmonics=4,
            resamples=5,
            seed=11,
        )
        for i, exact in enumerate(LINEAR_SHARES):
            self._close(result.first_order[i], result.first_order_error[i], exact, 0.01)
            self._close(
                result.total_effect[i], result.total_effect_error[i], exact, 0.01
            )
        self.assertEqual(result.evaluations, 3 * 5 * 193)
        self.assertEqual(result.ranking(), ("x3", "x2", "x1"))
        self.assertEqual(result.max_frequency, 24)
        self.assertEqual(len(result.first_order_resamples), 5)
        self.assertEqual(set(result.as_mapping()), {"x1", "x2", "x3"})

    def test_product_model_total_exceeds_first_order(self) -> None:
        """``x1 x2`` on ``U(0, 1)^3``: ``S = 3/7``, ``S_T = 4/7``, dummy zero.

        ``V = 7/144``, ``V_1 = Var(X1 / 2) = 3/144``, ``V_12 = 1/144``.

        References:
            [EQ-SENS-FAST-002].

        """
        result = fast_analysis(
            _uniform_model(_product),
            "y",
            fast_sample_size(3, 6),
            harmonics=6,
            resamples=5,
            seed=5,
        )
        for i in (0, 1):
            self._close(
                result.first_order[i], result.first_order_error[i], 3.0 / 7.0, 0.02
            )
            self._close(
                result.total_effect[i], result.total_effect_error[i], 4.0 / 7.0, 0.03
            )
        self._close(result.first_order[2], result.first_order_error[2], 0.0, 0.01)
        self._close(result.total_effect[2], result.total_effect_error[2], 0.0, 0.02)

    def test_ishigami_function(self) -> None:
        """Indices of the Ishigami function against the published values.

        The ``x3``-``x1`` interaction needs more harmonics than the default;
        the allowance of 0.03 is the truncation of the series at ``M = 8``.

        References:
            [REF-ISHIGAMI-HOMMA-1990]. Checks [EQ-SENS-FAST-002].

        """
        result = fast_analysis(
            ishigami_model(),
            "y",
            fast_sample_size(3, 8),
            harmonics=8,
            resamples=4,
            seed=3,
        )
        for i in range(3):
            self._close(
                result.first_order[i],
                result.first_order_error[i],
                ISHIGAMI_FIRST_ORDER[i],
                0.03,
            )
            self._close(
                result.total_effect[i],
                result.total_effect_error[i],
                ISHIGAMI_TOTAL[i],
                0.03,
            )

    def test_agrees_with_the_sobol_estimator(self) -> None:
        """FAST and Saltelli-Jansen agree within combined standard errors.

        The Sobol standard error is read from the 95 percent replicate
        interval; the allowance is the FAST truncation.

        References:
            [REF-SALTELLI-2010]. Checks [EQ-SENS-FAST-002].

        """
        model = ishigami_model()
        sobol = sobol_analysis(model, "y", 8192, replicates=8, seed=2)
        fast = fast_analysis(
            model, "y", fast_sample_size(3, 8), harmonics=8, resamples=4, seed=9
        )
        factor = StudentT(7.0).quantile(0.975)
        for i in range(3):
            for fast_value, fast_error, interval in (
                (fast.first_order[i], fast.first_order_error[i], sobol.first_order[i]),
                (
                    fast.total_effect[i],
                    fast.total_effect_error[i],
                    sobol.total_effect[i],
                ),
            ):
                sobol_error = (interval.upper - interval.lower) / (2.0 * factor)
                combined = math.hypot(fast_error, sobol_error)
                self.assertLessEqual(
                    abs(fast_value - interval.estimate), 4.0 * combined + 0.03
                )

    def test_plain_function_entry_matches_the_model_entry(self) -> None:
        """A callable on named marginals reproduces the model analysis.

        References:
            [EQ-SENS-FAST-002].

        """
        marginals = {f"x{i + 1}": Uniform(0.0, 1.0) for i in range(3)}
        plain = fast_indices(
            lambda x: x[0] + 2.0 * x[1] + 3.0 * x[2],
            marginals,
            fast_sample_size(3, 4),
            seed=11,
        )
        model = fast_analysis(_uniform_model(_weighted), "y", 193, seed=11)
        for a, b in zip(plain.first_order, model.first_order, strict=True):
            self.assertAlmostEqual(a, b, places=10)
        self.assertEqual(plain.input_names, ("x1", "x2", "x3"))
        self.assertIsNone(plain.quantity)
        self.assertEqual(model.quantity, "y")

    def test_marginal_transformation_uses_the_quantile_function(self) -> None:
        """The identity of a normal input has first-order index one.

        A single input needs ``N >= 4 M^2 + 1``; the output is the physical
        value, so the curve must carry the normal law through the quantile.

        References:
            [EQ-SENS-FAST-001].

        """
        result = fast_indices(
            lambda x: x[0], [Normal(10.0, 3.0)], 257, harmonics=8, seed=1
        )
        self.assertAlmostEqual(result.first_order[0], 1.0, delta=0.03)
        self.assertAlmostEqual(result.variance, 9.0, delta=0.5)

    def test_same_seed_same_result(self) -> None:
        """Equal seeds reproduce the estimate; different seeds change phases.

        References:
            [EQ-SENS-FAST-001].

        """
        model = _uniform_model(_product)
        first = fast_analysis(model, "y", 193, seed=4)
        again = fast_analysis(model, "y", 193, seed=4)
        other = fast_analysis(model, "y", 193, seed=5)
        self.assertEqual(first.first_order, again.first_order)
        self.assertNotEqual(first.first_order, other.first_order)

    def test_argument_validation(self) -> None:
        """Dependent inputs, bad sizes, bad types and constant outputs fail.

        References:
            [EQ-SENS-FAST-003].

        """
        model = _uniform_model(_weighted)
        with self.assertRaisesRegex(ValueError, "at least"):
            fast_analysis(model, "y", 65)
        with self.assertRaisesRegex(ValueError, "odd"):
            fast_analysis(model, "y", 194)
        with self.assertRaises(ValueError):
            fast_analysis(model, "y", 193, resamples=1)
        with self.assertRaises(TypeError):
            fast_analysis(model, "y", 193, harmonics=2.0)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            fast_analysis(model, "nope", 193)
        with self.assertRaises(TypeError):
            fast_indices(3, [Uniform(0.0, 1.0)], 129)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fast_indices(lambda x: x[0], [object()], 129)
        with self.assertRaises(ValueError):
            fast_indices(lambda x: x[0], [], 129)
        with self.assertRaises(ValueError):
            fast_indices(lambda x: 1.0, [Uniform(0.0, 1.0)], 129)
        with self.assertRaises(ValueError):
            fast_indices(lambda x: math.inf, [Uniform(0.0, 1.0)], 129)
        with self.assertRaises(ValueError):
            fast_indices(lambda x: x[0], {"a": Uniform(0.0, 1.0)}, 129, names=("b",))
        dependent = JointInput(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            copula=GaussianCopula(((1.0, 0.5), (0.5, 1.0))),
        )

        def response(parameters: Mapping[str, float]) -> Mapping[str, float]:
            """Return ``a + b``.

            References:
                [EQ-SENS-FAST-002]. No equation is evaluated.

            """
            return {"y": parameters["a"] + parameters["b"]}

        coupled = ProbabilisticModel(
            dependent, response, (QuantityOfInterest("y", "sum"),)
        )
        with self.assertRaisesRegex(ValueError, "independent"):
            fast_analysis(coupled, "y", 65)
        with self.assertRaisesRegex(ValueError, "independent"):
            rbd_fast_analysis(coupled, "y", 101)


class RandomBalanceTests(unittest.TestCase):
    """Random balance design and the Tissot-Prieur correction.

    References:
        [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012]. Checks
        [EQ-SENS-RBD-001] and [EQ-SENS-RBD-002].

    """

    def test_additive_linear_model(self) -> None:
        """First-order indices of ``x1 + 2 x2 + 3 x3`` are ``w^2 / 14``.

        References:
            [EQ-SENS-RBD-001].

        """
        result = rbd_fast_analysis(
            _uniform_model(_weighted), "y", 1001, replicates=6, seed=21
        )
        for estimate, error, exact in zip(
            result.first_order, result.first_order_error, LINEAR_SHARES, strict=True
        ):
            self.assertLessEqual(abs(estimate - exact), 5.0 * error + 0.01)
        self.assertEqual(result.evaluations, 6 * 1001)
        self.assertEqual(result.ranking(), ("x3", "x2", "x1"))
        self.assertTrue(result.bias_correction)

    def test_raw_estimator_has_the_predicted_positive_bias(self) -> None:
        """A dummy input has raw index ``2 M / N``; the corrected one is ``0``.

        With no effect of ``x3`` the expected raw estimate is
        ``lambda (1 - 0) = 2 M / N`` (EQ-SENS-RBD-002).

        References:
            [EQ-SENS-RBD-002].

        """
        model = _uniform_model(_dummy)
        raw = rbd_fast_analysis(
            model, "y", 501, harmonics=6, replicates=10, bias_correction=False, seed=8
        )
        corrected = rbd_fast_analysis(
            model, "y", 501, harmonics=6, replicates=10, seed=8
        )
        lam = 2.0 * 6 / 501
        self.assertAlmostEqual(raw.bias, lam, places=14)
        self.assertLessEqual(
            abs(raw.first_order[2] - lam), 5.0 * raw.first_order_error[2]
        )
        self.assertGreater(raw.first_order[2], 0.0)
        self.assertLessEqual(
            abs(corrected.first_order[2]), 5.0 * corrected.first_order_error[2]
        )
        # Same samples: the corrected mean is the algebraic image of the raw.
        for r, c in zip(raw.first_order, corrected.first_order, strict=True):
            self.assertAlmostEqual(c, (r - lam) / (1.0 - lam), places=12)
        for r, c in zip(raw.raw, corrected.raw, strict=True):
            self.assertAlmostEqual(r, c, places=14)
        # Exact first-order values 1/5 and 4/5 of x1 + 2 x2.
        self.assertLessEqual(
            abs(corrected.first_order[1] - 0.8),
            5.0 * corrected.first_order_error[1] + 0.01,
        )

    def test_product_and_ishigami(self) -> None:
        """First-order indices of ``x1 x2`` (``3/7``) and the Ishigami function.

        References:
            [REF-ISHIGAMI-HOMMA-1990]. Checks [EQ-SENS-RBD-001].

        """
        product = rbd_fast_analysis(
            _uniform_model(_product), "y", 1001, replicates=6, seed=2
        )
        for i in (0, 1):
            self.assertLessEqual(
                abs(product.first_order[i] - 3.0 / 7.0),
                5.0 * product.first_order_error[i] + 0.02,
            )
        ishigami = rbd_fast_analysis(
            ishigami_model(), "y", 2001, harmonics=8, replicates=6, seed=6
        )
        for i in range(3):
            self.assertLessEqual(
                abs(ishigami.first_order[i] - ISHIGAMI_FIRST_ORDER[i]),
                5.0 * ishigami.first_order_error[i] + 0.03,
            )

    def test_plain_function_entry(self) -> None:
        """A callable on laws gives the model result with the same seed.

        References:
            [EQ-SENS-RBD-001].

        """
        plain = rbd_fast_indices(
            lambda x: x[0] + 2.0 * x[1] + 3.0 * x[2],
            [Uniform(0.0, 1.0)] * 3,
            301,
            seed=3,
        )
        model = rbd_fast_analysis(_uniform_model(_weighted), "y", 301, seed=3)
        for a, b in zip(plain.first_order, model.first_order, strict=True):
            self.assertAlmostEqual(a, b, places=10)
        self.assertIsNone(plain.quantity)

    def test_argument_validation(self) -> None:
        """Even sizes, too many harmonics, bad flags and constants fail.

        References:
            [EQ-SENS-RBD-001].

        """
        model = _uniform_model(_weighted)
        with self.assertRaisesRegex(ValueError, "odd"):
            rbd_fast_analysis(model, "y", 100)
        with self.assertRaises(ValueError):
            rbd_fast_analysis(model, "y", 11, harmonics=6)
        with self.assertRaises(ValueError):
            rbd_fast_analysis(model, "y", 101, replicates=1)
        with self.assertRaises(TypeError):
            rbd_fast_analysis(model, "y", 101, bias_correction=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            rbd_fast_indices("f", [Uniform(0.0, 1.0)], 101)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            rbd_fast_indices(lambda x: 2.0, [Uniform(0.0, 1.0)], 101)
        with self.assertRaises(KeyError):
            rbd_fast_analysis(model, "nope", 101)


if __name__ == "__main__":
    unittest.main()
