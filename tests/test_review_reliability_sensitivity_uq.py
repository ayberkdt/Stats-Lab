"""Regression tests for the reliability/sensitivity/UQ defect review (ADR-0092).

References:
    [REF-SONG-NELSON-STAUM-2016], [REF-PLISCHKE-BORGONOVO-SMITH-2013],
    [REF-SOBOL-KUCHERENKO-2009], [REF-WILSON-1927].

"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.distributions import Normal, Poisson, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import ModelLimitState, model_monte_carlo
from probstat.sensitivity import dgsm, given_data_first_order, shapley_effects
from probstat.uq import QuantityOfInterest


def _model(inputs: tuple[RandomInput, ...], function: object) -> ProbabilisticModel:
    """Return a one-QoI model ``y = function(p)`` on independent inputs.

    References:
        [REF-SONG-NELSON-STAUM-2016]. No equation is evaluated.

    """
    return ProbabilisticModel(
        JointInput.independent(*inputs),
        lambda p: {"y": function(p)},  # type: ignore[operator]
        (QuantityOfInterest("y", "output"),),
    )


class ShapleyStandardErrorTest(unittest.TestCase):
    """The reported Shapley standard errors match the seeded spread.

    ``Y = A + B / 2`` with independent standard normals has
    ``Sh = (0.8, 0.2)``. With a small variance sample the normalizing variance
    dominates the error; before the fix the exact-enumeration errors were
    about 24 % too large and the permutation errors about 29 % too small
    (``rms(SE) / sd(Sh)`` of 1.24 and 0.71 over 200 seeds).

    References:
        [EQ-SENS-SHAPLEY-005], [REF-SONG-NELSON-STAUM-2016].

    """

    def _ratio(self, method: str) -> float:
        """Return ``rms(reported SE) / empirical sd`` of ``Sh_A`` over seeds.

        References:
            [EQ-SENS-SHAPLEY-005]. No new equation is introduced.

        """
        model = _model(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            lambda p: p["a"] + 0.5 * p["b"],
        )
        effects: list[float] = []
        errors: list[float] = []
        for seed in range(200):
            result = shapley_effects(
                model,
                "y",
                method=method,  # type: ignore[arg-type]
                seed=seed,
                outer_samples=50 if method == "exact" else 1,
                inner_samples=3,
                variance_samples=30,
                permutations=50,
            )
            effects.append(result.effects[0])
            errors.append(result.standard_errors[0])
        self.assertAlmostEqual(statistics.fmean(effects), 0.8, delta=0.03)
        spread = statistics.stdev(effects)
        return math.sqrt(statistics.fmean(e * e for e in errors)) / spread

    def test_exact_enumeration_errors_are_calibrated(self) -> None:
        """Check the exact-enumeration standard error against the spread.

        References:
            [EQ-SENS-SHAPLEY-005].

        """
        # 200 replicates estimate the spread to about 5 %.
        ratio = self._ratio("exact")
        self.assertGreater(ratio, 0.82)
        self.assertLess(ratio, 1.15)

    def test_permutation_errors_are_calibrated(self) -> None:
        """Check the random-permutation standard error against the spread.

        References:
            [EQ-SENS-SHAPLEY-005].

        """
        ratio = self._ratio("random")
        self.assertGreater(ratio, 0.82)
        self.assertLess(ratio, 1.15)


class GivenDataTiesTest(unittest.TestCase):
    """Tied input values stay in one class of the given-data estimator.

    References:
        [EQ-SENS-GIVEN-001], [REF-PLISCHKE-BORGONOVO-SMITH-2013].

    """

    def test_binary_input_is_not_biased_low(self) -> None:
        """``Y = B + 0.5 Z``, ``B ~ Bernoulli(1/2)`` has ``S_B = 1/2``.

        Splitting the tie runs of ``B`` across classes biased the estimate to
        about 0.490 (seven standard errors of the 200-replicate mean).

        References:
            [EQ-SENS-GIVEN-001].

        """
        generator = random.Random(1)
        estimates: list[float] = []
        for _ in range(100):
            rows: list[tuple[float, float]] = []
            outputs: list[float] = []
            for _ in range(1000):
                flag = 1.0 if generator.random() < 0.5 else 0.0
                rows.append((flag, generator.random()))
                outputs.append(flag + generator.gauss(0.0, 0.5))
            result = given_data_first_order(rows, outputs)
            self.assertEqual(result.classes_used[0], 2)
            estimates.append(result.first_order[0])
        mean = statistics.fmean(estimates)
        error = statistics.stdev(estimates) / math.sqrt(len(estimates))
        self.assertLess(abs(mean - 0.5), 4.0 * error)

    def test_single_binary_split_is_the_exact_correlation_ratio(self) -> None:
        """Two classes reproduce the closed-form between/total ratio.

        References:
            [EQ-SENS-GIVEN-001].

        """
        rows = [(0.0,), (1.0,), (0.0,), (1.0,), (0.0,), (1.0,), (0.0,), (1.0,)]
        outputs = [0.0, 2.0, 1.0, 3.0, 0.0, 2.0, 1.0, 3.0]
        result = given_data_first_order(rows, outputs, partitions=3)
        # Class means 0.5 and 2.5 around 1.5: SSB = 8, SST = 10.
        self.assertEqual(result.classes_used, (2,))
        self.assertAlmostEqual(result.first_order_raw[0], 0.8, places=12)
        self.assertAlmostEqual(result.first_order[0], 1.0 - 7.0 / 6.0 * 0.2, places=12)

    def test_continuous_inputs_are_unchanged(self) -> None:
        """Without ties every requested class is used.

        References:
            [EQ-SENS-GIVEN-001].

        """
        generator = random.Random(4)
        rows = [(generator.random(), generator.random()) for _ in range(400)]
        outputs = [math.sin(6.0 * a) + 0.1 * b for a, b in rows]
        result = given_data_first_order(rows, outputs)
        self.assertEqual(result.classes_used, (20, 20))


class DGSMDiscreteInputTest(unittest.TestCase):
    """DGSM refuses integer-valued inputs instead of differentiating them.

    References:
        [EQ-SENS-DGSM-001], [REF-SOBOL-KUCHERENKO-2009].

    """

    def test_discrete_input_raises(self) -> None:
        """A Poisson input used to give ``nu_k = 1`` for ``y = k + b``.

        References:
            [EQ-SENS-DGSM-001].

        """
        model = _model(
            (RandomInput("k", Poisson(3.0)), RandomInput("b", Uniform(0.0, 1.0))),
            lambda p: p["k"] + p["b"],
        )
        with self.assertRaisesRegex(ValueError, "continuous inputs"):
            dgsm(model, "y", 16, seed=1)

    def test_continuous_inputs_still_work(self) -> None:
        """A linear model of normal and uniform inputs keeps its exact ``nu``.

        References:
            [EQ-SENS-DGSM-001], [EQ-SENS-DGSM-002].

        """
        model = _model(
            (RandomInput("a", Normal(1.0, 2.0)), RandomInput("b", Uniform(0.0, 3.0))),
            lambda p: p["a"] + 2.0 * p["b"],
        )
        result = dgsm(model, "y", 64, seed=1)
        self.assertAlmostEqual(result.nu[0], 1.0, places=6)
        self.assertAlmostEqual(result.nu[1], 4.0, places=6)


class ZeroFailureMonteCarloTest(unittest.TestCase):
    """With no failures the Wilson interval, not the plug-in error, is honest.

    References:
        [EQ-MC-PROB-001], [REF-WILSON-1927].

    """

    def test_zero_failures(self) -> None:
        """No failure in 200 draws: SE is zero, the Wilson bound is positive.

        References:
            [EQ-MC-PROB-001].

        """
        model = _model((RandomInput("a", Normal(0.0, 1.0)),), lambda p: p["a"])
        state = ModelLimitState(model, "y", 8.0, "above")
        result = model_monte_carlo(state, 200, seed=1)
        self.assertEqual(result.probability, 0.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.coefficient_of_variation, math.inf)
        self.assertGreater(result.failure.confidence_interval.upper, 0.01)


if __name__ == "__main__":
    unittest.main()


class CachedSobolTests(unittest.TestCase):
    """A Sobol analysis repeated under one evaluation cache is accepted."""

    def test_repeated_analysis_under_a_cache_reuses_runs(self) -> None:
        """The second run costs no evaluations and gives identical indices.

        Before the fix the result record required ``evaluations >=
        sample_size`` and rejected the cached repeat with "sample_size and
        evaluations are inconsistent".
        """
        from probstat.distributions import Normal
        from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
        from probstat.sensitivity import sobol_analysis
        from probstat.uq import EvaluationCache, QuantityOfInterest

        inputs = JointInput.independent(
            RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))
        )
        model = ProbabilisticModel(
            inputs,
            lambda p: {"y": p["a"] + 0.5 * p["b"]},
            (QuantityOfInterest("y", "response", "m"),),
        )
        with model.evaluation_context(cache=EvaluationCache()):
            first = sobol_analysis(model, "y", 64, seed=3)
            second = sobol_analysis(model, "y", 64, seed=3)
        self.assertEqual(first.result.evaluations, 64 * 4)
        self.assertEqual(second.result.evaluations, 0)
        self.assertEqual(second.result.first_order, first.result.first_order)
        self.assertEqual(second.result.total_effect, first.result.total_effect)
