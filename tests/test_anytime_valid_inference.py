"""Tests for e-values, e-BH, betting confidence sequences, universal inference.

Oracles: closed forms (calibrator integrals, the Bernoulli capital under a
fixed bet, the Gaussian split likelihood ratio), exact enumeration of all
binary sequences (unit expectation of the capital under the null, the false
discovery rate of e-BH for discrete e-values), agreement with the library's
Benjamini-Hochberg adjustment, Gauss-Hermite integration, and seeded Monte
Carlo checks of Ville's inequality against binomial error bands.
"""

from __future__ import annotations

import itertools
import math
import unittest

from probstat.foundations import RandomStream
from probstat.inference import (
    BettingTestResult,
    ConfidenceSequence,
    EBenjaminiHochbergResult,
    SplitLikelihoodRatioResult,
    adjust_p_values,
    betting_confidence_sequence,
    betting_mean_test,
    calibrate_p_value,
    e_benjamini_hochberg,
    e_to_p_value,
    merge_e_values,
    paired_betting_test,
    split_likelihood_ratio_test,
    wilson_score_interval,
)
from probstat.modeling import gauss_hermite


def _bernoulli_expectation(
    probability: float, horizon: int, statistic: object
) -> float:
    """Return ``E[statistic(X_1..X_t)]`` for iid Bernoulli data by enumeration."""
    assert callable(statistic)
    total = 0.0
    for outcome in itertools.product((0.0, 1.0), repeat=horizon):
        ones = sum(outcome)
        weight = probability**ones * (1.0 - probability) ** (horizon - ones)
        total += weight * statistic(outcome)
    return total


def _normal_log_likelihood(theta: float, part: object) -> float:
    """Return the N(theta, 1) joint log-density of a data part."""
    assert isinstance(part, tuple)
    return sum(
        -0.5 * math.log(2.0 * math.pi) - 0.5 * (value - theta) ** 2 for value in part
    )


def _mean(part: object) -> float:
    """Return the sample mean of a data part."""
    assert isinstance(part, tuple)
    return math.fsum(part) / len(part)


class CalibrationTests(unittest.TestCase):
    """Calibrators between p-values and e-values."""

    def test_power_calibrator_integrates_to_one(self) -> None:
        """Midpoint quadrature in ``s = -log p`` recovers a unit integral."""
        for kappa in (0.2, 0.5, 0.9):
            # integral_0^1 f(p) dp = integral_0^inf f(e^-s) e^-s ds.
            upper, cells = 60.0 / kappa, 200_000
            step = upper / cells
            total = step * math.fsum(
                calibrate_p_value(math.exp(-(i + 0.5) * step), kappa=kappa)
                * math.exp(-(i + 0.5) * step)
                for i in range(cells)
            )
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_mixture_calibrator_equals_integral_over_kappa(self) -> None:
        """The closed form equals the kappa-average of the power calibrators."""
        cells = 20_000
        for p in (1e-6, 0.003, 0.05, 0.4, 0.97, 1.0 - 5e-5, 1.0 - 2e-4):
            average = (
                math.fsum(
                    calibrate_p_value(p, kappa=(i + 0.5) / cells) for i in range(cells)
                )
                / cells
            )
            self.assertAlmostEqual(calibrate_p_value(p) / average, 1.0, places=7)

    def test_mixture_calibrator_partial_integral(self) -> None:
        """The integral over ``[eps, 1]`` equals ``1 - (1 - eps)/log(1/eps)``."""
        upper, cells = 30.0, 300_000
        step = upper / cells
        total = step * math.fsum(
            calibrate_p_value(math.exp(-(i + 0.5) * step)) * math.exp(-(i + 0.5) * step)
            for i in range(cells)
        )
        epsilon = math.exp(-upper)
        self.assertAlmostEqual(total, 1.0 - (1.0 - epsilon) / upper, places=8)
        self.assertLess(total, 1.0)

    def test_mixture_calibrator_limits_and_monotonicity(self) -> None:
        """``F(1) = 1/2``, ``F(0) = inf`` and ``F`` decreases."""
        self.assertEqual(calibrate_p_value(1.0), 0.5)
        self.assertEqual(calibrate_p_value(0.0), math.inf)
        self.assertEqual(calibrate_p_value(0.0, kappa=0.5), math.inf)
        self.assertAlmostEqual(calibrate_p_value(0.25, kappa=0.5), 1.0)
        grid = [i / 1000 for i in range(1, 1001)]
        values = [calibrate_p_value(p) for p in grid]
        self.assertTrue(all(a > b for a, b in itertools.pairwise(values)))

    def test_markov_bound_is_exact_for_calibrated_uniform(self) -> None:
        """``P(F(U) >= 1/alpha) = F^-1(1/alpha) <= alpha`` for uniform ``U``."""
        for alpha in (0.2, 0.05, 0.01, 0.001):
            for kappa in (None, 0.3, 0.7):
                low, high = 0.0, 1.0
                for _ in range(200):
                    middle = 0.5 * (low + high)
                    if calibrate_p_value(middle, kappa=kappa) >= 1.0 / alpha:
                        low = middle
                    else:
                        high = middle
                self.assertLessEqual(low, alpha)
                self.assertGreater(low, 0.0)

    def test_mixture_calibrator_of_a_subnormal_p_value(self) -> None:
        """P-values below ``exp(-709)`` give infinity instead of overflowing."""
        self.assertEqual(calibrate_p_value(1.0e-320), math.inf)
        self.assertEqual(calibrate_p_value(5.0e-324), math.inf)
        self.assertGreater(calibrate_p_value(1.0e-300), 1.0e290)

    def test_e_to_p_value(self) -> None:
        """``min(1, 1/e)`` with its boundary cases."""
        self.assertEqual(e_to_p_value(0.0), 1.0)
        self.assertEqual(e_to_p_value(0.5), 1.0)
        self.assertEqual(e_to_p_value(20.0), 0.05)
        self.assertEqual(e_to_p_value(math.inf), 0.0)
        # The round trip p -> e -> p never gains evidence.
        for p in (0.001, 0.01, 0.05, 0.3):
            self.assertGreater(e_to_p_value(calibrate_p_value(p)), p)

    def test_validation(self) -> None:
        """Invalid p-values, exponents, and e-values are rejected."""
        with self.assertRaises(ValueError):
            calibrate_p_value(1.2)
        with self.assertRaises(ValueError):
            calibrate_p_value(0.2, kappa=1.0)
        with self.assertRaises(ValueError):
            calibrate_p_value(0.2, kappa=0.0)
        with self.assertRaises(ValueError):
            e_to_p_value(-1.0)
        with self.assertRaises(ValueError):
            e_to_p_value(math.nan)
        with self.assertRaises(TypeError):
            e_to_p_value("3")  # type: ignore[arg-type]


class MergingTests(unittest.TestCase):
    """Arithmetic-mean and product merging."""

    def test_mean_and_product(self) -> None:
        """Both merging rules reproduce hand computations."""
        self.assertAlmostEqual(merge_e_values((1.0, 4.0, 10.0)), 5.0)
        self.assertAlmostEqual(
            merge_e_values((1.0, 4.0, 10.0), weights=(0.5, 0.25, 0.25)), 4.0
        )
        self.assertAlmostEqual(merge_e_values((2.0, 4.0, 0.5), method="product"), 4.0)
        self.assertEqual(merge_e_values((0.0, math.inf), method="product"), 0.0)
        self.assertEqual(merge_e_values((0.0, math.inf)), math.inf)
        self.assertEqual(merge_e_values((3.0, math.inf), weights=(1.0, 0.0)), 3.0)

    def test_mean_is_valid_under_dependence_but_product_is_not(self) -> None:
        """For ``E_1 = E_2 = 2 * 1{U < 1/2}`` the mean has expectation 1."""
        outcomes = ((2.0, 0.5), (0.0, 0.5))
        mean = sum(w * merge_e_values((e, e)) for e, w in outcomes)
        product = sum(w * merge_e_values((e, e), method="product") for e, w in outcomes)
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(product, 2.0)
        # Independent copies: the product has expectation exactly one.
        independent = sum(
            w1 * w2 * merge_e_values((e1, e2), method="product")
            for e1, w1 in outcomes
            for e2, w2 in outcomes
        )
        self.assertAlmostEqual(independent, 1.0)

    def test_validation(self) -> None:
        """Empty families, bad weights, and unknown methods are rejected."""
        with self.assertRaises(ValueError):
            merge_e_values(())
        with self.assertRaises(ValueError):
            merge_e_values((1.0, 2.0), weights=(0.5, 0.4))
        with self.assertRaises(ValueError):
            merge_e_values((1.0, 2.0), weights=(1.0,))
        with self.assertRaises(ValueError):
            merge_e_values((1.0, 2.0), method="product", weights=(0.5, 0.5))
        with self.assertRaises(ValueError):
            merge_e_values((1.0, -2.0))
        with self.assertRaises(ValueError):
            merge_e_values((1.0, 2.0), method="median")  # type: ignore[arg-type]


class EBenjaminiHochbergTests(unittest.TestCase):
    """The e-BH procedure."""

    def test_reciprocal_p_values_reproduce_benjamini_hochberg(self) -> None:
        """With ``e_k = 1/p_k`` the rejection set is that of BH."""
        generator = RandomStream(11).generator()
        nonempty = 0
        for trial in range(400):
            count = 1 + trial % 25
            alpha = (0.01, 0.05, 0.1, 0.25)[trial % 4]
            p_values = [
                generator.random() ** (1 + 3 * (generator.random() < 0.4))
                * (0.02 if generator.random() < 0.3 else 1.0)
                for _ in range(count)
            ]
            bh = adjust_p_values(p_values, method="bh", alpha=alpha)
            ebh = e_benjamini_hochberg([1.0 / p for p in p_values], alpha=alpha)
            self.assertEqual(ebh.rejected, bh.rejected)
            self.assertEqual(ebh.rejection_count, bh.rejection_count)
            nonempty += ebh.rejection_count > 0
        self.assertGreater(nonempty, 100)

    def test_hand_example(self) -> None:
        """A four-hypothesis family worked by hand."""
        result = e_benjamini_hochberg((5.0, 100.0, 1.0, 45.0), alpha=0.1)
        # K/alpha = 40: k=1: 100 >= 40; k=2: 90 >= 40; k=3: 15 < 40; k=4: 4.
        self.assertIsInstance(result, EBenjaminiHochbergResult)
        self.assertEqual(result.rejected, (False, True, False, True))
        self.assertEqual(result.rejection_count, 2)
        self.assertAlmostEqual(result.threshold, 20.0)
        none = e_benjamini_hochberg((1.0, 2.0), alpha=0.1)
        self.assertEqual(none.rejection_count, 0)
        self.assertEqual(none.threshold, math.inf)

    def test_fdr_bound_by_enumeration(self) -> None:
        """Exact FDR of discrete e-values is at most ``alpha K_0 / K``."""
        null_law = ((0.0, 0.6), (2.0, 0.3), (4.0, 0.1))
        self.assertAlmostEqual(sum(e * w for e, w in null_law), 1.0)
        alpha = 0.5
        for signal in (0.0, 3.0, 30.0):
            fdr = 0.0
            for (e1, w1), (e2, w2) in itertools.product(null_law, repeat=2):
                result = e_benjamini_hochberg((e1, e2, signal), alpha=alpha)
                false = sum(result.rejected[:2])
                if result.rejection_count:
                    fdr += w1 * w2 * false / result.rejection_count
            self.assertLessEqual(fdr, alpha * 2.0 / 3.0 + 1e-12)

    def test_fdr_bound_is_attained_under_perfect_dependence(self) -> None:
        """Identical null e-values equal to ``1/alpha`` w.p. ``alpha``."""
        alpha = 0.05
        outcomes = ((1.0 / alpha, alpha), (0.0, 1.0 - alpha))
        self.assertAlmostEqual(sum(e * w for e, w in outcomes), 1.0)
        fdr = sum(
            w * (e_benjamini_hochberg((e,) * 6, alpha=alpha).rejection_count > 0)
            for e, w in outcomes
        )
        self.assertAlmostEqual(fdr, alpha)

    def test_validation(self) -> None:
        """Empty families and invalid levels are rejected."""
        with self.assertRaises(ValueError):
            e_benjamini_hochberg(())
        with self.assertRaises(ValueError):
            e_benjamini_hochberg((1.0,), alpha=0.0)
        with self.assertRaises(ValueError):
            e_benjamini_hochberg((1.0, -0.1))


class BettingTestTests(unittest.TestCase):
    """Capital processes for a bounded mean."""

    def test_fixed_bet_closed_form(self) -> None:
        """Bernoulli data: ``K = (1 + l(1-m))^s (1 - l m)^(t-s)``."""
        data = (1, 0, 1, 1, 0, 1, 1, 1, 0, 1)
        m, bet = 0.4, 0.8
        result = betting_mean_test(
            data, m, alternative="greater", bet="fixed", fixed_bet=bet
        )
        self.assertIsInstance(result, BettingTestResult)
        successes = 0
        for time, value in enumerate(data, start=1):
            successes += value
            expected = (1 + bet * (1 - m)) ** successes * (1 - bet * m) ** (
                time - successes
            )
            self.assertAlmostEqual(result.capital[time - 1], expected, places=12)
        self.assertEqual(result.positive_bets, (bet,) * 10)
        self.assertEqual(result.negative_bets, (0.0,) * 10)
        self.assertAlmostEqual(result.e_value, result.capital[-1])
        less = betting_mean_test(
            data, m, alternative="less", bet="fixed", fixed_bet=bet
        )
        self.assertAlmostEqual(
            less.e_value, (1 - bet * (1 - m)) ** 7 * (1 + bet * m) ** 3, places=12
        )

    def test_fixed_bet_is_truncated(self) -> None:
        """A bet above ``c/m`` is cut to ``c/m``."""
        result = betting_mean_test(
            (1.0, 0.0), 0.5, alternative="greater", bet="fixed", fixed_bet=5.0
        )
        self.assertEqual(result.positive_bets, (1.0, 1.0))
        self.assertAlmostEqual(result.e_value, 1.5 * 0.5)

    def test_unit_expectation_under_the_null_by_enumeration(self) -> None:
        """``E[K_t] = 1`` exactly for Bernoulli(m) data and predictable bets."""
        horizon = 10
        for m in (0.2, 0.5, 0.85):
            for kwargs in (
                {"bet": "agrapa", "alternative": "two-sided"},
                {"bet": "agrapa", "alternative": "greater"},
                {"bet": "plug-in", "alternative": "greater"},
                {"bet": "plug-in", "alternative": "less"},
                {"bet": "fixed", "fixed_bet": 0.7, "alternative": "less"},
            ):

                def capital(outcome: tuple[float, ...]) -> float:
                    """Return the final capital for one binary sequence."""
                    return betting_mean_test(outcome, m, **kwargs).e_value  # noqa: B023

                expectation = _bernoulli_expectation(m, horizon, capital)
                self.assertAlmostEqual(expectation, 1.0, places=12)

    def test_hedged_capital_is_dominated_by_a_martingale(self) -> None:
        """``max(theta K+, (1-theta) K-)`` has expectation at most one."""
        m, theta = 0.3, 0.35

        def hedged(outcome: tuple[float, ...]) -> float:
            """Return the final hedged capital for one binary sequence."""
            return betting_mean_test(
                outcome, m, bet="fixed", fixed_bet=0.6, theta=theta
            ).e_value

        self.assertLess(_bernoulli_expectation(m, 9, hedged), 1.0)
        data = (1, 1, 0, 1, 0, 0, 1, 1, 1)
        both = betting_mean_test(data, m, bet="fixed", fixed_bet=0.6, theta=theta)
        up = betting_mean_test(
            data, m, bet="fixed", fixed_bet=0.6, alternative="greater"
        )
        down = betting_mean_test(
            data, m, bet="fixed", fixed_bet=0.6, alternative="less"
        )
        for index in range(len(data)):
            expected = max(theta * up.capital[index], (1 - theta) * down.capital[index])
            self.assertAlmostEqual(both.capital[index], expected, places=12)

    def test_composite_null_is_a_supermartingale(self) -> None:
        """For a true mean below ``m0`` the upward capital shrinks on average."""
        for bet in ("agrapa", "plug-in"):

            def capital(outcome: tuple[float, ...]) -> float:
                """Return the final upward capital for one binary sequence."""
                return betting_mean_test(
                    outcome,
                    0.5,
                    alternative="greater",
                    bet=bet,  # type: ignore[arg-type]  # noqa: B023
                ).e_value

            self.assertLess(_bernoulli_expectation(0.4, 9, capital), 1.0)
            self.assertGreater(_bernoulli_expectation(0.7, 9, capital), 1.0)

    def test_bets_are_predictable(self) -> None:
        """The bet on observation ``t`` ignores observations ``t, t+1, ...``."""
        generator = RandomStream(5).generator()
        data = [generator.random() for _ in range(40)]
        changed = list(data)
        changed[25:] = [generator.random() for _ in range(15)]
        for bet in ("agrapa", "plug-in"):
            first = betting_mean_test(data, 0.45, bet=bet)  # type: ignore[arg-type]
            second = betting_mean_test(changed, 0.45, bet=bet)  # type: ignore[arg-type]
            self.assertEqual(first.positive_bets[:26], second.positive_bets[:26])
            self.assertEqual(first.negative_bets[:26], second.negative_bets[:26])
            self.assertEqual(first.log_capital[:25], second.log_capital[:25])
            self.assertNotEqual(first.log_capital[25:], second.log_capital[25:])

    def test_stopping_time_and_p_value(self) -> None:
        """The first crossing of ``1/alpha`` and ``min(1, 1/max K)``."""
        generator = RandomStream(3).generator()
        data = [float(generator.random() < 0.7) for _ in range(400)]
        result = betting_mean_test(data, 0.5, alternative="greater", alpha=0.01)
        self.assertTrue(result.rejected)
        assert result.stopping_time is not None
        self.assertGreaterEqual(result.capital[result.stopping_time - 1], 100.0)
        self.assertTrue(
            all(k < 100.0 for k in result.capital[: result.stopping_time - 1])
        )
        self.assertAlmostEqual(result.maximum_capital, max(result.capital))
        self.assertAlmostEqual(result.p_value, 1.0 / result.maximum_capital)
        self.assertEqual(result.sample_size, 400)
        wrong_side = betting_mean_test(data, 0.5, alternative="less", alpha=0.01)
        self.assertFalse(wrong_side.rejected)
        self.assertIsNone(wrong_side.stopping_time)
        self.assertEqual(wrong_side.p_value, 1.0)

    def test_ville_bound_and_naive_interval_contrast(self) -> None:
        """Ever-rejection frequency is below alpha; repeated Wilson is not."""
        alpha, horizon, replicates, truth = 0.05, 1000, 1000, 0.3
        stream = RandomStream(2024)
        crossings = {"agrapa": 0, "plug-in": 0}
        naive = 0
        for replicate in range(replicates):
            generator = stream.child("replicate", replicate).generator()
            data = [float(generator.random() < truth) for _ in range(horizon)]
            for bet in crossings:
                result = betting_mean_test(
                    data,
                    truth,
                    alpha=alpha,
                    bet=bet,  # type: ignore[arg-type]
                )
                crossings[bet] += result.rejected
            successes = 0
            for time, value in enumerate(data, start=1):
                successes += int(value)
                # Wilson excludes the truth iff the score test rejects.
                z = (successes - time * truth) / math.sqrt(time * truth * (1 - truth))
                if time >= 10 and abs(z) > 1.959963984540054:
                    naive += 1
                    break
        band = alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / replicates)
        for count in crossings.values():
            self.assertLessEqual(count / replicates, band)
        self.assertGreater(naive / replicates, 0.25)

    def test_score_test_matches_wilson_interval(self) -> None:
        """The z criterion used above is the Wilson interval's exclusion."""
        truth = 0.3
        for successes, trials in ((3, 30), (15, 30), (40, 100), (22, 100), (9, 12)):
            interval = wilson_score_interval(successes, trials)
            z = (successes - trials * truth) / math.sqrt(trials * truth * (1 - truth))
            excluded = not interval.lower <= truth <= interval.upper
            self.assertEqual(excluded, abs(z) > 1.959963984540054)

    def test_rescaling_of_bounds(self) -> None:
        """An affine change of units leaves the capital unchanged."""
        generator = RandomStream(8).generator()
        data = [generator.random() for _ in range(60)]
        scaled = [10.0 + 4.0 * value for value in data]
        for bet in ("agrapa", "plug-in"):
            unit = betting_mean_test(data, 0.4, bet=bet)  # type: ignore[arg-type]
            other = betting_mean_test(
                scaled,
                11.6,
                bet=bet,  # type: ignore[arg-type]
                lower=10.0,
                upper=14.0,
            )
            for a, b in zip(unit.log_capital, other.log_capital, strict=True):
                self.assertAlmostEqual(a, b, places=9)

    def test_boundary_null_means(self) -> None:
        """Null means on the boundary of the range give finite capital."""
        zero = betting_mean_test((0.0, 0.0, 0.2), 0.0)
        self.assertTrue(all(math.isfinite(v) for v in zero.log_capital))
        self.assertGreater(zero.e_value, 1.0)
        one = betting_mean_test((1.0, 1.0, 1.0), 1.0, bet="plug-in")
        self.assertTrue(all(math.isfinite(v) for v in one.log_capital))

    def test_validation(self) -> None:
        """Out-of-range data and bad options are rejected."""
        with self.assertRaises(ValueError):
            betting_mean_test((0.2, 1.4), 0.5)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2, 0.4), 1.5)
        with self.assertRaises(ValueError):
            betting_mean_test((), 0.5)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, bet="fixed")
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, bet="fixed", fixed_bet=-0.1)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, fixed_bet=0.1)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, bet="kelly")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, truncation=1.0)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, theta=0.0)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, alpha=1.0)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, lower=1.0, upper=0.0)
        with self.assertRaises(ValueError):
            betting_mean_test((0.2,), 0.5, alternative="up")  # type: ignore[arg-type]


class PairedBettingTests(unittest.TestCase):
    """Sequential paired comparison."""

    def test_matches_the_test_on_rescaled_differences(self) -> None:
        """Differences in [-1, 1] map to ``(d + 1)/2`` in the unit interval."""
        generator = RandomStream(21).generator()
        first = [generator.random() for _ in range(80)]
        second = [0.8 * generator.random() for _ in range(80)]
        paired = paired_betting_test(first, second)
        unit = betting_mean_test(
            [(a - b + 1.0) / 2.0 for a, b in zip(first, second, strict=True)],
            0.5,
            alternative="greater",
        )
        self.assertEqual(paired.null_mean, 0.0)
        self.assertEqual(paired.alternative, "greater")
        for a, b in zip(paired.log_capital, unit.log_capital, strict=True):
            self.assertAlmostEqual(a, b, places=10)

    def test_detects_the_better_stream(self) -> None:
        """A clear difference is detected; the reverse direction is not."""
        generator = RandomStream(22).generator()
        first = [float(generator.random() < 0.8) for _ in range(300)]
        second = [float(generator.random() < 0.5) for _ in range(300)]
        self.assertTrue(paired_betting_test(first, second).rejected)
        self.assertFalse(paired_betting_test(second, first).rejected)

    def test_validation(self) -> None:
        """Unequal lengths and differences beyond the bound are rejected."""
        with self.assertRaises(ValueError):
            paired_betting_test((0.1, 0.2), (0.1,))
        with self.assertRaises(ValueError):
            paired_betting_test((2.0,), (0.0,))
        with self.assertRaises(ValueError):
            paired_betting_test((0.2,), (0.1,), difference_bound=0.0)


class ConfidenceSequenceTests(unittest.TestCase):
    """Betting confidence sequences on a grid."""

    def setUp(self) -> None:
        """Draw one seeded sample on the unit interval."""
        generator = RandomStream(31).generator()
        self.data = [generator.betavariate(2.0, 3.0) for _ in range(150)]

    def test_hedged_sequence_agrees_with_pointwise_tests(self) -> None:
        """Grid means never rejected by the hedged test form the interval."""
        grid_size, alpha = 20, 0.1
        for kwargs in ({"bet": "plug-in"}, {"bet": "fixed", "fixed_bet": 0.4}):
            sequence = betting_confidence_sequence(
                self.data,
                alpha=alpha,
                grid_size=grid_size,
                **kwargs,  # type: ignore[arg-type]
            )
            self.assertIsInstance(sequence, ConfidenceSequence)
            crossing: list[float] = []
            for index in range(grid_size + 1):
                test = betting_mean_test(
                    self.data,
                    index / grid_size,
                    alpha=alpha,
                    **kwargs,  # type: ignore[arg-type]
                )
                crossing.append(
                    math.inf if test.stopping_time is None else test.stopping_time
                )
            for time in (1, 5, 20, 60, 150):
                low, high = sequence.interval(time)
                for index in range(grid_size + 1):
                    point = index / grid_size
                    survives = crossing[index] > time
                    if survives:
                        self.assertTrue(low <= point <= high)
                    if low + 1e-12 < point < high - 1e-12:
                        self.assertTrue(survives)

    def test_running_intersection_is_nested_and_tighter(self) -> None:
        """Intersected intervals never widen and lie inside the raw ones."""
        nested = betting_confidence_sequence(self.data, grid_size=100)
        raw = betting_confidence_sequence(
            self.data, grid_size=100, running_intersection=False
        )
        for time in range(1, len(self.data)):
            self.assertGreaterEqual(nested.lower[time], nested.lower[time - 1])
            self.assertLessEqual(nested.upper[time], nested.upper[time - 1])
            self.assertGreaterEqual(nested.lower[time], raw.lower[time])
            self.assertLessEqual(nested.upper[time], raw.upper[time])
        self.assertEqual(nested.lower[-1], max(raw.lower))
        self.assertEqual(nested.upper[-1], min(raw.upper))
        self.assertIsNone(nested.first_empty_time)
        low, high = nested.final
        self.assertTrue(low <= 0.4 <= high)
        self.assertLess(high - low, 0.35)
        self.assertEqual(nested.width[-1], high - low)
        self.assertAlmostEqual(nested.resolution, 0.01)

    def test_finer_grid_only_moves_endpoints_by_one_cell(self) -> None:
        """Coarse endpoints enclose fine ones within one coarse cell."""
        coarse = betting_confidence_sequence(self.data, grid_size=20)
        fine = betting_confidence_sequence(self.data, grid_size=400)
        for time in (10, 50, 150):
            self.assertLessEqual(coarse.lower[time - 1], fine.lower[time - 1] + 1e-12)
            self.assertGreaterEqual(
                coarse.upper[time - 1], fine.upper[time - 1] - 1e-12
            )
            self.assertLess(fine.lower[time - 1] - coarse.lower[time - 1], 0.05 + 1e-12)
            self.assertLess(coarse.upper[time - 1] - fine.upper[time - 1], 0.05 + 1e-12)

    def test_agrapa_sequence_contains_surviving_grid_points(self) -> None:
        """Grid means never rejected by the aGRAPA test are reported."""
        grid_size, alpha = 25, 0.1
        sequence = betting_confidence_sequence(
            self.data, alpha=alpha, grid_size=grid_size, bet="agrapa"
        )
        low, high = sequence.final
        survivors = [
            index / grid_size
            for index in range(grid_size + 1)
            if not betting_mean_test(
                self.data, index / grid_size, alpha=alpha, bet="agrapa"
            ).rejected
        ]
        self.assertTrue(survivors)
        self.assertAlmostEqual(low, max(0.0, min(survivors) - 1 / grid_size))
        self.assertAlmostEqual(high, min(1.0, max(survivors) + 1 / grid_size))
        self.assertTrue(low <= 0.4 <= high)

    def test_affine_bounds(self) -> None:
        """Endpoints transform with the units of the observations."""
        unit = betting_confidence_sequence(self.data, grid_size=50)
        scaled = betting_confidence_sequence(
            [-2.0 + 5.0 * value for value in self.data],
            grid_size=50,
            lower=-2.0,
            upper=3.0,
        )
        for a, b in zip(unit.lower, scaled.lower, strict=True):
            self.assertAlmostEqual(-2.0 + 5.0 * a, b, places=9)
        for a, b in zip(unit.upper, scaled.upper, strict=True):
            self.assertAlmostEqual(-2.0 + 5.0 * a, b, places=9)
        self.assertAlmostEqual(scaled.resolution, 0.1)

    def test_changing_mean_empties_the_sequence(self) -> None:
        """A mean that jumps from 0.05 to 0.95 leaves no candidate."""
        data = [0.05] * 150 + [0.95] * 300
        for bet in ("plug-in", "agrapa"):
            sequence = betting_confidence_sequence(
                data,
                grid_size=50,
                bet=bet,  # type: ignore[arg-type]
            )
            self.assertIsNotNone(sequence.first_empty_time)
            self.assertTrue(math.isnan(sequence.lower[-1]))
            self.assertTrue(math.isnan(sequence.upper[-1]))

    def test_agrapa_sequence_keeps_a_mean_between_grid_points(self) -> None:
        """A sequence narrower than one grid cell is not reported as empty.

        Constant data have conditional mean 0.325 at all times, the aGRAPA
        capital at 0.325 stays at one, and both neighbouring grid points
        (0.30 and 0.35) are eventually rejected. The reported interval must
        still contain 0.325 at every time.
        """
        truth = 0.325
        data = [truth] * 300
        self.assertFalse(betting_mean_test(data, truth, bet="agrapa").rejected)
        for running_intersection in (True, False):
            sequence = betting_confidence_sequence(
                data,
                grid_size=20,
                bet="agrapa",
                running_intersection=running_intersection,
            )
            self.assertIsNone(sequence.first_empty_time)
            for low, high in zip(sequence.lower, sequence.upper, strict=True):
                self.assertTrue(low <= truth <= high)
            self.assertLessEqual(sequence.final[1] - sequence.final[0], 0.1 + 1e-12)

    def test_agrapa_coverage_on_a_coarse_grid(self) -> None:
        """Bernoulli(0.983) on a 20-cell grid: misses stay near the test's.

        Before the cell next to the smallest capital was kept, the reported
        sequence lost the true mean in more than half of such runs although
        the capital at the true mean had not crossed its boundary.
        """
        alpha, replicates, horizon, truth = 0.05, 60, 300, 0.983
        stream = RandomStream(5)
        misses = crossings = 0
        for replicate in range(replicates):
            generator = stream.child("coarse", replicate).generator()
            data = [float(generator.random() < truth) for _ in range(horizon)]
            sequence = betting_confidence_sequence(
                data, alpha=alpha, grid_size=20, bet="agrapa"
            )
            low, high = sequence.final
            misses += not low <= truth <= high
            crossings += betting_mean_test(
                data, truth, alpha=alpha, bet="agrapa"
            ).rejected
        self.assertLessEqual(misses, crossings)

    def test_time_uniform_coverage(self) -> None:
        """The true mean leaves the sequence in at most alpha of the runs."""
        alpha, replicates, horizon, truth = 0.1, 150, 120, 0.25
        stream = RandomStream(77)
        misses = 0
        for replicate in range(replicates):
            generator = stream.child("coverage", replicate).generator()
            data = [float(generator.random() < truth) for _ in range(horizon)]
            sequence = betting_confidence_sequence(data, alpha=alpha, grid_size=40)
            low, high = sequence.final
            misses += not low <= truth <= high
        band = alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / replicates)
        self.assertLessEqual(misses / replicates, band)

    def test_validation(self) -> None:
        """Invalid grids, times, and flags are rejected."""
        with self.assertRaises(ValueError):
            betting_confidence_sequence(self.data, grid_size=1)
        with self.assertRaises(ValueError):
            betting_confidence_sequence(self.data, bet="fixed")
        with self.assertRaises(TypeError):
            betting_confidence_sequence(
                self.data,
                running_intersection=1,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            betting_confidence_sequence((0.5, 2.0))
        sequence = betting_confidence_sequence(self.data[:5], grid_size=10)
        with self.assertRaises(ValueError):
            sequence.interval(0)
        with self.assertRaises(ValueError):
            sequence.interval(6)


class SplitLikelihoodRatioTests(unittest.TestCase):
    """Universal inference."""

    def test_gaussian_closed_form(self) -> None:
        """N(theta, 1), ``H0: theta = 0``: ``T = exp(t1 S0 - n0 t1^2 / 2)``."""
        generator = RandomStream(41).generator()
        data = tuple(generator.gauss(0.4, 1.0) for _ in range(30))
        result = split_likelihood_ratio_test(
            data,
            _normal_log_likelihood,
            _mean,
            lambda part: 0.0,
            shuffle=False,
            estimation_fraction=0.4,
        )
        self.assertIsInstance(result, SplitLikelihoodRatioResult)
        theta_1 = math.fsum(data[:12]) / 12
        expected = theta_1 * math.fsum(data[12:]) - 18 * theta_1**2 / 2
        self.assertAlmostEqual(result.log_e_value, expected, places=10)
        self.assertAlmostEqual(result.e_value, math.exp(expected), places=10)
        self.assertEqual((result.estimation_size, result.inference_size), (12, 18))
        self.assertEqual(result.p_value, min(1.0, 1.0 / result.e_value))
        self.assertEqual(result.rejected, result.e_value >= 20.0)
        self.assertEqual((result.likelihood_evaluations, result.fits), (2, 2))

    def test_conditional_expectation_is_one_by_gauss_hermite(self) -> None:
        """``E_H0[T | D1] = 1`` for one and two inference observations."""
        nodes, weights = gauss_hermite(40)
        estimation = (0.7, -0.1, 1.2)

        def statistic(inference: tuple[float, ...]) -> float:
            """Return ``T`` for a fixed ``D1`` and the given ``D0``."""
            return split_likelihood_ratio_test(
                estimation + inference,
                _normal_log_likelihood,
                _mean,
                lambda part: 0.0,
                shuffle=False,
                estimation_fraction=3 / (3 + len(inference)),
            ).e_value

        one = math.fsum(
            w * statistic((x,)) for x, w in zip(nodes, weights, strict=True)
        )
        self.assertAlmostEqual(one, 1.0, places=10)
        two = math.fsum(
            w1 * w2 * statistic((x1, x2))
            for x1, w1 in zip(nodes, weights, strict=True)
            for x2, w2 in zip(nodes, weights, strict=True)
        )
        self.assertAlmostEqual(two, 1.0, places=9)

    def test_cross_fit_is_the_average_of_both_splits(self) -> None:
        """Swapping the halves and averaging reproduces the two statistics."""
        generator = RandomStream(42).generator()
        data = tuple(generator.gauss(0.3, 1.0) for _ in range(24))
        forward = split_likelihood_ratio_test(
            data, _normal_log_likelihood, _mean, lambda part: 0.0, shuffle=False
        )
        backward = split_likelihood_ratio_test(
            data[12:] + data[:12],
            _normal_log_likelihood,
            _mean,
            lambda part: 0.0,
            shuffle=False,
        )
        both = split_likelihood_ratio_test(
            data,
            _normal_log_likelihood,
            _mean,
            lambda part: 0.0,
            shuffle=False,
            cross_fit=True,
        )
        self.assertAlmostEqual(
            both.e_value, 0.5 * (forward.e_value + backward.e_value), places=10
        )
        self.assertEqual(len(both.log_statistics), 2)
        self.assertEqual(both.likelihood_evaluations, 4)

    def test_composite_null_level_and_power(self) -> None:
        """``H0: theta <= 0`` holds its level at the boundary and has power."""
        stream = RandomStream(43)
        alpha, replicates = 0.1, 400

        def null_fit(part: object) -> float:
            """Return the MLE of theta restricted to ``theta <= 0``."""
            return min(0.0, _mean(part))

        rejections = {0.0: 0, 0.8: 0}
        for truth in rejections:
            for replicate in range(replicates):
                generator = stream.child(str(truth), replicate).generator()
                data = tuple(generator.gauss(truth, 1.0) for _ in range(40))
                result = split_likelihood_ratio_test(
                    data,
                    _normal_log_likelihood,
                    _mean,
                    null_fit,
                    alpha=alpha,
                    cross_fit=True,
                    seed=stream.child("split", replicate),
                )
                rejections[truth] += result.rejected
        band = alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / replicates)
        self.assertLessEqual(rejections[0.0] / replicates, band)
        self.assertGreater(rejections[0.8] / replicates, 0.5)

    def test_very_small_e_value_has_p_value_one(self) -> None:
        """A log e-value below -709 gives p = 1 instead of an overflow."""

        def log_likelihood(theta: float, part: object) -> float:
            """Return the ``N(theta, 1)`` log-likelihood up to its constant."""
            return sum(-0.5 * (x - theta) ** 2 for x in part)  # type: ignore[attr-defined]

        result = split_likelihood_ratio_test(
            [5.0] * 1000,
            log_likelihood,
            lambda part: -5.0,
            lambda part: 5.0,
            shuffle=False,
        )
        self.assertEqual(result.log_e_value, -25000.0)
        self.assertEqual(result.e_value, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.rejected)

    def test_seed_reproducibility_and_shuffle(self) -> None:
        """One seed gives one split; different seeds usually differ."""
        generator = RandomStream(44).generator()
        data = tuple(generator.gauss(0.5, 1.0) for _ in range(20))
        args = (data, _normal_log_likelihood, _mean, lambda part: 0.0)
        first = split_likelihood_ratio_test(*args, seed=1)
        again = split_likelihood_ratio_test(*args, seed=1)
        other = split_likelihood_ratio_test(*args, seed=2)
        self.assertEqual(first, again)
        self.assertNotEqual(first.log_e_value, other.log_e_value)

    def test_zero_alternative_likelihood_and_validation(self) -> None:
        """A zero alternative likelihood gives ``T = 0``; bad inputs raise."""
        data = (0.1, 0.2, 0.3, 0.4)
        zero = split_likelihood_ratio_test(
            data,
            lambda theta, part: -math.inf if theta else 0.0,
            lambda part: 1,
            lambda part: 0,
            shuffle=False,
        )
        self.assertEqual(zero.e_value, 0.0)
        self.assertEqual(zero.p_value, 1.0)
        self.assertFalse(zero.rejected)
        with self.assertRaises(ValueError):
            split_likelihood_ratio_test(
                data,
                lambda theta, part: -math.inf,
                lambda part: 1,
                lambda part: 0,
                shuffle=False,
            )
        with self.assertRaises(ValueError):
            split_likelihood_ratio_test(
                data,
                lambda theta, part: math.nan,
                lambda part: 1,
                lambda part: 0,
                shuffle=False,
            )
        with self.assertRaises(ValueError):
            split_likelihood_ratio_test(
                (1.0,), _normal_log_likelihood, _mean, lambda part: 0.0
            )
        with self.assertRaises(ValueError):
            split_likelihood_ratio_test(
                data,
                _normal_log_likelihood,
                _mean,
                lambda part: 0.0,
                estimation_fraction=1.0,
            )
        with self.assertRaises(TypeError):
            split_likelihood_ratio_test(
                data,
                _normal_log_likelihood,
                _mean,
                lambda part: 0.0,
                cross_fit="yes",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
