"""Tests for the conformal extensions beyond exchangeable split conformal.

Each method is checked against an independent oracle: exact reduction to
split conformal prediction or jackknife+ in a special case, exact
enumeration of ranks, by-hand weighted quantiles, a deterministic telescoping
bound, and coverage or risk guarantees checked within Monte Carlo standard
errors over seeded replicates.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction

from probstat.conformal import (
    OnlineQuantileTracker,
    conformal_outlier_detection,
    conformal_p_values,
    conformal_quantile,
    conformal_rank,
    conformal_risk_control,
    cv_plus_regression,
    jackknife_plus_regression,
    mondrian_conformal_regression,
    nonexchangeable_conformal_quantile,
    nonexchangeable_coverage_gap,
    nonexchangeable_split_conformal_regression,
    online_quantile_intervals,
    split_conformal_regression,
    weighted_conformal_quantile,
    weighted_split_conformal_regression,
)
from probstat.foundations import RandomStream
from probstat.inference import adjust_p_values


def _identity(x: Sequence[float]) -> float:
    """Predict the first coordinate."""
    return x[0]


def _zero(x: Sequence[float]) -> float:
    """Predict zero."""
    return 0.0


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean and its standard error."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def _ols(points: tuple[tuple[float, ...], ...], values: tuple[float, ...]):  # type: ignore[no-untyped-def]
    """Fit a least-squares line and return it as a predictor."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(values) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    slope = sum((p[0] - mx) * (y - my) for p, y in zip(points, values, strict=True))
    slope /= sxx
    return lambda x: my + slope * (x[0] - mx)


def _line_data(
    generator: random.Random, size: int
) -> tuple[list[tuple[float]], list[float]]:
    """Draw a noisy line on ``U(0, 3)`` inputs."""
    xs = [(3.0 * generator.random(),) for _ in range(size)]
    ys = [0.5 + 0.8 * x[0] + generator.gauss(0.0, 0.4) for x in xs]
    return xs, ys


def _brute_weighted_quantile(
    scores: Sequence[float], weights: Sequence[float], test: float, alpha: Fraction
) -> float:
    """Weighted quantile by scanning candidate values in exact arithmetic."""
    total = sum((Fraction(w) for w in weights), Fraction(test))
    for candidate in sorted(scores):
        mass = sum(
            (
                Fraction(w)
                for s, w in zip(scores, weights, strict=True)
                if s <= candidate
            ),
            Fraction(0),
        )
        if mass / total >= 1 - alpha:
            return candidate
    return math.inf


class WeightedQuantileTests(unittest.TestCase):
    """The weighted quantile and its reduction to the split conformal quantile."""

    def test_unit_weights_reproduce_split_conformal_exactly(self) -> None:
        """Equal weights give ``conformal_quantile`` for every size and level."""
        generator = RandomStream(11).generator()
        for n in (1, 2, 5, 9, 10, 19, 20, 57, 99):
            scores = [generator.random() for _ in range(n)]
            for alpha in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9):
                expected = conformal_quantile(scores, alpha)
                for weight in (1.0, 0.3, 7.0):
                    got = weighted_conformal_quantile(
                        scores, [weight] * n, weight, alpha
                    )
                    self.assertEqual(got, expected)
                self.assertEqual(
                    nonexchangeable_conformal_quantile(scores, [1.0] * n, alpha),
                    expected,
                )

    def test_by_hand(self) -> None:
        """Masses (1, 2, 1) plus a test mass 1 give thresholds 0.6 * 5 = 3."""
        scores, weights = [3.0, 1.0, 2.0], [1.0, 1.0, 2.0]
        self.assertEqual(weighted_conformal_quantile(scores, weights, 1.0, 0.4), 2.0)
        self.assertEqual(weighted_conformal_quantile(scores, weights, 1.0, 0.8), 1.0)
        self.assertEqual(weighted_conformal_quantile(scores, weights, 1.0, 0.2), 3.0)
        self.assertEqual(
            weighted_conformal_quantile(scores, weights, 1.0, 0.1), math.inf
        )
        # A test point of zero weight puts no mass at +inf.
        self.assertEqual(weighted_conformal_quantile(scores, weights, 0.0, 0.01), 3.0)
        # A heavy test point pushes the quantile to +inf.
        self.assertEqual(
            weighted_conformal_quantile(scores, weights, 50.0, 0.4), math.inf
        )

    def test_matches_brute_force(self) -> None:
        """Random weights agree with an exact scan over candidate values."""
        generator = RandomStream(12).generator()
        for _ in range(60):
            n = generator.randint(1, 25)
            scores = [round(generator.random(), 2) for _ in range(n)]
            weights = [generator.choice((0.0, generator.random())) for _ in range(n)]
            test = generator.random()
            for alpha in (0.05, 0.1, 0.3, 0.6):
                self.assertEqual(
                    weighted_conformal_quantile(scores, weights, test, alpha),
                    _brute_weighted_quantile(
                        scores, weights, test, Fraction(str(alpha))
                    ),
                )
            unit = [min(w, 1.0) for w in weights]
            self.assertEqual(
                nonexchangeable_conformal_quantile(scores, unit, 0.2),
                _brute_weighted_quantile(scores, unit, 1.0, Fraction(1, 5)),
            )

    def test_validation(self) -> None:
        """Bad weights, sizes, and levels are rejected."""
        with self.assertRaises(ValueError):
            weighted_conformal_quantile([1.0], [-1.0], 1.0, 0.1)
        with self.assertRaises(ValueError):
            weighted_conformal_quantile([1.0], [0.0], 0.0, 0.1)
        with self.assertRaises(ValueError):
            weighted_conformal_quantile([1.0, 2.0], [1.0], 1.0, 0.1)
        with self.assertRaises(ValueError):
            weighted_conformal_quantile([], [], 1.0, 0.1)
        with self.assertRaises(ValueError):
            weighted_conformal_quantile([1.0], [1.0], math.inf, 0.1)
        with self.assertRaises(ValueError):
            weighted_conformal_quantile([1.0], [1.0], 1.0, 1.0)
        with self.assertRaises(ValueError):
            nonexchangeable_conformal_quantile([1.0], [1.5], 0.1)


class CovariateShiftTests(unittest.TestCase):
    """Weighted split conformal under a known Gaussian covariate shift."""

    @staticmethod
    def _ratio(x: Sequence[float]) -> float:
        """Likelihood ratio of ``N(1, 1)`` to ``N(0, 1)``."""
        return math.exp(x[0] - 0.5)

    @staticmethod
    def _sample(
        generator: random.Random, size: int, shift: float
    ) -> tuple[list[tuple[float]], list[float]]:
        """Draw inputs ``N(shift, 1)`` and a response whose noise grows with x."""
        xs = [(generator.gauss(shift, 1.0),) for _ in range(size)]
        ys = [x[0] + generator.gauss(0.0, math.exp(0.5 * x[0])) for x in xs]
        return xs, ys

    def test_unit_weight_function_equals_split_conformal(self) -> None:
        """``w = 1`` gives the split conformal regressor's intervals."""
        generator = RandomStream(21).generator()
        x_cal, y_cal = self._sample(generator, 37, 0.0)
        plain = split_conformal_regression(_identity, x_cal, y_cal, alpha=0.1)
        scale = lambda x: 1.0 + abs(x[0])  # noqa: E731
        scaled = split_conformal_regression(
            _identity, x_cal, y_cal, alpha=0.2, scale=scale
        )
        by_function = weighted_split_conformal_regression(
            _identity, x_cal, y_cal, lambda x: 1.0, alpha=0.1
        )
        by_values = weighted_split_conformal_regression(
            _identity, x_cal, y_cal, [2.0] * 37, alpha=0.2, scale=scale
        )
        self.assertEqual(by_function.scores, plain.scores)
        self.assertAlmostEqual(by_function.effective_sample_size, 37.0)
        for x in ((-1.0,), (0.3,), (2.5,)):
            a, b = by_function.interval(x), plain.interval(x)
            self.assertEqual((a.lower, a.upper, a.prediction), (b.lower, b.upper, x[0]))
            c, d = by_values.interval(x, weight=2.0), scaled.interval(x)
            self.assertEqual((c.lower, c.upper), (d.lower, d.upper))

    def test_coverage_under_shift(self) -> None:
        """Exact weights restore 90% coverage; unweighted intervals undercover."""
        weighted: list[float] = []
        unweighted: list[float] = []
        for stream in RandomStream(22).spawn(300):
            generator = stream.generator()
            x_cal, y_cal = self._sample(generator, 200, 0.0)
            x_test, y_test = self._sample(generator, 25, 1.0)
            shifted = weighted_split_conformal_regression(
                _identity, x_cal, y_cal, self._ratio, alpha=0.1
            )
            plain = split_conformal_regression(_identity, x_cal, y_cal, alpha=0.1)
            pairs = list(zip(x_test, y_test, strict=True))
            weighted.append(
                sum(shifted.interval(x).contains(y) for x, y in pairs) / len(pairs)
            )
            unweighted.append(
                sum(plain.interval(x).contains(y) for x, y in pairs) / len(pairs)
            )
        mean_w, se_w = _mean_se(weighted)
        mean_u, se_u = _mean_se(unweighted)
        self.assertGreaterEqual(mean_w, 0.9 - 3.0 * se_w)
        self.assertLess(mean_w, 0.95)
        self.assertLess(mean_u, 0.9 - 8.0 * se_u)

    def test_values_and_function_agree(self) -> None:
        """Calibration weights as values need the test weight passed in."""
        generator = RandomStream(23).generator()
        x_cal, y_cal = self._sample(generator, 60, 0.0)
        by_function = weighted_split_conformal_regression(
            _identity, x_cal, y_cal, self._ratio, alpha=0.1
        )
        by_values = weighted_split_conformal_regression(
            _identity, x_cal, y_cal, [5.0 * self._ratio(x) for x in x_cal], alpha=0.1
        )
        xs = [(-0.5,), (0.8,), (1.9,)]
        ratios = [5.0 * self._ratio(x) for x in xs]
        for a, b in zip(
            by_function.intervals(xs),
            by_values.intervals(xs, weights=ratios),
            strict=True,
        ):
            self.assertEqual((a.lower, a.upper), (b.lower, b.upper))
        self.assertLess(by_function.effective_sample_size, 60.0)
        self.assertTrue(math.isinf(by_function.interval((9.0,)).upper))
        with self.assertRaises(ValueError):
            by_values.interval((0.0,))
        with self.assertRaises(ValueError):
            by_values.intervals(xs, weights=[1.0])
        with self.assertRaises(ValueError):
            weighted_split_conformal_regression(
                _identity, x_cal, y_cal, [1.0] * 59, alpha=0.1
            )
        with self.assertRaises(ValueError):
            weighted_split_conformal_regression(
                _identity, x_cal, y_cal, lambda x: -1.0, alpha=0.1
            )


class NonexchangeableTests(unittest.TestCase):
    """Fixed data-independent weights and the documented coverage gap."""

    def test_unit_weights_equal_split_conformal(self) -> None:
        """``w = 1`` gives the split conformal quantile and intervals."""
        generator = RandomStream(31).generator()
        x_cal, y_cal = _line_data(generator, 29)
        plain = split_conformal_regression(
            _ols(tuple(x_cal), tuple(y_cal)), x_cal, y_cal
        )
        fixed = nonexchangeable_split_conformal_regression(
            plain.predictor, x_cal, y_cal, [1.0] * 29
        )
        self.assertEqual(fixed.quantile(1.0), plain.quantile)
        for x in ((0.2,), (2.2,)):
            a, b = fixed.interval(x), plain.interval(x)
            self.assertEqual((a.lower, a.upper), (b.lower, b.upper))

    def test_coverage_for_exchangeable_data(self) -> None:
        """With i.i.d. data the gap is zero: decaying weights still cover."""
        weights = [0.97 ** (60 - i) for i in range(1, 61)]
        rates: list[float] = []
        for stream in RandomStream(32).spawn(300):
            generator = stream.generator()
            x_cal, y_cal = _line_data(generator, 60)
            x_test, y_test = _line_data(generator, 20)
            model = nonexchangeable_split_conformal_regression(
                lambda x: 0.5 + 0.8 * x[0], x_cal, y_cal, weights, alpha=0.1
            )
            rates.append(
                sum(
                    model.interval(x).contains(y)
                    for x, y in zip(x_test, y_test, strict=True)
                )
                / 20
            )
        mean, se = _mean_se(rates)
        self.assertGreaterEqual(mean, 0.9 - 3.0 * se)
        self.assertLess(mean, 0.97)

    def test_recent_weights_help_after_a_changepoint(self) -> None:
        """After a noise jump, weighting recent points loses less coverage."""
        weights = [0.9 ** (100 - i) for i in range(1, 101)]
        fixed: list[float] = []
        plain: list[float] = []
        for stream in RandomStream(33).spawn(300):
            generator = stream.generator()
            scales = [1.0] * 70 + [3.0] * 30
            y_cal = [generator.gauss(0.0, s) for s in scales]
            x_cal = [(0.0,)] * 100
            y_test = [generator.gauss(0.0, 3.0) for _ in range(20)]
            recent = nonexchangeable_split_conformal_regression(
                _zero, x_cal, y_cal, weights, alpha=0.1
            ).interval((0.0,))
            uniform = split_conformal_regression(_zero, x_cal, y_cal).interval((0.0,))
            fixed.append(sum(recent.contains(y) for y in y_test) / 20)
            plain.append(sum(uniform.contains(y) for y in y_test) / 20)
        mean_f, se_f = _mean_se(fixed)
        mean_p, se_p = _mean_se(plain)
        self.assertGreater(mean_f, 0.87)
        self.assertLess(mean_p, 0.80)
        self.assertGreater(mean_f - mean_p, 5.0 * math.hypot(se_f, se_p))

    def test_coverage_gap(self) -> None:
        """The gap is the normalized weighted sum of the supplied bounds."""
        self.assertEqual(nonexchangeable_coverage_gap([1.0, 0.5], [0.0, 0.0]), 0.0)
        self.assertAlmostEqual(
            nonexchangeable_coverage_gap([1.0, 0.5, 0.25], [0.1, 0.2, 0.4]),
            (0.1 + 0.1 + 0.1) / 2.75,
        )
        self.assertLess(nonexchangeable_coverage_gap([1.0] * 5, [1.0] * 5), 1.0)
        with self.assertRaises(ValueError):
            nonexchangeable_coverage_gap([1.0], [1.5])
        with self.assertRaises(ValueError):
            nonexchangeable_coverage_gap([1.0, 1.0], [0.5])
        with self.assertRaises(ValueError):
            nonexchangeable_coverage_gap([], [])

    def test_validation(self) -> None:
        """Weights outside ``[0, 1]`` are rejected."""
        with self.assertRaises(ValueError):
            nonexchangeable_split_conformal_regression(
                _zero, [(0.0,)], [1.0], [2.0], alpha=0.1
            )
        with self.assertRaises(ValueError):
            nonexchangeable_split_conformal_regression(
                _zero, [(0.0,)], [1.0], [1.0, 1.0], alpha=0.1
            )


class RiskControlTests(unittest.TestCase):
    """Conformal risk control on a grid."""

    def test_miscoverage_loss_reproduces_split_conformal(self) -> None:
        """Loss ``1{s > lambda}`` on the score grid gives the conformal quantile."""
        generator = RandomStream(41).generator()
        for n in (9, 10, 19, 33, 100):
            scores = [generator.random() for _ in range(n)]
            grid = sorted(scores)
            losses = [[1.0 if s > lam else 0.0 for lam in grid] for s in scores]
            for alpha in (0.05, 0.1, 0.2, 0.3, 0.5):
                result = conformal_risk_control(losses, grid, alpha=alpha)
                expected = conformal_quantile(scores, alpha)
                if math.isinf(expected):
                    self.assertFalse(result.feasible)
                    self.assertEqual(result.lambda_hat, grid[-1])
                else:
                    self.assertTrue(result.feasible)
                    self.assertEqual(result.lambda_hat, expected)
                    self.assertEqual(result.index + 1, conformal_rank(n, alpha))
                    self.assertLessEqual(result.adjusted_risk, alpha + 1e-12)

    def test_by_hand(self) -> None:
        """Risks, adjusted risks, and the selected index for a small table."""
        losses = [[1.0, 0.5, 0.0], [1.0, 1.0, 0.25], [0.5, 0.0, 0.0]]
        result = conformal_risk_control(losses, [0.0, 1.0, 2.0], alpha=0.5)
        self.assertEqual(result.size, 3)
        for got, want in zip(result.risks, (2.5 / 3, 0.5, 0.25 / 3), strict=True):
            self.assertAlmostEqual(got, want)
        for got, want in zip(
            result.adjusted_risks, (3.5 / 4, 2.5 / 4, 1.25 / 4), strict=True
        ):
            self.assertAlmostEqual(got, want)
        self.assertEqual((result.lambda_hat, result.index), (2.0, 2))
        self.assertAlmostEqual(result.adjusted_risk, 1.25 / 4)
        self.assertAlmostEqual(result.empirical_risk, 0.25 / 3)
        self.assertTrue(result.feasible)
        # The boundary case (sum + B)/(n + 1) == alpha exactly is accepted.
        edge = conformal_risk_control([[1.0, 0.0]] * 9, [0.0, 1.0], alpha=0.1)
        self.assertEqual((edge.lambda_hat, edge.feasible), (1.0, True))
        tight = conformal_risk_control([[1.0, 0.0]] * 8, [0.0, 1.0], alpha=0.1)
        self.assertFalse(tight.feasible)
        self.assertEqual(tight.lambda_hat, 1.0)
        # A larger bound B makes the same data infeasible.
        self.assertFalse(
            conformal_risk_control(
                losses, [0.0, 1.0, 2.0], alpha=0.5, bound=2.0
            ).feasible
        )

    @staticmethod
    def _multilabel(generator: random.Random) -> tuple[list[float], list[int]]:
        """Draw class probabilities-as-scores and a non-empty label set."""
        while True:
            probabilities = [generator.random() ** 2 for _ in range(6)]
            labels = [k for k, p in enumerate(probabilities) if generator.random() < p]
            if labels:
                noisy = [
                    min(1.0, max(0.0, p + generator.gauss(0.0, 0.25)))
                    for p in probabilities
                ]
                return noisy, labels

    @staticmethod
    def _false_negative_rate(
        scores: Sequence[float], labels: Sequence[int], lam: float
    ) -> float:
        """Share of true labels missing from ``{k : score_k >= 1 - lambda}``."""
        return sum(scores[k] < 1.0 - lam for k in labels) / len(labels)

    def test_false_negative_rate_is_controlled(self) -> None:
        """The expected false-negative rate at ``lambda_hat`` is at most alpha."""
        grid = [j / 50 for j in range(51)]
        alpha = 0.1
        risks: list[float] = []
        chosen: list[float] = []
        for stream in RandomStream(42).spawn(400):
            generator = stream.generator()
            calibration = [self._multilabel(generator) for _ in range(40)]
            losses = [
                [self._false_negative_rate(s, y, lam) for lam in grid]
                for s, y in calibration
            ]
            result = conformal_risk_control(losses, grid, alpha=alpha)
            self.assertTrue(result.feasible)
            chosen.append(result.lambda_hat)
            test = [self._multilabel(generator) for _ in range(20)]
            risks.append(
                sum(self._false_negative_rate(s, y, result.lambda_hat) for s, y in test)
                / 20
            )
        mean, se = _mean_se(risks)
        self.assertLessEqual(mean, alpha + 3.0 * se)
        # Not vacuous: the risk is close to alpha and lambda_hat is mostly interior.
        self.assertGreater(mean, alpha - 0.05)
        self.assertLess(sum(chosen) / len(chosen), 0.95)

    def test_validation(self) -> None:
        """Non-monotone losses, bound violations, and bad grids are rejected."""
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            conformal_risk_control([[0.2, 0.5]], [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "bound"):
            conformal_risk_control([[1.5, 0.5]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            conformal_risk_control([[0.5, 0.2]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            conformal_risk_control([[0.5, 0.2, 0.1]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            conformal_risk_control([], [0.0, 1.0])
        with self.assertRaises(ValueError):
            conformal_risk_control([[]], [])
        with self.assertRaises(ValueError):
            conformal_risk_control([[math.nan, 0.0]], [0.0, 1.0])


class ConformalPValueTests(unittest.TestCase):
    """Conformal p-values and Benjamini-Hochberg outlier detection."""

    def test_exactly_uniform_by_enumeration(self) -> None:
        """Over all orderings of distinct scores, p is uniform on its grid."""
        for n in (1, 3, 5):
            counts: Counter[Fraction] = Counter()
            for order in itertools.permutations(range(n + 1)):
                (p,) = conformal_p_values(order[:n], order[n:])
                counts[Fraction(p).limit_denominator(n + 1)] += 1
            self.assertEqual(set(counts), {Fraction(k, n + 1) for k in range(1, n + 2)})
            self.assertEqual(set(counts.values()), {math.factorial(n)})

    def test_by_hand_with_ties(self) -> None:
        """Ties count as ``>=``, which keeps the p-value conservative."""
        p = conformal_p_values([1.0, 2.0, 2.0, 3.0], [0.0, 2.0, 2.5, 3.0, 4.0])
        self.assertEqual(p, (1.0, 0.8, 0.4, 0.4, 0.2))

    def test_marginal_super_uniformity(self) -> None:
        """Under the null ``P(p <= u) <= u`` for discrete (tied) scores too."""
        hits = {0.1: 0, 0.25: 0, 0.5: 0}
        total = 0
        for stream in RandomStream(51).spawn(400):
            generator = stream.generator()
            calibration = [generator.randint(0, 9) for _ in range(19)]
            for p in conformal_p_values(
                calibration, [generator.randint(0, 9) for _ in range(10)]
            ):
                total += 1
                for level in hits:
                    hits[level] += p <= level
        for level, count in hits.items():
            rate = count / total
            self.assertLessEqual(
                rate, level + 3.0 * math.sqrt(level * (1 - level) / 400)
            )

    def test_benjamini_hochberg_controls_fdr(self) -> None:
        """FDR over replicates is at most alpha, with non-trivial power."""
        alpha = 0.1
        proportions: list[float] = []
        powers: list[float] = []
        for stream in RandomStream(52).spawn(400):
            generator = stream.generator()
            calibration = [generator.gauss(0.0, 1.0) for _ in range(199)]
            inliers = [generator.gauss(0.0, 1.0) for _ in range(40)]
            outliers = [generator.gauss(3.5, 1.0) for _ in range(10)]
            result = conformal_outlier_detection(
                calibration, inliers + outliers, alpha=alpha
            )
            false = sum(result.rejected[:40])
            true = sum(result.rejected[40:])
            proportions.append(false / max(1, false + true))
            powers.append(true / 10)
            self.assertEqual(result.rejection_count, false + true)
        fdr, se = _mean_se(proportions)
        self.assertLessEqual(fdr, alpha + 3.0 * se)
        # With 80% inliers BH controls FDR at 0.8 * alpha; check it is close.
        self.assertLessEqual(fdr, 0.8 * alpha + 3.0 * se)
        self.assertGreater(fdr, 0.03)
        self.assertGreater(sum(powers) / len(powers), 0.5)

    def test_matches_library_benjamini_hochberg(self) -> None:
        """Decisions equal ``adjust_p_values(method='bh')`` on the p-values."""
        generator = RandomStream(53).generator()
        calibration = [generator.gauss(0.0, 1.0) for _ in range(99)]
        test = [generator.gauss(2.0, 1.5) for _ in range(30)]
        result = conformal_outlier_detection(calibration, test, alpha=0.2)
        reference = adjust_p_values(
            conformal_p_values(calibration, test), method="bh", alpha=0.2
        )
        self.assertEqual(result.rejected, reference.rejected)
        self.assertEqual(result.adjusted_p_values, reference.adjusted_p_values)
        self.assertEqual(result.calibration_size, 99)
        self.assertEqual(result.smallest_attainable_p_value, 0.01)
        self.assertGreater(result.rejection_count, 0)

    def test_validation(self) -> None:
        """Empty score sets and non-finite scores are rejected."""
        with self.assertRaises(ValueError):
            conformal_p_values([], [1.0])
        with self.assertRaises(ValueError):
            conformal_p_values([1.0], [math.inf])
        with self.assertRaises(ValueError):
            conformal_outlier_detection([1.0], [])
        with self.assertRaises(ValueError):
            conformal_outlier_detection([1.0], [2.0], alpha=0.0)


class MondrianTests(unittest.TestCase):
    """Group-conditional split conformal."""

    def test_one_group_equals_split_conformal(self) -> None:
        """A single label gives the split conformal rank, quantile, intervals."""
        generator = RandomStream(61).generator()
        x_cal, y_cal = _line_data(generator, 30)
        plain = split_conformal_regression(_identity, x_cal, y_cal, alpha=0.1)
        grouped = mondrian_conformal_regression(
            _identity, x_cal, y_cal, ["all"] * 30, alpha=0.1
        )
        (group,) = grouped.groups
        self.assertEqual(
            (group.size, group.rank, group.quantile, group.scores),
            (30, plain.rank, plain.quantile, plain.scores),
        )
        self.assertEqual(group.guaranteed_coverage, plain.guaranteed_coverage)
        for x in ((0.4,), (2.0,)):
            a, b = grouped.interval(x, "all"), plain.interval(x)
            self.assertEqual((a.lower, a.upper), (b.lower, b.upper))

    def test_group_conditional_coverage(self) -> None:
        """Per-group quantiles fix the over/under-coverage of marginal split."""
        sigma = {"quiet": 0.5, "noisy": 2.0}
        labels = ["quiet"] * 49 + ["noisy"] * 49
        x_cal = [(0.0,)] * 98
        mondrian: dict[str, list[float]] = {"quiet": [], "noisy": []}
        marginal: dict[str, list[float]] = {"quiet": [], "noisy": []}
        for stream in RandomStream(62).spawn(300):
            generator = stream.generator()
            y_cal = [generator.gauss(0.0, sigma[g]) for g in labels]
            grouped = mondrian_conformal_regression(
                _zero, x_cal, y_cal, labels, alpha=0.1
            )
            plain = split_conformal_regression(_zero, x_cal, y_cal, alpha=0.1)
            for label, sd in sigma.items():
                ys = [generator.gauss(0.0, sd) for _ in range(20)]
                box = grouped.interval((0.0,), label)
                wide = plain.interval((0.0,))
                mondrian[label].append(sum(box.contains(y) for y in ys) / 20)
                marginal[label].append(sum(wide.contains(y) for y in ys) / 20)
        for label in sigma:
            mean, se = _mean_se(mondrian[label])
            # n_g = 49 gives k/(n_g + 1) = 45/50 exactly.
            self.assertLess(abs(mean - 0.9), 3.0 * se)
        quiet, se_q = _mean_se(marginal["quiet"])
        noisy, se_n = _mean_se(marginal["noisy"])
        self.assertGreater(quiet, 0.9 + 8.0 * se_q)
        self.assertLess(noisy, 0.9 - 8.0 * se_n)

    def test_small_and_unseen_groups_are_infinite(self) -> None:
        """Fewer than nine points at alpha = 0.1, or none, give the whole line."""
        labels = [0] * 9 + [1] * 8
        ys = [float(i) for i in range(17)]
        grouped = mondrian_conformal_regression(
            _zero, [(0.0,)] * 17, ys, labels, alpha=0.1
        )
        self.assertEqual(grouped.group_sizes, {0: 9, 1: 8})
        self.assertEqual(grouped.infinite_groups, (1,))
        self.assertEqual(grouped.quantile(0), 8.0)
        self.assertTrue(grouped.groups[0].finite)
        self.assertEqual(grouped.groups[1].rank, 9)
        self.assertEqual(grouped.groups[1].guaranteed_coverage, 1.0)
        for label in (1, "never seen"):
            box = grouped.interval((0.0,), label)
            self.assertEqual((box.lower, box.upper), (-math.inf, math.inf))
        boxes = grouped.intervals([(0.0,), (0.0,)], [0, 1])
        self.assertEqual(boxes[0].upper, 8.0)
        self.assertTrue(math.isinf(boxes[1].upper))

    def test_normalized_scores_and_validation(self) -> None:
        """A scale divides scores within groups; mismatched labels raise."""
        grouped = mondrian_conformal_regression(
            _zero,
            [(1.0,), (2.0,), (4.0,)],
            [1.0, -4.0, 2.0],
            ["a", "a", "b"],
            alpha=0.5,
            scale=lambda x: x[0],
        )
        self.assertEqual(grouped.groups[0].scores, (1.0, 2.0))
        self.assertEqual(grouped.quantile("a"), 2.0)
        box = grouped.interval((3.0,), "a")
        self.assertEqual((box.lower, box.upper), (-6.0, 6.0))
        with self.assertRaises(ValueError):
            mondrian_conformal_regression(_zero, [(0.0,)], [1.0], [0, 1])
        with self.assertRaises(ValueError):
            grouped.intervals([(0.0,)], ["a", "b"])
        with self.assertRaises(ValueError):
            mondrian_conformal_regression(_zero, [(0.0,)], [1.0], [0], alpha=1.5)


class CVPlusTests(unittest.TestCase):
    """CV+ against jackknife+ and a by-hand recomputation."""

    def test_n_folds_reproduce_jackknife_plus(self) -> None:
        """``folds = n`` gives the jackknife+ intervals exactly."""
        generator = RandomStream(71).generator()
        x_train, y_train = _line_data(generator, 23)
        for alpha in (0.1, 0.2):
            jackknife = jackknife_plus_regression(_ols, x_train, y_train, alpha=alpha)
            cv = cv_plus_regression(_ols, x_train, y_train, folds=23, alpha=alpha)
            self.assertEqual(cv.residuals, jackknife.residuals)
            self.assertEqual((cv.folds, cv.model_fits, cv.size), (23, 23, 23))
            for x in ((-0.5,), (0.7,), (1.6,), (4.0,)):
                a, b = cv.interval(x), jackknife.interval(x)
                self.assertEqual(
                    (a.lower, a.upper, a.prediction), (b.lower, b.upper, b.prediction)
                )

    def test_by_hand_five_folds(self) -> None:
        """Limits are order statistics of out-of-fold predictions -/+ residuals."""
        generator = RandomStream(72).generator()
        x_train, y_train = _line_data(generator, 22)
        cv = cv_plus_regression(_ols, x_train, y_train, folds=5, alpha=0.1)
        self.assertEqual(cv.fold_of, tuple(i * 5 // 22 for i in range(22)))
        self.assertEqual(sorted(Counter(cv.fold_of).values()), [4, 4, 4, 5, 5])
        x = (1.3,)
        lows: list[float] = []
        highs: list[float] = []
        for i in range(22):
            keep = [j for j in range(22) if cv.fold_of[j] != cv.fold_of[i]]
            line = _ols(
                tuple(x_train[j] for j in keep), tuple(y_train[j] for j in keep)
            )
            residual = abs(y_train[i] - line(x_train[i]))
            lows.append(line(x) - residual)
            highs.append(line(x) + residual)
        box = cv.interval(x)
        # floor(0.1 * 23) = 2 and ceil(0.9 * 23) = 21.
        self.assertAlmostEqual(box.lower, sorted(lows)[1], places=12)
        self.assertAlmostEqual(box.upper, sorted(highs)[20], places=12)
        self.assertEqual(len(cv.intervals([x, (0.1,)])), 2)

    def test_coverage(self) -> None:
        """Five-fold CV+ covers well above the ``1 - 2 alpha`` level."""
        rates: list[float] = []
        for stream in RandomStream(73).spawn(200):
            generator = stream.generator()
            x_train, y_train = _line_data(generator, 20)
            x_test, y_test = _line_data(generator, 20)
            cv = cv_plus_regression(_ols, x_train, y_train, folds=5, alpha=0.1)
            rates.append(
                sum(
                    cv.interval(x).contains(y)
                    for x, y in zip(x_test, y_test, strict=True)
                )
                / 20
            )
        mean, se = _mean_se(rates)
        self.assertGreater(mean - 3.0 * se, 0.8)
        self.assertGreater(mean, 0.87)
        self.assertLess(mean, 0.97)

    def test_shuffle_is_reproducible(self) -> None:
        """A seed shuffles balanced folds reproducibly."""
        generator = RandomStream(74).generator()
        x_train, y_train = _line_data(generator, 20)
        first = cv_plus_regression(_ols, x_train, y_train, folds=4, seed=3)
        again = cv_plus_regression(_ols, x_train, y_train, folds=4, seed=3)
        other = cv_plus_regression(_ols, x_train, y_train, folds=4, seed=4)
        plain = cv_plus_regression(_ols, x_train, y_train, folds=4)
        self.assertEqual(first.fold_of, again.fold_of)
        self.assertEqual(first.residuals, again.residuals)
        self.assertNotEqual(first.fold_of, other.fold_of)
        self.assertNotEqual(first.fold_of, plain.fold_of)
        self.assertEqual(sorted(Counter(first.fold_of).values()), [5, 5, 5, 5])

    def test_validation(self) -> None:
        """The fold count must lie in ``2..n`` and ``fit`` must be callable."""
        xs, ys = [(0.0,), (1.0,), (2.0,)], [0.0, 1.0, 2.5]
        for folds in (0, 1, 4):
            with self.assertRaises(ValueError):
                cv_plus_regression(_ols, xs, ys, folds=folds)
        with self.assertRaises(TypeError):
            cv_plus_regression(_ols, xs, ys, folds=2.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cv_plus_regression(None, xs, ys, folds=2)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cv_plus_regression(_ols, xs, ys, folds=2, alpha=0.0)


class OnlineQuantileTests(unittest.TestCase):
    """Quantile tracking with fixed and decaying step sizes."""

    @staticmethod
    def _sequences(seed: int, length: int) -> dict[str, list[float]]:
        """Bounded score sequences in ``[0, 1]``: random, drifting, adversarial."""
        generator = RandomStream(seed).generator()
        return {
            "iid": [generator.random() for _ in range(length)],
            "drift": [
                min(1.0, generator.random() * (0.2 + 0.8 * t / length))
                for t in range(length)
            ],
            "alternating": [float(t % 2) for t in range(length)],
            "all-large": [1.0] * length,
            "all-zero": [0.0] * length,
        }

    def test_fixed_step_identity_and_bound(self) -> None:
        """Telescoping: mean(err) - alpha = (q_{T+1} - q_1)/(eta T) <= bound."""
        for name, scores in self._sequences(81, 600).items():
            for alpha, eta, start in (
                (0.1, 0.05, 0.0),
                (0.3, 0.5, 1.0),
                (0.5, 2.0, 0.4),
            ):
                tracker = OnlineQuantileTracker(
                    alpha, eta, initial_quantile=start, score_bound=1.0
                )
                for t, score in enumerate(scores, start=1):
                    tracker.update_score(score)
                    rate = sum(tracker.errors) / t
                    self.assertLessEqual(
                        abs(rate - alpha), tracker.long_run_bound() + 1e-12, name
                    )
                    self.assertAlmostEqual(
                        tracker.long_run_bound(), (1.0 + eta) / (eta * t)
                    )
                    self.assertGreaterEqual(tracker.quantile, -alpha * eta - 1e-12)
                    self.assertLessEqual(
                        tracker.quantile, 1.0 + (1 - alpha) * eta + 1e-12
                    )
                rate = sum(tracker.errors) / len(scores)
                self.assertAlmostEqual(
                    rate - alpha, (tracker.quantile - start) / (eta * len(scores))
                )
                self.assertFalse(tracker.bound_violated)

    def test_decaying_step_bound(self) -> None:
        """For non-increasing steps the bound is ``(b + eta_1)/(eta_T T)``."""
        for name, scores in self._sequences(82, 800).items():
            for decay in (0.5, 0.6, 0.75, 1.0):
                tracker = OnlineQuantileTracker(
                    0.2, 0.4, decay=decay, initial_quantile=0.5, score_bound=1.0
                )
                for t, score in enumerate(scores, start=1):
                    tracker.update_score(score)
                    eta_t = 0.4 * t ** (-decay)
                    self.assertAlmostEqual(tracker.step_size(t), eta_t)
                    self.assertAlmostEqual(
                        tracker.long_run_bound(), 1.4 / (eta_t * t), places=10
                    )
                    rate = sum(tracker.errors) / t
                    self.assertLessEqual(
                        abs(rate - 0.2), tracker.long_run_bound() + 1e-12, name
                    )

    def test_decaying_steps_settle_on_the_quantile(self) -> None:
        """On i.i.d. U(0, 1) scores q_t settles near 0.9; fixed steps keep moving."""
        scores = RandomStream(83).generator()
        values = [scores.random() for _ in range(20000)]
        zeros = [0.0] * len(values)
        fixed = online_quantile_intervals(
            zeros, values, alpha=0.1, step=0.5, score_bound=1.0, initial_quantile=0.5
        )
        decaying = online_quantile_intervals(
            zeros,
            values,
            alpha=0.1,
            step=0.5,
            decay=0.6,
            score_bound=1.0,
            initial_quantile=0.5,
        )
        self.assertAlmostEqual(fixed.long_run_bound, 1.5 / (0.5 * 20000))
        self.assertAlmostEqual(
            decaying.long_run_bound, 1.5 / (0.5 * 20000**-0.6 * 20000)
        )
        for result in (fixed, decaying):
            self.assertTrue(result.bound_valid)
            self.assertLessEqual(
                abs(result.miscoverage_rate - 0.1), result.long_run_bound
            )
        self.assertLess(abs(fixed.miscoverage_rate - 0.1), 1e-3)
        self.assertLess(abs(decaying.miscoverage_rate - 0.1), 0.01)
        tail_fixed = fixed.quantiles[10000:]
        tail_decaying = decaying.quantiles[10000:]
        self.assertLess(max(abs(q - 0.9) for q in tail_decaying), 0.03)
        self.assertGreater(max(tail_fixed) - min(tail_fixed), 0.4)
        self.assertLess(max(tail_decaying) - min(tail_decaying), 0.06)

    def test_intervals_and_empty_set(self) -> None:
        """The interval is ``yhat +/- q s``; a negative threshold is empty."""
        tracker = OnlineQuantileTracker(0.25, 1.0, initial_quantile=2.0)
        box = tracker.interval(10.0, scale=3.0)
        self.assertEqual((box.lower, box.upper, box.prediction), (4.0, 16.0, 10.0))
        self.assertFalse(tracker.update(10.0, 13.0, scale=3.0))  # score 1 <= 2
        self.assertAlmostEqual(tracker.quantile, 1.75)
        self.assertTrue(tracker.update(10.0, 0.0))  # score 10 > 1.75
        self.assertAlmostEqual(tracker.quantile, 2.5)
        self.assertEqual(tracker.errors, (False, True))
        self.assertEqual(tracker.alpha, 0.25)
        self.assertEqual(tracker.long_run_bound(), math.inf)
        cold = OnlineQuantileTracker(0.5, 1.0)
        self.assertFalse(cold.update_score(0.0))
        self.assertEqual(cold.quantile, -0.5)
        empty = cold.interval(0.0)
        self.assertLess(empty.upper, empty.lower)
        self.assertEqual(empty.width, 0.0)
        self.assertTrue(cold.update_score(0.0))

    def test_bound_violation_is_flagged(self) -> None:
        """A score above the declared bound voids the bound instead of hiding it."""
        result = online_quantile_intervals(
            [0.0, 0.0, 0.0], [0.5, 3.0, 0.1], step=0.1, score_bound=1.0
        )
        self.assertFalse(result.bound_valid)
        self.assertEqual(result.long_run_bound, math.inf)
        self.assertEqual(result.errors, (True, True, False))
        self.assertAlmostEqual(result.final_quantile, 0.09 + 0.09 - 0.01)
        self.assertEqual(len(result.intervals), 3)
        self.assertEqual((result.step, result.decay, result.alpha), (0.1, 0.0, 0.1))
        unbounded = online_quantile_intervals([0.0], [0.5], scales=[2.0])
        self.assertFalse(unbounded.bound_valid)

    def test_validation(self) -> None:
        """Invalid levels, steps, exponents, bounds, and lengths are rejected."""
        for kwargs in (
            {"alpha": 0.0},
            {"step": 0.0},
            {"decay": -0.1},
            {"decay": 1.5},
            {"initial_quantile": -1.0},
            {"score_bound": 0.0},
            {"initial_quantile": 2.0, "score_bound": 1.0},
        ):
            with self.assertRaises(ValueError):
                OnlineQuantileTracker(**kwargs)  # type: ignore[arg-type]
        tracker = OnlineQuantileTracker()
        with self.assertRaises(ValueError):
            tracker.update_score(-1.0)
        with self.assertRaises(ValueError):
            tracker.step_size(0)
        with self.assertRaises(ValueError):
            tracker.interval(0.0, scale=0.0)
        with self.assertRaises(ValueError):
            online_quantile_intervals([], [])
        with self.assertRaises(ValueError):
            online_quantile_intervals([0.0], [0.0, 1.0])
        with self.assertRaises(ValueError):
            online_quantile_intervals([0.0], [0.0], scales=[1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
