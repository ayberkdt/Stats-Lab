"""Tests for distribution-free conformal prediction in ``probstat.conformal``.

The finite-sample index ``k = ceil((1 - alpha)(n + 1))`` is checked exactly,
including cases where binary floating point would round the wrong way. The
coverage checks average over seeded calibration/test replicates: with
continuous scores, split conformal coverage equals ``k / (n + 1)`` in
expectation, so the replicate mean must sit within a few Monte Carlo standard
errors of it (and therefore not below ``1 - alpha``).
"""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Sequence

from probstat.conformal import (
    PredictionInterval,
    conformal_quantile,
    conformal_rank,
    conformalize_surrogate,
    conformalized_quantile_regression,
    empirical_coverage,
    jackknife_plus_regression,
    lower_conformal_rank,
    split_conformal_coverage,
    split_conformal_regression,
)
from probstat.distributions import Normal
from probstat.foundations import RandomStream
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.surrogates import fit_gaussian_process_to_model, fit_polynomial_chaos
from probstat.uq import QuantityOfInterest


def _noise_scale(x: float) -> float:
    """Heteroscedastic noise standard deviation."""
    return 0.2 + 0.2 * x


def _draw(
    generator: random.Random, size: int
) -> tuple[list[tuple[float]], list[float]]:
    """Draw ``(x, y)`` with ``x ~ U(0, 3)`` and a nonlinear noisy response."""
    xs: list[tuple[float]] = []
    ys: list[float] = []
    for _ in range(size):
        x = 3.0 * generator.random()
        xs.append((x,))
        ys.append(math.sin(2.0 * x) + 0.3 * x + generator.gauss(0.0, _noise_scale(x)))
    return xs, ys


def _misspecified(x: Sequence[float]) -> float:
    """Return a deliberately wrong linear prediction (no training, no reuse)."""
    return 0.1 + 0.3 * x[0]


def _ols(points: tuple[tuple[float, ...], ...], values: tuple[float, ...]):  # type: ignore[no-untyped-def]
    """Return the least-squares line through one-dimensional training data."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(values) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (y - my) for p, y in zip(points, values, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    return lambda x: intercept + slope * x[0]


class ConformalQuantileTests(unittest.TestCase):
    """Exact order-statistic indices and the quantile itself."""

    def test_rank_is_exact(self) -> None:
        """``k = ceil((1 - alpha)(n + 1))`` in exact decimal arithmetic."""
        cases = {
            (9, 0.1): 9,
            (19, 0.1): 18,
            (10, 0.1): 10,
            (99, 0.05): 95,
            (9, 0.3): 7,
            (999, 0.1): 900,
            (5, 0.1): 6,
            (1, 0.5): 1,
        }
        for (n, alpha), k in cases.items():
            self.assertEqual(conformal_rank(n, alpha), k, (n, alpha))
        self.assertEqual(lower_conformal_rank(19, 0.1), 2)
        self.assertEqual(lower_conformal_rank(8, 0.1), 0)
        self.assertEqual(lower_conformal_rank(99, 0.05), 5)

    def test_quantile_is_kth_smallest(self) -> None:
        """``q_hat`` is the ``k``-th order statistic, ``inf`` when ``k > n``."""
        scores = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]
        self.assertEqual(conformal_quantile(scores, 0.1), 0.9)  # k = 9
        self.assertEqual(conformal_quantile(scores, 0.2), 0.8)  # k = 8
        self.assertEqual(conformal_quantile(scores, 0.5), 0.5)  # k = 5
        self.assertTrue(math.isinf(conformal_quantile(scores[:5], 0.1)))  # k = 6

    def test_guaranteed_coverage(self) -> None:
        """``k / (n + 1)`` lies in ``[1 - alpha, 1 - alpha + 1/(n + 1))``."""
        for n in (1, 4, 9, 19, 50, 199):
            for alpha in (0.05, 0.1, 0.2, 0.5):
                value = split_conformal_coverage(n, alpha)
                self.assertGreaterEqual(value, 1.0 - alpha - 1e-12)
                if conformal_rank(n, alpha) <= n:
                    self.assertLess(value, 1.0 - alpha + 1.0 / (n + 1) + 1e-12)
        self.assertEqual(split_conformal_coverage(19, 0.1), 0.9)
        self.assertEqual(split_conformal_coverage(4, 0.1), 1.0)

    def test_validation(self) -> None:
        """Invalid levels, sizes, and empty score sets are rejected."""
        for alpha in (0.0, 1.0, -0.1, math.nan):
            with self.assertRaises(ValueError):
                conformal_rank(10, alpha)
        with self.assertRaises(ValueError):
            conformal_rank(0, 0.1)
        with self.assertRaises(TypeError):
            conformal_rank(2.5, 0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            conformal_quantile([], 0.1)


class SplitConformalTests(unittest.TestCase):
    """Marginal coverage of split conformal, normalized scores, and CQR."""

    REPLICATES = 500
    CALIBRATION = 10
    TARGET = 10 / 11  # k = ceil(0.9 * 11) = 10; the naive ceil(0.9 * 10) = 9
    TEST = 40
    ALPHA = 0.1

    def _replicate_coverage(self, build, seed: int) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        """Return mean coverage and its Monte Carlo standard error."""
        rates: list[float] = []
        for stream in RandomStream(seed).spawn(self.REPLICATES):
            generator = stream.generator()
            x_cal, y_cal = _draw(generator, self.CALIBRATION)
            x_test, y_test = _draw(generator, self.TEST)
            regressor = build(x_cal, y_cal)
            report = empirical_coverage(regressor.intervals(x_test), y_test)
            rates.append(report.coverage)
        mean = sum(rates) / len(rates)
        spread = math.sqrt(sum((r - mean) ** 2 for r in rates) / (len(rates) - 1))
        return mean, spread / math.sqrt(len(rates))

    def test_absolute_residual_coverage(self) -> None:
        """Mean coverage is ``k/(n+1) = 10/11`` within 3.5 standard errors."""
        target = split_conformal_coverage(self.CALIBRATION, self.ALPHA)
        self.assertEqual(target, self.TARGET)
        mean, error = self._replicate_coverage(
            lambda x, y: split_conformal_regression(
                _misspecified, x, y, alpha=self.ALPHA
            ),
            seed=101,
        )
        self.assertLess(abs(mean - target), 3.5 * error)
        self.assertGreater(mean, 1.0 - self.ALPHA - 3.5 * error)
        self.assertLess(error, 0.006)
        # The naive index ceil((1 - alpha) n) would give 9/11 = 0.818.
        self.assertGreater(mean - 9 / 11, 10 * error)

    def test_normalized_score_coverage_and_adaptivity(self) -> None:
        """Locally weighted scores keep coverage and widen where noise is larger."""
        mean, error = self._replicate_coverage(
            lambda x, y: split_conformal_regression(
                _misspecified,
                x,
                y,
                alpha=self.ALPHA,
                scale=lambda v: _noise_scale(v[0]),
            ),
            seed=202,
        )
        self.assertLess(abs(mean - self.TARGET), 3.5 * error)
        generator = RandomStream(3).generator()
        x_cal, y_cal = _draw(generator, 99)
        regressor = split_conformal_regression(
            _misspecified, x_cal, y_cal, alpha=0.1, scale=lambda v: _noise_scale(v[0])
        )
        self.assertEqual(regressor.score, "normalized")
        self.assertEqual(regressor.rank, 90)
        self.assertGreater(
            regressor.interval((2.8,)).width, regressor.interval((0.1,)).width
        )
        with self.assertRaises(ValueError):
            split_conformal_regression(
                _misspecified, x_cal, y_cal, scale=lambda v: v[0] - 1.0
            )

    def test_conformalized_quantile_regression(self) -> None:
        """CQR repairs badly calibrated quantile predictors to ``k/(n+1)``."""

        def low(x: Sequence[float]) -> float:
            """Too-narrow lower band."""
            return math.sin(2.0 * x[0]) + 0.3 * x[0] - 0.5 * _noise_scale(x[0])

        def high(x: Sequence[float]) -> float:
            """Too-narrow upper band."""
            return math.sin(2.0 * x[0]) + 0.3 * x[0] + 0.5 * _noise_scale(x[0])

        mean, error = self._replicate_coverage(
            lambda x, y: conformalized_quantile_regression(
                low, high, x, y, alpha=self.ALPHA
            ),
            seed=303,
        )
        self.assertLess(abs(mean - self.TARGET), 3.5 * error)
        generator = RandomStream(4).generator()
        x_cal, y_cal = _draw(generator, 99)
        cqr = conformalized_quantile_regression(low, high, x_cal, y_cal, alpha=0.1)
        self.assertGreater(cqr.quantile, 0.0)
        box = cqr.interval((1.0,))
        self.assertAlmostEqual(box.lower, low((1.0,)) - cqr.quantile)
        self.assertAlmostEqual(box.upper, high((1.0,)) + cqr.quantile)
        scores = sorted(
            max(low(x) - y, y - high(x)) for x, y in zip(x_cal, y_cal, strict=True)
        )
        self.assertEqual(cqr.quantile, scores[89])

    def test_interval_arithmetic_and_errors(self) -> None:
        """Intervals are ``mu +/- q_hat``; mismatched data are rejected."""
        x_cal = [(0.0,), (1.0,), (2.0,), (3.0,)]
        y_cal = [0.3, 0.0, 1.5, 0.2]
        regressor = split_conformal_regression(_misspecified, x_cal, y_cal, alpha=0.2)
        residuals = sorted(
            abs(y - _misspecified(x)) for x, y in zip(x_cal, y_cal, strict=True)
        )
        self.assertEqual(regressor.rank, 4)
        self.assertEqual(regressor.quantile, residuals[3])
        box = regressor.interval((1.5,))
        self.assertAlmostEqual(box.prediction, 0.55)
        self.assertAlmostEqual(box.upper - box.lower, 2 * residuals[3])
        self.assertTrue(
            math.isinf(
                split_conformal_regression(
                    _misspecified, x_cal, y_cal, alpha=0.1
                ).quantile
            )
        )
        with self.assertRaises(ValueError):
            split_conformal_regression(_misspecified, x_cal, y_cal[:3])
        with self.assertRaises(TypeError):
            split_conformal_regression(42, x_cal, y_cal)  # type: ignore[arg-type]


class JackknifePlusTests(unittest.TestCase):
    """Jackknife+ intervals from leave-one-out refits."""

    def test_coverage_sanity(self) -> None:
        """Coverage exceeds ``1 - 2 alpha`` and is near ``1 - alpha`` on average."""
        rates: list[float] = []
        for stream in RandomStream(404).spawn(150):
            generator = stream.generator()
            x_train, y_train = _draw(generator, 20)
            x_test, y_test = _draw(generator, 20)
            model = jackknife_plus_regression(_ols, x_train, y_train, alpha=0.1)
            rates.append(empirical_coverage(model.intervals(x_test), y_test).coverage)
        mean = sum(rates) / len(rates)
        self.assertGreater(mean, 0.85)
        self.assertLess(mean, 0.97)

    def test_order_statistics_by_hand(self) -> None:
        """Limits are the floor/ceil order statistics of ``mu_{-i}(x) -/+ R_i``."""
        generator = RandomStream(5).generator()
        x_train, y_train = _draw(generator, 19)
        model = jackknife_plus_regression(_ols, x_train, y_train, alpha=0.1)
        x = (1.7,)
        lows: list[float] = []
        highs: list[float] = []
        for i in range(19):
            points = tuple(tuple(p) for p in x_train[:i] + x_train[i + 1 :])
            values = tuple(y_train[:i] + y_train[i + 1 :])
            line = _ols(points, values)
            residual = abs(y_train[i] - line(x_train[i]))
            lows.append(line(x) - residual)
            highs.append(line(x) + residual)
        box = model.interval(x)
        self.assertAlmostEqual(box.lower, sorted(lows)[1], places=12)  # floor(2.0)
        self.assertAlmostEqual(box.upper, sorted(highs)[17], places=12)  # ceil(18.0)
        with self.assertRaises(ValueError):
            jackknife_plus_regression(_ols, x_train[:1], y_train[:1])


class SurrogateConformalTests(unittest.TestCase):
    """Conformal error bars for PCE and GP emulators of a probabilistic model."""

    @staticmethod
    def _model(*, smooth: bool = False) -> ProbabilisticModel:
        """Return a quadratic (or, with ``smooth``, a sine-product) model."""
        inputs = JointInput.independent(
            RandomInput("x1", Normal(0.0, 1.0)), RandomInput("x2", Normal(0.0, 1.0))
        )

        def forward(p):  # type: ignore[no-untyped-def]
            """Response that a linear PCE cannot represent."""
            if smooth:
                return {"y": math.sin(p["x1"]) + 0.5 * math.sin(p["x2"]) * p["x1"]}
            return {"y": p["x1"] + 0.5 * p["x2"] + 0.3 * p["x1"] ** 2}

        return ProbabilisticModel(
            inputs, forward, (QuantityOfInterest("y", "response"),), name="quad"
        )

    def _test_coverage(self, result, model: ProbabilisticModel) -> float:  # type: ignore[no-untyped-def]
        """Return the coverage on 2000 fresh Monte Carlo inputs."""
        points = model.inputs.sample(2000, seed=RandomStream(99))
        truth = model.evaluate_many(points)["y"]
        intervals = [result.interval(x) for x in points]
        return empirical_coverage(intervals, truth, nominal=0.9).coverage

    def test_pce_emulator_error_bars(self) -> None:
        """A degree-1 PCE gets intervals that cover the true QoI about 90%."""
        model = self._model()
        pce = fit_polynomial_chaos(model, degree=1, size=30, seed=1)
        before = model.evaluations
        result = conformalize_surrogate(model, pce, alpha=0.1, size=99, seed=1)
        self.assertEqual(model.evaluations - before, 99)
        self.assertEqual(result.model_evaluations, 99)
        self.assertEqual(result.guaranteed_coverage, 0.9)
        self.assertGreater(result.half_width, 0.0)
        coverage = self._test_coverage(result, model)
        self.assertTrue(0.8 < coverage < 0.98, coverage)

    def test_gp_normalized_error_bars(self) -> None:
        """GP-standard-deviation scores keep coverage with input-dependent widths."""
        model = self._model(smooth=True)
        gp = fit_gaussian_process_to_model(model, size=20, seed=2, restarts=2)
        plain = conformalize_surrogate(model, gp, alpha=0.1, size=99, seed=2)
        result = conformalize_surrogate(
            model, gp, alpha=0.1, size=99, normalized=True, seed=2
        )
        self.assertEqual(result.regressor.score, "normalized")
        near = result.interval((0.0, 0.0)).width
        far = result.interval((3.0, -3.0)).width
        self.assertGreater(far, 10.0 * near)
        self.assertAlmostEqual(
            plain.interval((0.0, 0.0)).width, plain.interval((3.0, -3.0)).width
        )
        coverage = self._test_coverage(result, model)
        self.assertTrue(0.85 < coverage < 0.98, coverage)
        with self.assertRaises(TypeError):
            conformalize_surrogate(
                model,
                fit_polynomial_chaos(model, degree=1, size=30, seed=1),
                normalized=True,
                size=10,
            )
        with self.assertRaises(ValueError):
            conformalize_surrogate(model, gp, normalized=True, scale_offset=-1.0)

    def test_vanishing_gp_variance_needs_an_offset(self) -> None:
        """An interpolating GP with zero predictive variance asks for an offset."""
        model = self._model()
        gp = fit_gaussian_process_to_model(model, size=12, seed=2, restarts=1)
        with self.assertRaises(ValueError):
            conformalize_surrogate(model, gp, size=99, normalized=True, seed=2)
        result = conformalize_surrogate(
            model, gp, size=99, normalized=True, scale_offset=1e-3, seed=2
        )
        self.assertTrue(math.isfinite(result.half_width))


class CoverageReportTests(unittest.TestCase):
    """The empirical coverage utility."""

    def test_report_fields(self) -> None:
        """Counts, Clopper-Pearson interval, widths, and nominal consistency."""
        boxes = [PredictionInterval(-1.0, 1.0, 0.0)] * 10
        values = [0.0] * 9 + [2.0]
        report = empirical_coverage(boxes, values, nominal=0.9)
        self.assertEqual((report.covered, report.size), (9, 10))
        self.assertEqual(report.coverage, 0.9)
        self.assertEqual(report.mean_width, 2.0)
        self.assertTrue(report.consistent_with_nominal)
        self.assertLess(report.coverage_interval.lower, 0.9)
        empty = PredictionInterval(1.0, -1.0, 0.0)
        self.assertEqual(empty.width, 0.0)
        self.assertFalse(empty.contains(0.0))
        with self.assertRaises(ValueError):
            empirical_coverage(boxes, values[:3])
        with self.assertRaises(ValueError):
            empirical_coverage(boxes, values, nominal=1.5)


if __name__ == "__main__":
    unittest.main()
