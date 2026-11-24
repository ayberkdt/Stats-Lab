"""Tests for recursive co-kriging, stochastic kriging, and Bayesian quadrature.

Oracles are independent of the code under test: the recursive co-kriging
predictor is compared with the joint Gaussian conditional built from the
explicit block AR(1) covariance; stochastic kriging with dense linear algebra
on ``S_M + S_e``, with the library GP in the equal-variance case, and with
the explicit weights of the best linear unbiased predictor; the kernel
integrals of Bayesian quadrature with the library's adaptive quadrature, and
its estimates with Gauss-Hermite rules and closed-form integrals.
"""

from __future__ import annotations

import math
import random
import unittest
import warnings

from probstat.linalg import solve_spd
from probstat.modeling import gauss_hermite
from probstat.numerics import integrate
from probstat.surrogates import (
    BayesianQuadratureResult,
    GaussianMeasure,
    GaussianProcessConditioningWarning,
    ProbabilisticSurrogate,
    RecursiveCoKriging,
    StochasticKriging,
    UniformMeasure,
    bayesian_quadrature,
    fit_gaussian_process,
    fit_recursive_cokriging,
    fit_stochastic_kriging,
    gaussian_process_from_hyperparameters,
    kernel_double_integral,
    kernel_mean,
    kernel_value,
    stochastic_kriging_from_summaries,
)


def expensive(x: float) -> float:
    """Return the high-fidelity test response on ``[0, 1]``."""
    return (6.0 * x - 2.0) ** 2 * math.sin(12.0 * x - 4.0)


def cheap(x: float) -> float:
    """Return the low-fidelity response: a scaled, tilted, shifted copy."""
    return 0.5 * expensive(x) + 10.0 * (x - 0.5) - 5.0


def dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return the inner product of two equally long vectors."""
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


class JointCoKrigingOracleTests(unittest.TestCase):
    """The recursive predictor equals the joint GP with block AR(1) covariance."""

    kernel = "matern52"
    # Per level: (signal variance, length scales, noise variance, mean).
    parts = (
        (2.0, (0.6, 0.9), 1e-6, 0.4),
        (0.5, (0.35, 0.5), 1e-6, -0.3),
        (0.2, (0.5, 0.3), 1e-6, 0.1),
    )
    rho = (1.7, -0.8)

    def setUp(self) -> None:
        """Draw nested designs and arbitrary responses for three levels."""
        generator = random.Random(20261121)
        self.designs = [
            tuple((generator.random(), generator.random()) for _ in range(9))
        ]
        self.designs.append(self.designs[0][:6])
        self.designs.append(self.designs[1][1:5])
        self.responses = [
            tuple(generator.gauss(0.0, 1.5) for _ in design) for design in self.designs
        ]

    def recursive(self) -> RecursiveCoKriging:
        """Assemble the recursive model from fixed-hyperparameter GPs."""
        levels = []
        for index, (signal, scales, noise, mean) in enumerate(self.parts):
            values = self.responses[index]
            if index > 0:
                lookup = dict(
                    zip(self.designs[index - 1], self.responses[index - 1], strict=True)
                )
                values = tuple(
                    value - self.rho[index - 1] * lookup[point]
                    for point, value in zip(self.designs[index], values, strict=True)
                )
            levels.append(
                gaussian_process_from_hyperparameters(
                    self.designs[index],
                    values,
                    kernel=self.kernel,
                    signal_variance=signal,
                    length_scales=scales,
                    noise_variance=noise,
                    mean=mean,
                )
            )
        return RecursiveCoKriging(tuple(levels), self.rho)

    def coefficient(self, level: int, part: int) -> float:
        """Return the weight of independent process ``part`` in ``level``."""
        if part > level:
            return 0.0
        return math.prod(self.rho[part:level])

    def joint(self, x: tuple[float, float], top: int) -> tuple[float, float]:
        """Condition the joint Gaussian of all observations on the data.

        Level ``t`` is ``sum_j c_tj U_j`` with independent processes ``U_j``
        of covariance ``k_j`` (plus the noise variance at coinciding design
        points), which is the block AR(1) covariance written out.
        """
        labels = [
            (level, point) for level in range(top + 1) for point in self.designs[level]
        ]
        data = [value for level in range(top + 1) for value in self.responses[level]]

        def level_mean(level: int) -> float:
            """Return the prior mean of one level."""
            return math.fsum(
                self.coefficient(level, part) * self.parts[part][3]
                for part in range(level + 1)
            )

        def covariance(
            a: tuple[int, tuple[float, float]],
            b: tuple[int, tuple[float, float]],
            *,
            noisy: bool,
        ) -> float:
            """Return the covariance of two (level, point) observations."""
            total = 0.0
            for part, (signal, scales, noise, _) in enumerate(self.parts):
                weight = self.coefficient(a[0], part) * self.coefficient(b[0], part)
                if weight == 0.0:
                    continue
                value = kernel_value(self.kernel, a[1], b[1], scales, signal)
                if noisy and a[1] == b[1]:
                    value += noise
                total += weight * value
            return total

        matrix = tuple(
            tuple(covariance(a, b, noisy=True) for b in labels) for a in labels
        )
        target = (top, x)
        cross = tuple(covariance(target, b, noisy=False) for b in labels)
        residual = tuple(
            value - level_mean(label[0])
            for value, label in zip(data, labels, strict=True)
        )
        mean = level_mean(top) + dot(cross, solve_spd(matrix, residual))
        variance = covariance(target, target, noisy=False) - dot(
            cross, solve_spd(matrix, cross)
        )
        return mean, variance

    def test_recursive_predictor_equals_joint_gaussian_conditional(self) -> None:
        """Mean and variance of every level agree with the joint GP to 1e-8."""
        model = self.recursive()
        worst = 0.0
        for x in ((0.21, 0.67), (0.5, 0.5), (0.93, 0.08), (1.3, -0.2)):
            predictions = model.level_predictions(x)
            for top in range(3):
                mean, variance = self.joint(x, top)
                worst = max(
                    worst,
                    abs(predictions[top][0] - mean),
                    abs(predictions[top][1] - variance),
                )
                self.assertAlmostEqual(predictions[top][0], mean, delta=1e-8)
                self.assertAlmostEqual(predictions[top][1], variance, delta=1e-8)
        self.assertLess(worst, 1e-8)

    def test_level_selection_and_protocol(self) -> None:
        """``level`` selects a fidelity; the default is the highest."""
        model = self.recursive()
        x = (0.3, 0.4)
        self.assertIsInstance(model, ProbabilisticSurrogate)
        self.assertEqual(model.level_count, 3)
        self.assertEqual(model.design_sizes, (9, 6, 4))
        self.assertEqual(
            model.predict_with_variance(x), model.predict_with_variance(x, level=3)
        )
        self.assertEqual(
            model.predict_with_variance(x, level=1),
            model.levels[0].predict_with_variance(x),
        )
        self.assertEqual(model.predict(x), model.predict_with_variance(x)[0])
        noisy = model.predict_with_variance(x, include_noise=True)[1]
        self.assertGreater(noisy, model.predict_with_variance(x)[1])
        for level in (0, 4, True, 1.0):
            with self.assertRaises(ValueError):
                model.predict_with_variance(x, level=level)  # type: ignore[arg-type]

    def test_structure_validation(self) -> None:
        """Mismatched levels and scale factors are rejected."""
        model = self.recursive()
        with self.assertRaises(ValueError):
            RecursiveCoKriging(model.levels, (1.0,))
        with self.assertRaises(ValueError):
            RecursiveCoKriging((), ())
        one_d = gaussian_process_from_hyperparameters(
            [(0.0,), (1.0,)], [0.0, 1.0], signal_variance=1.0, length_scales=(1.0,)
        )
        with self.assertRaises(ValueError):
            RecursiveCoKriging((model.levels[0], one_d), (1.0,))


class FixedHyperparameterGaussianProcessTests(unittest.TestCase):
    """``gaussian_process_from_hyperparameters`` is plain GP conditioning."""

    def test_matches_dense_algebra_and_interpolates(self) -> None:
        """Mean and variance equal ``m + k'K^-1(y-m)`` and ``k** - k'K^-1k``."""
        points = [(0.0,), (0.3,), (0.7,), (1.0,)]
        values = [1.0, -0.5, 0.25, 2.0]
        model = gaussian_process_from_hyperparameters(
            points,
            values,
            kernel="squared_exponential",
            signal_variance=1.5,
            length_scales=(0.4,),
            noise_variance=0.01,
            mean=0.3,
        )
        matrix = tuple(
            tuple(
                kernel_value("squared_exponential", a, b, (0.4,), 1.5)
                + (0.01 if a == b else 0.0)
                for b in points
            )
            for a in points
        )
        x = (0.45,)
        cross = tuple(
            kernel_value("squared_exponential", x, b, (0.4,), 1.5) for b in points
        )
        residual = tuple(value - 0.3 for value in values)
        mean = 0.3 + dot(cross, solve_spd(matrix, residual))
        variance = 1.5 - dot(cross, solve_spd(matrix, cross))
        got = model.predict_with_variance(x)
        self.assertAlmostEqual(got[0], mean, delta=1e-12)
        self.assertAlmostEqual(got[1], variance, delta=1e-12)
        exact = gaussian_process_from_hyperparameters(
            points, values, signal_variance=1.0, length_scales=(0.5,)
        )
        for point, value in zip(points, values, strict=True):
            self.assertAlmostEqual(exact.predict(point), value, delta=1e-9)
        self.assertTrue(exact.converged)
        self.assertEqual(exact.evaluations, 1)

    def test_validation(self) -> None:
        """Invalid hyperparameters and shapes raise ``ValueError``."""
        points = [(0.0,), (1.0,)]
        cases = [
            {"signal_variance": 0.0, "length_scales": (1.0,)},
            {"signal_variance": 1.0, "length_scales": (0.0,)},
            {"signal_variance": 1.0, "length_scales": (1.0, 1.0)},
            {"signal_variance": 1.0, "length_scales": (1.0,), "noise_variance": -1.0},
            {"signal_variance": 1e20, "length_scales": (1.0,)},
            {"signal_variance": 1.0, "length_scales": (1.0,), "kernel": "cubic"},
        ]
        for options in cases:
            with self.assertRaises(ValueError):
                gaussian_process_from_hyperparameters(
                    points,
                    [0.0, 1.0],
                    **options,  # type: ignore[arg-type]
                )
        with self.assertRaises(ValueError):
            gaussian_process_from_hyperparameters(
                points, [0.0], signal_variance=1.0, length_scales=(1.0,)
            )


class FittedCoKrigingTests(unittest.TestCase):
    """Fitting on a two-fidelity pair with few expensive runs."""

    low_design = tuple((i / 10.0,) for i in range(11))
    high_design = ((0.0,), (0.4,), (0.6,), (1.0,))

    def data(self) -> tuple[list[tuple[tuple[float, ...], ...]], list[list[float]]]:
        """Return the nested designs and both response vectors."""
        return (
            [self.low_design, self.high_design],
            [
                [cheap(x) for (x,) in self.low_design],
                [expensive(x) for (x,) in self.high_design],
            ],
        )

    def rmse(self, model: object) -> float:
        """Return the high-fidelity root mean squared error on a fine grid."""
        grid = [(i + 0.5) / 100.0 for i in range(100)]
        return math.sqrt(
            math.fsum(
                (model.predict((x,)) - expensive(x)) ** 2  # type: ignore[attr-defined]
                for x in grid
            )
            / len(grid)
        )

    def test_beats_single_fidelity_gp_with_four_expensive_runs(self) -> None:
        """Co-kriging recovers ``rho = 2`` and cuts the error at least tenfold."""
        designs, responses = self.data()
        model = fit_recursive_cokriging(designs, responses, seed=1)
        single = fit_gaussian_process(self.high_design, responses[1], seed=1)
        self.assertIsInstance(model.converged, bool)
        self.assertTrue(model.scale_estimated)
        self.assertAlmostEqual(model.scale_factors[0], 2.0, delta=0.05)
        self.assertGreater(model.likelihood_evaluations, 0)
        error, reference = self.rmse(model), self.rmse(single)
        self.assertLess(error, 0.5)
        self.assertLess(error, 0.1 * reference)
        for (x,) in self.high_design:
            mean, variance = model.predict_with_variance((x,))
            self.assertAlmostEqual(mean, expensive(x), delta=1e-3)
            self.assertLess(variance, 1e-4)

    def test_fixed_scale_factor_and_reproducibility(self) -> None:
        """A given ``rho`` is used as is; a seed reproduces the fit."""
        designs, responses = self.data()
        fixed = fit_recursive_cokriging(
            designs, responses, scale_factors=(2.0,), seed=4
        )
        self.assertEqual(fixed.scale_factors, (2.0,))
        self.assertFalse(fixed.scale_estimated)
        self.assertLess(self.rmse(fixed), 0.5)
        again = fit_recursive_cokriging(
            designs, responses, scale_factors=(2.0,), seed=4
        )
        self.assertEqual(
            fixed.predict_with_variance((0.33,)), again.predict_with_variance((0.33,))
        )
        first = fit_recursive_cokriging(designs, responses, seed=9)
        second = fit_recursive_cokriging(designs, responses, seed=9)
        self.assertEqual(first.scale_factors, second.scale_factors)

    def test_single_level_is_ordinary_kriging(self) -> None:
        """One level gives a GP that interpolates its own data."""
        designs, responses = self.data()
        model = fit_recursive_cokriging(designs[:1], responses[:1], seed=2)
        self.assertEqual(model.scale_factors, ())
        self.assertFalse(model.scale_estimated)
        for (x,) in self.low_design:
            self.assertAlmostEqual(model.predict((x,)), cheap(x), delta=1e-3)

    def test_exact_linear_relation_between_levels(self) -> None:
        """A level that is exactly ``rho z + beta`` is reproduced everywhere."""
        designs, responses = self.data()
        responses[1] = [3.0 * cheap(x) - 1.0 for (x,) in self.high_design]
        model = fit_recursive_cokriging(designs, responses, seed=3)
        self.assertAlmostEqual(model.scale_factors[0], 3.0, delta=1e-6)
        for (x,) in self.low_design:
            self.assertAlmostEqual(
                model.predict((x,)), 3.0 * cheap(x) - 1.0, delta=1e-2
            )

    def test_input_validation(self) -> None:
        """Non-nested or repeated designs and bad options are rejected."""
        designs, responses = self.data()
        with self.assertRaisesRegex(ValueError, "nested"):
            fit_recursive_cokriging(
                [self.low_design, ((0.05,), (0.4,), (0.6,), (1.0,))], responses
            )
        with self.assertRaisesRegex(ValueError, "repeated"):
            fit_recursive_cokriging(
                [self.low_design, ((0.4,), (0.4,), (0.6,), (1.0,))], responses
            )
        with self.assertRaises(ValueError):
            fit_recursive_cokriging(designs, responses[:1])
        with self.assertRaises(ValueError):
            fit_recursive_cokriging(designs, responses, scale_factors=(1.0, 2.0))
        with self.assertRaises(ValueError):
            fit_recursive_cokriging(designs, [responses[0], responses[1][:3]])
        with self.assertRaises(ValueError):
            fit_recursive_cokriging([], [])
        with self.assertRaisesRegex(ValueError, "level 2"):
            fit_recursive_cokriging(
                [self.low_design, self.high_design[:2]],
                [responses[0], responses[1][:2]],
            )


class StochasticKrigingAlgebraTests(unittest.TestCase):
    """Exact identities of the stochastic kriging predictor and its MSE."""

    points = ((0.0,), (0.25,), (0.5,), (0.8,), (1.0,))
    means = (1.2, 0.4, -0.3, 0.9, 1.6)
    variances = (0.02, 0.2, 0.05, 0.01, 0.3)
    signal = 1.3
    scales = (0.35,)

    def sigma(
        self, variances: tuple[float, ...], kernel: str = "squared_exponential"
    ) -> tuple[tuple[float, ...], ...]:
        """Return ``S_M + S_e`` as a dense matrix."""
        return tuple(
            tuple(
                kernel_value(kernel, a, b, self.scales, self.signal)
                + (variances[i] if i == j else 0.0)
                for j, b in enumerate(self.points)
            )
            for i, a in enumerate(self.points)
        )

    def cross(self, x: tuple[float, ...]) -> tuple[float, ...]:
        """Return ``S_M(x, .)``."""
        return tuple(
            kernel_value("squared_exponential", x, b, self.scales, self.signal)
            for b in self.points
        )

    def test_known_trend_equals_heteroscedastic_gp_algebra(self) -> None:
        """With known ``b0`` the predictor is GP regression with diagonal noise."""
        model = stochastic_kriging_from_summaries(
            self.points,
            self.means,
            self.variances,
            signal_variance=self.signal,
            length_scales=self.scales,
            trend=0.5,
        )
        self.assertFalse(model.trend_estimated)
        self.assertEqual(model.trend, 0.5)
        matrix = self.sigma(self.variances)
        residual = tuple(value - 0.5 for value in self.means)
        for x in ((0.1,), (0.5,), (0.66,), (1.4,)):
            cross = self.cross(x)
            mean = 0.5 + dot(cross, solve_spd(matrix, residual))
            mse = self.signal - dot(cross, solve_spd(matrix, cross))
            got = model.predict_with_variance(x)
            self.assertAlmostEqual(got[0], mean, delta=1e-12)
            self.assertAlmostEqual(got[1], mse, delta=1e-12)
            self.assertEqual(model.predict(x), got[0])

    def test_equal_variances_equal_the_homoscedastic_library_gp(self) -> None:
        """Equal ``V/n`` reproduces the library GP with that noise variance."""
        common = (0.07,) * len(self.points)
        for kernel in ("squared_exponential", "matern52"):
            model = stochastic_kriging_from_summaries(
                self.points,
                self.means,
                common,
                kernel=kernel,
                signal_variance=self.signal,
                length_scales=self.scales,
                trend=-0.2,
            )
            reference = gaussian_process_from_hyperparameters(
                self.points,
                self.means,
                kernel=kernel,
                signal_variance=self.signal,
                length_scales=self.scales,
                noise_variance=0.07,
                mean=-0.2,
            )
            for x in ((0.05,), (0.4,), (0.9,)):
                got = model.predict_with_variance(x)
                want = reference.predict_with_variance(x)
                self.assertAlmostEqual(got[0], want[0], delta=1e-11)
                self.assertAlmostEqual(got[1], want[1], delta=1e-11)
            self.assertAlmostEqual(
                model.log_likelihood, reference.log_marginal_likelihood, delta=1e-10
            )

    def test_estimated_trend_is_the_best_linear_unbiased_predictor(self) -> None:
        """Explicit BLUP weights reproduce the predictor and its MSE."""
        model = stochastic_kriging_from_summaries(
            self.points,
            self.means,
            self.variances,
            signal_variance=self.signal,
            length_scales=self.scales,
        )
        self.assertTrue(model.trend_estimated)
        matrix = self.sigma(self.variances)
        ones = (1.0,) * len(self.points)
        solved_ones = solve_spd(matrix, ones)
        self.assertAlmostEqual(
            model.trend, dot(solved_ones, self.means) / sum(solved_ones), delta=1e-12
        )
        for x in ((0.1,), (0.5,), (2.5,)):
            cross = self.cross(x)
            solved = solve_spd(matrix, cross)
            multiplier = (1.0 - dot(ones, solved)) / sum(solved_ones)
            weights = tuple(
                a + multiplier * b for a, b in zip(solved, solved_ones, strict=True)
            )
            self.assertAlmostEqual(sum(weights), 1.0, delta=1e-12)
            mean = dot(weights, self.means)
            sandwich = dot(weights, tuple(dot(row, weights) for row in matrix))
            mse = self.signal - 2.0 * dot(weights, cross) + sandwich
            got = model.predict_with_variance(x)
            self.assertAlmostEqual(got[0], mean, delta=1e-11)
            self.assertAlmostEqual(got[1], mse, delta=1e-11)
        far = model.predict_with_variance((50.0,))
        self.assertAlmostEqual(far[0], model.trend, delta=1e-12)
        self.assertAlmostEqual(far[1], self.signal + 1.0 / sum(solved_ones), delta=1e-9)

    def test_noise_free_points_are_interpolated(self) -> None:
        """A design point with zero intrinsic variance is reproduced exactly."""
        variances = (0.0, 0.2, 0.05, 0.01, 0.3)
        model = stochastic_kriging_from_summaries(
            self.points,
            self.means,
            variances,
            signal_variance=self.signal,
            length_scales=self.scales,
        )
        mean, mse = model.predict_with_variance(self.points[0])
        self.assertAlmostEqual(mean, self.means[0], delta=1e-9)
        self.assertLess(mse, 1e-9)
        noisy = model.predict_with_variance(self.points[4])
        self.assertGreater(noisy[1], 1e-3)


class StochasticKrigingFitTests(unittest.TestCase):
    """Fitting stochastic kriging to replicated noisy simulation output."""

    @staticmethod
    def truth(x: float) -> float:
        """Return the mean response surface."""
        return math.sin(2.0 * math.pi * x) + x

    def replicated(
        self, seed: int, count: int = 12
    ) -> tuple[list[tuple[float]], list[list[float]]]:
        """Simulate replications whose noise grows with ``x``."""
        generator = random.Random(seed)
        points = [(i / 9.0,) for i in range(10)]
        outputs = [
            [
                self.truth(x) + generator.gauss(0.0, 0.1 + 0.6 * x)
                for _ in range(count + 2 * index)
            ]
            for index, (x,) in enumerate(points)
        ]
        return points, outputs

    def test_summaries_and_likelihood(self) -> None:
        """Means, ``V/n``, counts, and the likelihood ordering are right."""
        points, outputs = self.replicated(5)
        model = fit_stochastic_kriging(points, outputs, seed=1)
        self.assertIsInstance(model, StochasticKriging)
        self.assertIsInstance(model, ProbabilisticSurrogate)
        self.assertEqual(model.replications, tuple(len(row) for row in outputs))
        for row, mean, variance in zip(
            outputs, model.means, model.intrinsic_variances, strict=True
        ):
            centre = sum(row) / len(row)
            spread = sum((v - centre) ** 2 for v in row) / (len(row) - 1)
            self.assertAlmostEqual(mean, centre, delta=1e-12)
            self.assertAlmostEqual(variance, spread / len(row), delta=1e-12)
        self.assertTrue(model.converged)
        self.assertEqual(model.restarts, 3)
        self.assertGreater(model.evaluations, 3)
        start = stochastic_kriging_from_summaries(
            points,
            model.means,
            model.intrinsic_variances,
            signal_variance=1.0,
            length_scales=(0.5,),
        )
        self.assertGreaterEqual(model.log_likelihood, start.log_likelihood)
        again = fit_stochastic_kriging(points, outputs, seed=1)
        self.assertEqual(model.length_scales, again.length_scales)

    def test_prediction_errors_are_consistent_with_the_reported_mse(self) -> None:
        """Standardized errors over seeds and points have mean square near one."""
        squares = []
        errors = []
        for seed in range(8):
            points, outputs = self.replicated(100 + seed)
            model = fit_stochastic_kriging(points, outputs, seed=seed)
            for index in range(20):
                x = (index + 0.5) / 20.0
                mean, mse = model.predict_with_variance((x,))
                errors.append(abs(mean - self.truth(x)))
                squares.append((mean - self.truth(x)) ** 2 / mse)
        mean_square = sum(squares) / len(squares)
        self.assertGreater(mean_square, 0.3)
        self.assertLess(mean_square, 3.0)
        self.assertLess(sum(errors) / len(errors), 0.15)

    def test_input_validation(self) -> None:
        """Bad replications, variances, and hyperparameters are rejected."""
        points = [(0.0,), (0.5,), (1.0,)]
        with self.assertRaisesRegex(ValueError, "two replications"):
            fit_stochastic_kriging(points, [[1.0, 2.0], [1.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            fit_stochastic_kriging(points, [[1.0, 2.0], [1.0, 3.0]])
        with self.assertRaises(ValueError):
            stochastic_kriging_from_summaries(points, [1.0, 2.0, 3.0], [0.1, -0.1, 0.1])
        with self.assertRaises(ValueError):
            stochastic_kriging_from_summaries(
                points, [1.0, 2.0, 3.0], [0.1] * 3, kernel="cubic"
            )
        with self.assertRaises(ValueError):
            stochastic_kriging_from_summaries(
                points, [1.0, 2.0, 3.0], [0.1] * 3, signal_variance=-1.0
            )
        with self.assertRaises(ValueError):
            stochastic_kriging_from_summaries(
                points, [1.0, 2.0, 3.0], [0.1] * 3, length_scales=(1.0, 1.0)
            )
        with self.assertRaises(ValueError):
            stochastic_kriging_from_summaries(
                points, [1.0, 2.0, 3.0], [0.1] * 3, restarts=0
            )
        with self.assertRaisesRegex(ValueError, "constant"):
            stochastic_kriging_from_summaries(points, [1.0, 1.0, 1.0], [0.1] * 3)
        model = stochastic_kriging_from_summaries(
            points,
            [1.0, 2.0, 3.0],
            [0.1] * 3,
            signal_variance=1.0,
            length_scales=(0.5,),
        )
        with self.assertRaises(ValueError):
            model.predict((0.1, 0.2))


class KernelIntegralTests(unittest.TestCase):
    """Closed-form kernel means and double integrals against quadrature."""

    @staticmethod
    def density(measure: GaussianMeasure | UniformMeasure, t: float) -> float:
        """Return the density of a one-dimensional measure."""
        if isinstance(measure, GaussianMeasure):
            z = (t - measure.mean) / measure.sd
            return math.exp(-0.5 * z * z) / (measure.sd * math.sqrt(2.0 * math.pi))
        return 1.0 / (measure.upper - measure.lower)

    @staticmethod
    def limits(measure: GaussianMeasure | UniformMeasure) -> tuple[float, float]:
        """Return the integration range of a measure."""
        if isinstance(measure, GaussianMeasure):
            return -math.inf, math.inf
        return measure.lower, measure.upper

    def test_kernel_mean_and_double_integral_match_quadrature(self) -> None:
        """Both closed forms agree with adaptive quadrature to 1e-9."""
        measures = (
            GaussianMeasure(0.3, 0.7),
            GaussianMeasure(-2.0, 0.2),
            UniformMeasure(-1.0, 2.0),
            UniformMeasure(0.0, 0.3),
        )
        for measure in measures:
            low, high = self.limits(measure)
            for scale in (0.15, 0.6, 2.5):
                for node in (-1.5, 0.4, 3.0):
                    reference = integrate(
                        lambda t, node=node, scale=scale, measure=measure: (
                            math.exp(-0.5 * ((t - node) / scale) ** 2)
                            * self.density(measure, t)
                        ),
                        low,
                        high,
                        rel_tol=1e-12,
                    )
                    self.assertAlmostEqual(
                        kernel_mean(measure, node, scale), reference.value, delta=1e-9
                    )
                double = integrate(
                    lambda u, scale=scale, measure=measure: (
                        kernel_mean(measure, u, scale) * self.density(measure, u)
                    ),
                    low,
                    high,
                    rel_tol=1e-12,
                )
                self.assertAlmostEqual(
                    kernel_double_integral(measure, scale), double.value, delta=1e-9
                )

    def test_limits_and_validation(self) -> None:
        """Long length scales give one; bad arguments raise."""
        for measure in (GaussianMeasure(1.0, 0.5), UniformMeasure(-1.0, 1.0)):
            self.assertAlmostEqual(kernel_mean(measure, 0.0, 1e6), 1.0, delta=1e-9)
            self.assertAlmostEqual(
                kernel_double_integral(measure, 1e6), 1.0, delta=1e-9
            )
            self.assertLess(kernel_double_integral(measure, 1e-3), 0.01)
            with self.assertRaises(ValueError):
                kernel_mean(measure, 0.0, 0.0)
            with self.assertRaises(ValueError):
                kernel_double_integral(measure, -1.0)
        with self.assertRaises(ValueError):
            GaussianMeasure(0.0, 0.0)
        with self.assertRaises(ValueError):
            UniformMeasure(1.0, 1.0)
        with self.assertRaises(TypeError):
            kernel_mean("normal", 0.0, 1.0)  # type: ignore[arg-type]


class BayesianQuadratureTests(unittest.TestCase):
    """Posterior mean and variance of integrals."""

    def test_kernel_sections_are_integrated_exactly(self) -> None:
        """For ``f = k(., x_j)`` the estimate is ``z_j`` with zero variance."""
        nodes = [(-1.5,), (-0.4,), (0.3,), (1.1,), (2.0,)]
        for measure in (GaussianMeasure(0.2, 0.8), UniformMeasure(-2.0, 2.5)):
            for j, centre in enumerate(nodes):
                values = [
                    kernel_value("squared_exponential", node, centre, (0.7,), 1.4)
                    for node in nodes
                ]
                result = bayesian_quadrature(
                    nodes,
                    values,
                    [measure],
                    length_scales=(0.7,),
                    signal_variance=1.4,
                )
                self.assertAlmostEqual(
                    result.estimate, result.kernel_means[j], delta=1e-10
                )
                self.assertAlmostEqual(
                    result.kernel_means[j],
                    1.4 * kernel_mean(measure, centre[0], 0.7),
                    delta=1e-14,
                )

    def test_variance_is_non_negative_and_decreases_with_nodes(self) -> None:
        """Adding nodes never increases the posterior variance."""
        generator = random.Random(7)
        nodes = [(generator.uniform(-2.5, 2.5),) for _ in range(9)]
        for measure in (GaussianMeasure(0.0, 1.0), UniformMeasure(-2.5, 2.5)):
            previous = kernel_double_integral(measure, 0.8)
            for count in range(1, len(nodes) + 1):
                result = bayesian_quadrature(
                    nodes[:count],
                    [math.cos(x) for (x,) in nodes[:count]],
                    [measure],
                    length_scales=(0.8,),
                    noise_variance=1e-10,
                )
                self.assertAlmostEqual(
                    result.prior_variance,
                    kernel_double_integral(measure, 0.8),
                    delta=1e-15,
                )
                self.assertGreaterEqual(result.variance, 0.0)
                self.assertLessEqual(result.variance, previous + 1e-12)
                self.assertEqual(result.standard_deviation, math.sqrt(result.variance))
                previous = result.variance
            self.assertLess(previous, 0.25 * result.prior_variance)

    def test_agrees_with_gauss_hermite_on_smooth_integrands(self) -> None:
        """Gaussian-measure integrals match a 40-point Gauss-Hermite rule."""
        abscissae, weights = gauss_hermite(40)
        nodes = [(-3.0 + 0.5 * i,) for i in range(13)]
        integrands = (
            math.cos,
            lambda x: math.exp(0.5 * x),
            lambda x: 1.0 / (1.0 + x * x),
            lambda x: math.sin(x) + 0.3 * x * x,
        )
        for function in integrands:
            reference = math.fsum(
                w * function(0.3 + 0.9 * z)
                for z, w in zip(abscissae, weights, strict=True)
            )
            result = bayesian_quadrature(
                [(0.3 + 0.9 * x,) for (x,) in nodes],
                [function(0.3 + 0.9 * x) for (x,) in nodes],
                [GaussianMeasure(0.3, 0.9)],
                length_scales=(1.0,),
                noise_variance=1e-10,
            )
            self.assertIsInstance(result, BayesianQuadratureResult)
            self.assertAlmostEqual(result.estimate, reference, delta=2e-3)
            self.assertEqual(result.evaluations, 13)
        self.assertAlmostEqual(
            math.fsum(w * math.cos(z) for z, w in zip(abscissae, weights, strict=True)),
            math.exp(-0.5),
            delta=1e-12,
        )

    def test_product_measure_in_two_dimensions(self) -> None:
        """A Gaussian-by-uniform integral of a separable integrand is right."""
        nodes = [(-2.4 + 0.8 * i, 0.125 + 0.25 * j) for i in range(7) for j in range(4)]
        values = [math.cos(x) * math.exp(y) for x, y in nodes]
        result = bayesian_quadrature(
            nodes,
            values,
            [GaussianMeasure(0.0, 1.0), UniformMeasure(0.0, 1.0)],
            length_scales=(1.0, 0.8),
            noise_variance=1e-10,
        )
        exact = math.exp(-0.5) * (math.e - 1.0)
        self.assertAlmostEqual(result.estimate, exact, delta=5e-3)
        self.assertAlmostEqual(
            result.prior_variance,
            kernel_double_integral(GaussianMeasure(0.0, 1.0), 1.0)
            * kernel_double_integral(UniformMeasure(0.0, 1.0), 0.8),
            delta=1e-15,
        )
        self.assertAlmostEqual(
            result.estimate,
            dot(result.weights, tuple(values)),
            delta=1e-12,
        )

    def test_prior_mean_and_estimated_hyperparameters(self) -> None:
        """A constant is integrated exactly by its prior mean; fitting works."""
        nodes = [(0.1,), (0.5,), (0.9,)]
        constant = bayesian_quadrature(
            nodes,
            [3.0, 3.0, 3.0],
            [UniformMeasure(0.0, 1.0)],
            length_scales=(0.2,),
            prior_mean=3.0,
        )
        self.assertAlmostEqual(constant.estimate, 3.0, delta=1e-12)
        grid = [(i / 8.0,) for i in range(9)]
        with warnings.catch_warnings():
            # A squared-exponential fit to nine smooth values is ill-conditioned.
            warnings.simplefilter("ignore", GaussianProcessConditioningWarning)
            fitted = bayesian_quadrature(
                grid,
                [math.exp(x) for (x,) in grid],
                [UniformMeasure(0.0, 1.0)],
                seed=3,
            )
        self.assertTrue(fitted.hyperparameters_estimated)
        self.assertAlmostEqual(fitted.estimate, math.e - 1.0, delta=1e-4)
        self.assertGreaterEqual(fitted.variance, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GaussianProcessConditioningWarning)
            again = bayesian_quadrature(
                grid,
                [math.exp(x) for (x,) in grid],
                [UniformMeasure(0.0, 1.0)],
                seed=3,
            )
        self.assertEqual(fitted.estimate, again.estimate)

    def test_noise_variance_is_a_floor_for_the_fitted_nugget(self) -> None:
        """With fitted hyperparameters the documented remedy must be honoured.

        For a quadratic integrand the fitted length scale is so long that the
        kernel matrix is numerically singular. The error message advises a
        larger ``noise_variance``; before the fix that argument was ignored
        on the fitted path, so the call failed whatever value was passed.
        Whether the call without a nugget fails is not asserted: the fitted
        length scale moves with float rounding (Python 3.11 sums differently
        from 3.12+), and on Python 3.11 the default call succeeds.
        """
        nodes = [(-3.0 + 0.5 * i,) for i in range(13)]
        values = [x * x for (x,) in nodes]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GaussianProcessConditioningWarning)
            result = bayesian_quadrature(
                nodes, values, [GaussianMeasure()], noise_variance=1e-6, seed=0
            )
            with self.assertRaises(ValueError):
                bayesian_quadrature(
                    nodes, values, [GaussianMeasure()], noise_variance=-1.0, seed=0
                )
        self.assertEqual(result.noise_variance, 1e-6)
        self.assertAlmostEqual(result.estimate, 1.0, delta=1e-4)
        self.assertGreater(result.standard_deviation, abs(result.estimate - 1.0))

    def test_standard_deviation_is_not_an_error_bound(self) -> None:
        """The reported deviation ignores the values; errors can be far larger.

        Pins the documented warning: same nodes and kernel give the same
        standard deviation for every integrand, while the actual error for
        ``x^2`` and ``|x|`` is more than ten and two hundred deviations.
        """
        nodes = [(-3.0 + 0.5 * i,) for i in range(13)]
        results = {
            name: bayesian_quadrature(
                nodes,
                [function(x) for (x,) in nodes],
                [GaussianMeasure()],
                length_scales=(1.0,),
                noise_variance=1e-10,
            )
            for name, function in (
                ("cos", math.cos),
                ("square", lambda x: x * x),
                ("abs", abs),
            )
        }
        deviation = results["cos"].standard_deviation
        self.assertAlmostEqual(deviation, 8.98e-5, delta=1e-6)
        for result in results.values():
            self.assertAlmostEqual(result.standard_deviation, deviation, delta=1e-12)
        truths = {
            "cos": math.exp(-0.5),
            "square": 1.0,
            "abs": math.sqrt(2.0 / math.pi),
        }
        ratios = {
            name: abs(result.estimate - truths[name]) / deviation
            for name, result in results.items()
        }
        self.assertLess(ratios["cos"], 1.0)
        self.assertGreater(ratios["square"], 10.0)
        self.assertGreater(ratios["abs"], 200.0)

    def test_input_validation(self) -> None:
        """Shape mismatches, bad hyperparameters, and singular kernels raise."""
        measure = [GaussianMeasure()]
        with self.assertRaises(ValueError):
            bayesian_quadrature([], [], measure, length_scales=(1.0,))
        with self.assertRaises(ValueError):
            bayesian_quadrature([(0.0,)], [1.0, 2.0], measure, length_scales=(1.0,))
        with self.assertRaises(ValueError):
            bayesian_quadrature([(0.0, 1.0)], [1.0], measure, length_scales=(1.0,))
        with self.assertRaises(ValueError):
            bayesian_quadrature([(0.0,)], [1.0], measure, length_scales=(0.0,))
        with self.assertRaises(ValueError):
            bayesian_quadrature(
                [(0.0,)], [1.0], measure, length_scales=(1.0,), signal_variance=0.0
            )
        with self.assertRaises(ValueError):
            bayesian_quadrature(
                [(0.0,)], [1.0], measure, length_scales=(1.0,), noise_variance=-1.0
            )
        with self.assertRaises(TypeError):
            bayesian_quadrature(
                [(0.0,)],
                [1.0],
                ["gaussian"],  # type: ignore[list-item]
                length_scales=(1.0,),
            )
        with self.assertRaisesRegex(ValueError, "positive definite"):
            bayesian_quadrature(
                [(0.0,), (0.0,)], [1.0, 1.0], measure, length_scales=(1.0,)
            )


if __name__ == "__main__":
    unittest.main()
