"""Tests for Stein control variates, control functionals, and the KSD (ADR-0072)."""

from __future__ import annotations

import math
import random
import statistics
import unittest
from collections.abc import Callable, Sequence

from probstat.monte_carlo import (
    estimate_control_functional,
    estimate_control_functional_weights,
    estimate_zero_variance,
    kernel_stein_discrepancy,
    median_heuristic_length_scale,
    stein_kernel,
    stein_polynomial_controls,
)

MEAN = (1.0, -0.5)
COVARIANCE = ((2.0, 0.6), (0.6, 1.0))
_DETERMINANT = COVARIANCE[0][0] * COVARIANCE[1][1] - COVARIANCE[0][1] ** 2
PRECISION = (
    (COVARIANCE[1][1] / _DETERMINANT, -COVARIANCE[0][1] / _DETERMINANT),
    (-COVARIANCE[1][0] / _DETERMINANT, COVARIANCE[0][0] / _DETERMINANT),
)


def gaussian_sample(size: int, seed: int) -> list[tuple[float, float]]:
    """Draw independent points from the correlated bivariate Gaussian target.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2] (Cholesky colouring).

    """
    generator = random.Random(seed)
    l00 = math.sqrt(COVARIANCE[0][0])
    l10 = COVARIANCE[1][0] / l00
    l11 = math.sqrt(COVARIANCE[1][1] - l10 * l10)
    points = []
    for _ in range(size):
        z0, z1 = generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)
        points.append((MEAN[0] + l00 * z0, MEAN[1] + l10 * z0 + l11 * z1))
    return points


def gaussian_score(point: Sequence[float]) -> tuple[float, float]:
    """Return ``-P (x - m)`` for the bivariate Gaussian target.

    References:
        [EQ-MC-STEIN-001].

    """
    d0, d1 = point[0] - MEAN[0], point[1] - MEAN[1]
    return (
        -(PRECISION[0][0] * d0 + PRECISION[0][1] * d1),
        -(PRECISION[1][0] * d0 + PRECISION[1][1] * d1),
    )


def quadratic(point: Sequence[float]) -> float:
    """Return a full quadratic polynomial with a cross term.

    References:
        [EQ-MC-STEIN-001].

    """
    x, y = point
    return x * x + 3.0 * x * y - y * y + 2.0 * x - y + 4.0


QUADRATIC_MEAN = (
    COVARIANCE[0][0]
    + MEAN[0] ** 2
    + 3.0 * (COVARIANCE[0][1] + MEAN[0] * MEAN[1])
    - (COVARIANCE[1][1] + MEAN[1] ** 2)
    + 2.0 * MEAN[0]
    - MEAN[1]
    + 4.0
)


def standard_score(value: float) -> float:
    """Return the score ``-x`` of the standard normal target.

    References:
        [EQ-MC-STEIN-001].

    """
    return -value


def normal_draws(size: int, seed: int) -> list[float]:
    """Draw independent standard-normal values.

    References:
        [REF-OWEN-2013, Chapter 2].

    """
    generator = random.Random(seed)
    return [generator.gauss(0.0, 1.0) for _ in range(size)]


def root_mean_square(errors: Sequence[float]) -> float:
    """Return the root-mean-square of a list of errors.

    References:
        [EQ-MC-SE-001].

    """
    return math.sqrt(math.fsum(e * e for e in errors) / len(errors))


class Counter:
    """Count the calls of a wrapped function."""

    def __init__(self, function: Callable[..., object]) -> None:
        """Store the wrapped function and reset the count."""
        self.function = function
        self.calls = 0

    def __call__(self, value: object) -> object:
        """Forward the call and increment the count."""
        self.calls += 1
        return self.function(value)


class ZeroVarianceTests(unittest.TestCase):
    """Exactness, variance reduction, unbiasedness, and costs of ZV-CV."""

    def test_degree_two_is_exact_for_gaussian_quadratics(self) -> None:
        """Degree-2 ZV-CV integrates every quadratic exactly under a Gaussian."""
        points = gaussian_sample(40, seed=3)
        for method in ("same-sample", "split-sample"):
            result = estimate_zero_variance(
                points, quadratic, gaussian_score, method=method
            )
            self.assertAlmostEqual(result.estimate, QUADRATIC_MEAN, delta=1e-11)
            self.assertLess(result.adjusted_variance, 1e-20 * result.raw_variance)
            self.assertLess(result.standard_error, 1e-10)
            self.assertGreater(result.variance_reduction_factor, 1e15)

    def test_degree_one_is_exact_for_linear_functions(self) -> None:
        """Degree-1 controls span every centred linear function of a Gaussian."""
        points = gaussian_sample(30, seed=4)
        result = estimate_zero_variance(
            points,
            lambda p: 2.0 * p[0] - 3.0 * p[1] + 1.0,
            gaussian_score,
            degree=1,
        )
        self.assertAlmostEqual(
            result.estimate, 2.0 * MEAN[0] - 3.0 * MEAN[1] + 1.0, delta=1e-12
        )
        self.assertEqual(len(result.coefficients[0]), 2)

    def test_variance_reduction_on_a_non_polynomial_integrand(self) -> None:
        """``exp(x/2)`` under N(0,1) gains well above one, and the CI covers."""
        draws = normal_draws(200, seed=11)
        truth = math.exp(0.125)
        for degree, minimum in ((1, 3.0), (2, 30.0)):
            for method in ("same-sample", "split-sample"):
                result = estimate_zero_variance(
                    draws,
                    lambda v: math.exp(0.5 * v),
                    standard_score,
                    degree=degree,
                    method=method,
                )
                self.assertGreater(result.variance_reduction_factor, minimum)
                self.assertLess(
                    abs(result.estimate - truth), 4.0 * result.standard_error
                )

    def test_split_sample_estimator_is_unbiased_across_seeds(self) -> None:
        """Repeated cross-fitted estimates average to the truth within 4 SE."""
        truth = math.exp(0.5)
        estimates = [
            estimate_zero_variance(
                normal_draws(12, seed=5_000 + replicate),
                math.exp,
                standard_score,
                method="split-sample",
            ).estimate
            for replicate in range(400)
        ]
        center = statistics.fmean(estimates)
        error = statistics.stdev(estimates) / math.sqrt(len(estimates))
        self.assertLess(abs(center - truth), 4.0 * error)

    def test_ridge_shrinks_coefficients(self) -> None:
        """A large ridge penalty moves the estimate toward the plain mean."""
        draws = normal_draws(50, seed=21)
        plain = estimate_zero_variance(draws, math.exp, standard_score)
        ridged = estimate_zero_variance(draws, math.exp, standard_score, ridge=1e8)
        self.assertLess(
            max(abs(c) for c in ridged.coefficients[0]),
            max(abs(c) for c in plain.coefficients[0]),
        )
        self.assertAlmostEqual(
            ridged.estimate, statistics.fmean(math.exp(v) for v in draws), delta=1e-4
        )

    def test_markov_chain_standard_error_accounts_for_correlation(self) -> None:
        """An AR(1) chain targeting N(0,1) gets a larger MCMC standard error."""
        generator = random.Random(9)
        state, chain = 0.0, []
        for _ in range(500):
            state = 0.8 * state + 0.6 * generator.gauss(0.0, 1.0)
            chain.append(state)
        iid = estimate_zero_variance(chain, math.exp, standard_score)
        mcmc = estimate_zero_variance(
            chain, math.exp, standard_score, standard_error="mcmc"
        )
        self.assertEqual(iid.estimate, mcmc.estimate)
        self.assertGreater(mcmc.standard_error, iid.standard_error)
        self.assertLess(abs(mcmc.estimate - math.exp(0.5)), 4.0 * mcmc.standard_error)

    def test_integrand_and_score_evaluations_are_counted_separately(self) -> None:
        """Each callable is evaluated exactly once per sample point."""
        integrand, score = Counter(math.exp), Counter(standard_score)
        result = estimate_zero_variance(
            normal_draws(30, seed=2), integrand, score, method="split-sample", folds=3
        )
        self.assertEqual(integrand.calls, 30)
        self.assertEqual(score.calls, 30)
        self.assertEqual(result.integrand_evaluations, 30)
        self.assertEqual(result.score_evaluations, 30)
        self.assertEqual(result.evaluations, 30)
        self.assertEqual(len(result.coefficients), 3)

    def test_polynomial_controls_follow_the_stein_operator(self) -> None:
        """Controls equal ``Laplacian phi + grad phi . u`` for each monomial."""
        controls = stein_polynomial_controls((2.0, 3.0), (0.5, -1.0), degree=2)
        expected = (0.5, -1.0, 2.0 + 2.0 * 2.0 * 0.5, 3.0 * 0.5 + 2.0 * -1.0,
                    2.0 + 2.0 * 3.0 * -1.0)
        for value, target in zip(controls, expected, strict=True):
            self.assertAlmostEqual(value, target, places=14)

    def test_invalid_inputs_raise(self) -> None:
        """Bad degree, ridge, method, folds, sizes, and scores are rejected."""
        draws = normal_draws(20, seed=1)
        with self.assertRaises(ValueError):
            estimate_zero_variance(draws, math.exp, standard_score, degree=3)
        with self.assertRaises(ValueError):
            estimate_zero_variance(draws, math.exp, standard_score, ridge=-1.0)
        with self.assertRaises(ValueError):
            estimate_zero_variance(
                draws, math.exp, standard_score, method="pooled"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            estimate_zero_variance(
                draws, math.exp, standard_score, method="split-sample", folds=1
            )
        with self.assertRaises(ValueError):
            estimate_zero_variance(draws[:3], math.exp, standard_score)
        with self.assertRaises(ValueError):
            estimate_zero_variance([1.0], math.exp, standard_score)
        with self.assertRaises(ValueError):
            estimate_zero_variance(draws, math.exp, lambda v: math.inf)
        with self.assertRaises(ValueError):
            estimate_zero_variance(draws, math.exp, lambda v: (v, v))
        with self.assertRaises(ValueError):
            estimate_zero_variance([(0.0, 1.0), (1.0,)], math.exp, standard_score)
        with self.assertRaises(TypeError):
            estimate_zero_variance(draws, math.exp, None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            estimate_zero_variance(draws, math.exp, standard_score, confidence=1.0)
        with self.assertRaises(ValueError):
            estimate_zero_variance([0.0] * 10, math.exp, standard_score)


class SteinKernelTests(unittest.TestCase):
    """The Stein kernel matches its definition and has zero mean under pi."""

    def test_kernel_matches_finite_difference_definition(self) -> None:
        """``k0`` agrees with central differences of the base kernel in 1-D."""
        x, y, ux, uy, step = 0.3, -0.7, -0.3, 0.7, 1e-4

        def base(a: float, b: float) -> float:
            """Gaussian base kernel with unit length-scale."""
            return math.exp(-0.5 * (a - b) ** 2)

        mixed = (
            base(x + step, y + step)
            - base(x + step, y - step)
            - base(x - step, y + step)
            + base(x - step, y - step)
        ) / (4.0 * step * step)
        dx = (base(x + step, y) - base(x - step, y)) / (2.0 * step)
        dy = (base(x, y + step) - base(x, y - step)) / (2.0 * step)
        reference = mixed + ux * dy + uy * dx + ux * uy * base(x, y)
        self.assertAlmostEqual(stein_kernel(x, y, ux, uy), reference, places=6)

    def test_kernel_has_zero_mean_under_the_target(self) -> None:
        """``int k0(x, y) phi(x) dx = 0`` by trapezoid quadrature on N(0,1)."""
        nodes = [-12.0 + 24.0 * i / 4_000 for i in range(4_001)]
        density = [math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi) for v in nodes]
        for kernel in ("gaussian", "imq"):
            for y in (-1.3, 0.0, 0.8):
                values = [
                    stein_kernel(v, y, -v, -y, kernel=kernel, length_scale=0.9) * p
                    for v, p in zip(nodes, density, strict=True)
                ]
                integral = (24.0 / 4_000) * (
                    math.fsum(values) - 0.5 * (values[0] + values[-1])
                )
                self.assertLess(abs(integral), 1e-9)

    def test_median_heuristic(self) -> None:
        """``sqrt(H_n/2)`` for three collinear points."""
        self.assertAlmostEqual(
            median_heuristic_length_scale([0.0, 1.0, 3.0]), math.sqrt(2.0), places=14
        )

    def test_kernel_parameters_are_validated(self) -> None:
        """Unknown kernels and out-of-range parameters are rejected."""
        with self.assertRaises(ValueError):
            stein_kernel(0.0, 1.0, 0.0, -1.0, kernel="laplace")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            stein_kernel(0.0, 1.0, 0.0, -1.0, length_scale=0.0)
        with self.assertRaises(ValueError):
            stein_kernel(0.0, 1.0, 0.0, -1.0, kernel="imq", exponent=-1.5)
        with self.assertRaises(ValueError):
            stein_kernel((0.0, 1.0), 1.0, 0.0, -1.0)
        with self.assertRaises(ValueError):
            median_heuristic_length_scale([2.0, 2.0, 2.0])


class ControlFunctionalTests(unittest.TestCase):
    """Control functionals beat the Monte Carlo rate on smooth integrands."""

    def test_error_shrinks_faster_than_plain_monte_carlo(self) -> None:
        """RMSE over 8 seeds falls by far more than the MC factor of two.

        Quadrupling ``n`` halves the plain Monte Carlo RMSE. With 8 replicates
        an RMSE estimate has relative standard error about ``1/sqrt(16)``, so
        the log of a ratio of two of them has standard error about
        ``sqrt(2)/4 = 0.35``; the ratio must exceed ``2 exp(3 * 0.35)``.
        """
        truth = math.exp(-0.5)
        sigma = math.sqrt(0.5 * (1.0 + math.exp(-2.0)) - math.exp(-1.0))
        threshold = 2.0 * math.exp(3.0 * math.sqrt(2.0) / 4.0)
        estimators = (estimate_control_functional, estimate_control_functional_weights)
        for estimator in estimators:
            rmse = {}
            for size in (16, 64):
                errors = [
                    estimator(
                        normal_draws(size, seed=1_000 + replicate),
                        math.cos,
                        standard_score,
                    ).estimate
                    - truth
                    for replicate in range(8)
                ]
                rmse[size] = root_mean_square(errors)
            self.assertGreater(rmse[16] / rmse[64], threshold)
            self.assertLess(rmse[64], 0.1 * sigma / math.sqrt(64))

    def test_two_dimensional_integrand(self) -> None:
        """A smooth 2-D integrand beats the plain MC standard error tenfold."""
        truth = math.exp(-1.0)

        def integrand(point: Sequence[float]) -> float:
            """Product of cosines plus a cross term."""
            return math.cos(point[0]) * math.cos(point[1]) + 0.5 * point[0] * point[1]

        def score(point: Sequence[float]) -> tuple[float, float]:
            """Return the standard bivariate normal score."""
            return (-point[0], -point[1])

        errors = []
        plain = []
        for replicate in range(4):
            generator = random.Random(7_000 + replicate)
            points = [
                (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
                for _ in range(60)
            ]
            result = estimate_control_functional_weights(points, integrand, score)
            errors.append(result.estimate - truth)
            plain.append(statistics.variance([integrand(p) for p in points]))
        mc_error = math.sqrt(statistics.fmean(plain) / 60)
        self.assertLess(root_mean_square(errors), 0.1 * mc_error)

    def test_weights_sum_to_one_and_report_stein_discrepancy(self) -> None:
        """The quadrature weights sum to one and integrate constants exactly."""
        draws = normal_draws(30, seed=8)
        result = estimate_control_functional_weights(
            draws, lambda v: 2.5, standard_score, length_scale="median"
        )
        self.assertAlmostEqual(math.fsum(result.weights), 1.0, places=10)
        self.assertAlmostEqual(result.estimate, 2.5, places=10)
        self.assertGreater(result.stein_discrepancy, 0.0)
        self.assertEqual(result.length_scale_rule, "median")

    def test_split_sample_result_and_counts(self) -> None:
        """Split-sample CF reports per-fold settings and covers the truth."""
        integrand, score = Counter(math.cos), Counter(standard_score)
        result = estimate_control_functional(
            normal_draws(40, seed=1), integrand, score, kernel="imq"
        )
        self.assertEqual((integrand.calls, score.calls), (40, 40))
        self.assertEqual(result.score_evaluations, 40)
        self.assertEqual(len(result.length_scales), 2)
        self.assertEqual(result.length_scale_rule, "cv")
        self.assertGreater(result.variance_reduction_factor, 1.0)
        self.assertLess(
            abs(result.estimate - math.exp(-0.5)), 4.0 * result.standard_error
        )

    def test_invalid_control_functional_inputs_raise(self) -> None:
        """Bad rules, regularizations, folds, and candidates are rejected."""
        draws = normal_draws(12, seed=3)
        with self.assertRaises(ValueError):
            estimate_control_functional(
                draws, math.cos, standard_score, length_scale="mode"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            estimate_control_functional(
                draws, math.cos, standard_score, regularization=0.5
            )
        with self.assertRaises(ValueError):
            estimate_control_functional(draws, math.cos, standard_score, folds=13)
        with self.assertRaises(ValueError):
            estimate_control_functional_weights(
                draws, math.cos, standard_score, length_scale_candidates=(1.0, -1.0)
            )
        with self.assertRaises(ValueError):
            estimate_control_functional_weights(
                draws, math.cos, standard_score, length_scale=-2.0
            )


class KernelSteinDiscrepancyTests(unittest.TestCase):
    """The IMQ KSD ranks a correct sample above a shifted one."""

    def test_correct_sample_has_smaller_discrepancy(self) -> None:
        """A shift by one unit raises the KSD and the U-statistic clearly."""
        draws = normal_draws(200, seed=5)
        correct = kernel_stein_discrepancy(draws, standard_score)
        shifted = kernel_stein_discrepancy([v + 1.0 for v in draws], standard_score)
        self.assertLess(correct.discrepancy, shifted.discrepancy)
        self.assertLess(3.0 * correct.discrepancy, shifted.discrepancy)
        self.assertLess(abs(correct.u_statistic), 0.1 * shifted.u_statistic)
        self.assertEqual(correct.score_evaluations, 200)
        self.assertEqual((correct.length_scale, correct.exponent), (1.0, -0.5))

    def test_discrepancy_works_for_vectors_and_validates(self) -> None:
        """Two-dimensional samples work; invalid settings raise."""
        points = gaussian_sample(50, seed=6)
        result = kernel_stein_discrepancy(points, gaussian_score, kernel="gaussian")
        self.assertGreaterEqual(result.v_statistic, 0.0)
        self.assertAlmostEqual(result.discrepancy ** 2, result.v_statistic, places=14)
        with self.assertRaises(ValueError):
            kernel_stein_discrepancy(points, gaussian_score, exponent=0.0)
        with self.assertRaises(TypeError):
            kernel_stein_discrepancy(["a", "b"], gaussian_score)


if __name__ == "__main__":
    unittest.main()
