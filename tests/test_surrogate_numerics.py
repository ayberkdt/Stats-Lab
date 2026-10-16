"""PCE by Householder QR and the one-factorization GP likelihood.

Oracles: the closed-form Hermite coefficients and lognormal moments of
``exp(a X)`` with ``X ~ N(0, 1)`` (``c_k = e^(a^2/2) a^k / sqrt(k!)``), and a
dense reference GP likelihood built with an explicit ``inverse_spd`` (a
different algorithm from the triangular-inverse implementation).
"""

from __future__ import annotations

import math
import unittest
import warnings

from probstat.distributions import Normal
from probstat.linalg import inverse_spd, log_det_spd
from probstat.modeling import JointInput, RandomInput
from probstat.sampling import latin_hypercube
from probstat.surrogates import (
    GaussianProcessConditioningWarning,
    GaussianProcessJitterWarning,
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
    kernel_value,
    polynomial_chaos_from_samples,
)
from probstat.surrogates import gaussian_process as gp_module


class PolynomialChaosQRTests(unittest.TestCase):
    """Defect 0.15: degree-20 Hermite fits by QR, rank deficiency reported."""

    def test_degree_twenty_hermite_fit_succeeds_for_every_seed(self) -> None:
        """40 seeds, 60 LHS points: all fits succeed; moments match closed forms."""
        inputs = JointInput.independent(RandomInput("x", Normal(0.0, 1.0)))
        a = 0.5
        mean = math.exp(0.5 * a * a)
        variance = math.exp(2.0 * a * a) - math.exp(a * a)
        standard = Normal()
        for seed in range(40):
            with self.subTest(seed=seed):
                design = latin_hypercube(60, 1, seed=seed)
                points = [(standard.quantile(p[0]),) for p in design.points]
                values = [math.exp(a * p[0]) for p in points]
                fit = polynomial_chaos_from_samples(inputs, points, values, degree=20)
                self.assertLess(abs(fit.mean - mean), 1e-6)
                self.assertLess(abs(fit.variance - variance), 1e-5)
                low = {
                    alpha[0]: c
                    for alpha, c in zip(fit.indices, fit.coefficients, strict=True)
                }
                for k in range(4):
                    exact = mean * a**k / math.sqrt(math.factorial(k))
                    self.assertLess(abs(low[k] - exact), 1e-4)

    def test_rank_deficiency_is_reported(self) -> None:
        """Five distinct inputs cannot identify nine coefficients."""
        inputs = JointInput.independent(RandomInput("x", Normal(0.0, 1.0)))
        points = [((i % 5) - 2.0,) for i in range(30)]
        values = [math.sin(p[0]) for p in points]
        with self.assertRaisesRegex(ValueError, "numerically dependent"):
            polynomial_chaos_from_samples(inputs, points, values, degree=8)


def dense_reference(points, targets, theta, kernel, nugget):
    """Return the GP log likelihood and gradient with an explicit inverse."""
    signal = math.exp(theta[0])
    scales = tuple(math.exp(t) for t in theta[1:])
    count = len(points)
    matrix = tuple(
        tuple(
            kernel_value(kernel, points[i], points[j], scales, signal)
            + (nugget if i == j else 0.0)
            for j in range(count)
        )
        for i in range(count)
    )
    inverse = inverse_spd(matrix)
    alpha = [
        math.fsum(inverse[i][j] * targets[j] for j in range(count))
        for i in range(count)
    ]
    value = (
        -0.5 * math.fsum(a * y for a, y in zip(alpha, targets, strict=True))
        - 0.5 * log_det_spd(matrix)
        - 0.5 * count * math.log(2.0 * math.pi)
    )
    weight = [
        [alpha[i] * alpha[j] - inverse[i][j] for j in range(count)]
        for i in range(count)
    ]
    gradient = []
    for index in range(len(theta)):
        derivative = [[0.0] * count for _ in range(count)]
        for i in range(count):
            for j in range(count):
                parts = [
                    ((a - b) / ell) ** 2
                    for a, b, ell in zip(points[i], points[j], scales, strict=True)
                ]
                r2 = sum(parts)
                r = math.sqrt(r2)
                if index == 0:
                    derivative[i][j] = kernel_value(
                        kernel, points[i], points[j], scales, signal
                    )
                    continue
                # d k / d log ell_a, derived by hand from k(r) and
                # dr/dlog ell_a = -(Delta_a / ell_a)^2 / r.
                if kernel == "squared_exponential":
                    factor = math.exp(-0.5 * r2)
                else:
                    factor = (
                        (5.0 / 3.0)
                        * (1.0 + math.sqrt(5.0) * r)
                        * math.exp(-math.sqrt(5.0) * r)
                    )
                derivative[i][j] = signal * factor * parts[index - 1]
        gradient.append(
            0.5
            * math.fsum(
                weight[i][j] * derivative[i][j]
                for i in range(count)
                for j in range(count)
            )
        )
    return value, gradient


class GaussianProcessNumericsTests(unittest.TestCase):
    """Defect P2: one factorization, adaptive nugget, conditioning warning."""

    POINTS = tuple((0.13 * i % 1.0, (0.37 * i * i) % 1.0) for i in range(1, 16))

    def targets(self) -> list[float]:
        """Return standardized targets of a smooth function."""
        values = [math.sin(3 * a) + b * b for a, b in self.POINTS]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        return [(v - mean) / sd for v in values]

    def test_matches_dense_explicit_inverse_reference(self) -> None:
        """Value to 1e-11 and gradient to 1e-7 against the dense reference."""
        targets = self.targets()
        for kernel in ("squared_exponential", "matern52"):
            theta = (0.2, math.log(0.4), math.log(0.6))
            with self.subTest(kernel=kernel):
                value, gradient = gaussian_process_log_marginal_likelihood(
                    self.POINTS, targets, theta, kernel=kernel, nugget=1e-6
                )
                ref_value, ref_gradient = dense_reference(
                    self.POINTS, targets, theta, kernel, 1e-6
                )
                self.assertAlmostEqual(
                    value, ref_value, delta=1e-11 * max(1, abs(ref_value))
                )
                for got, expected in zip(gradient, ref_gradient, strict=True):
                    self.assertAlmostEqual(
                        got, expected, delta=1e-7 * max(1, abs(expected))
                    )

    def test_one_factorization_per_point(self) -> None:
        """Value and gradient requests at one ``theta`` share a factorization."""
        calls = []
        original = gp_module.gaussian_process_log_marginal_likelihood

        def counted(*args, **kwargs):
            """Count likelihood computations."""
            calls.append(tuple(args[2]))
            return original(*args, **kwargs)

        values = [math.sin(3 * a) + b * b for a, b in self.POINTS]
        gp_module.gaussian_process_log_marginal_likelihood = counted
        try:
            fit = fit_gaussian_process(
                self.POINTS, values, seed=1, restarts=1, nugget=1e-6
            )
        finally:
            gp_module.gaussian_process_log_marginal_likelihood = original
        self.assertEqual(len(calls), fit.evaluations)
        self.assertEqual(len(calls), len(set(calls)))

    def test_adaptive_nugget_is_reported(self) -> None:
        """Duplicated inputs with a vanishing nugget: retried and warned."""
        points = ((0.0,), (0.0,), (0.5,), (1.0,), (1.0,))
        values = (0.0, 0.0, 0.4, 1.0, 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_gaussian_process(
                points, values, nugget=1e-300, seed=2, restarts=1
            )
        self.assertGreater(fit.nugget, 1e-300)
        self.assertTrue(any(w.category is GaussianProcessJitterWarning for w in caught))
        with self.assertRaises(ValueError):
            fit_gaussian_process(
                points, values, nugget=1e-300, seed=2, restarts=1, adaptive_nugget=False
            )

    def test_condition_number_and_warning(self) -> None:
        """The reported 1-norm condition number matches an explicit inverse."""
        points = tuple((0.1 * i,) for i in range(12))
        values = [math.sin(p[0]) for p in points]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_gaussian_process(
                points,
                values,
                kernel="squared_exponential",
                nugget=1e-12,
                optimize=False,
                initial_log_parameters=(0.0, math.log(2.0)),
            )
        signal = math.exp(fit.log_parameters[0])
        scales = (math.exp(fit.log_parameters[1]),)
        matrix = tuple(
            tuple(
                kernel_value("squared_exponential", a, b, scales, signal)
                + (1e-12 if a == b else 0.0)
                for b in points
            )
            for a in points
        )
        inverse = inverse_spd(matrix)
        reference = max(sum(abs(v) for v in row) for row in matrix) * max(
            sum(abs(v) for v in row) for row in inverse
        )
        self.assertGreater(fit.condition_number, 1e12)
        self.assertLess(abs(fit.condition_number / reference - 1.0), 1e-2)
        self.assertTrue(
            any(w.category is GaussianProcessConditioningWarning for w in caught)
        )


if __name__ == "__main__":
    unittest.main()
