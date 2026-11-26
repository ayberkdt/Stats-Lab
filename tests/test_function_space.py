"""Tests for function-space bases, spectral priors and credible bands."""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse.function_space import (
    HatBasis,
    PiecewiseConstantBasis,
    SineBasis,
    SpectralGaussianPrior,
    credible_bands,
)
from probstat.inverse.linear_gaussian import linear_gaussian_posterior
from probstat.linalg import DenseOperator, ScaledIdentityCovariance


def _midpoint_gram(basis: object, panels: int = 4000) -> list[list[float]]:
    """Return the Gram matrix by fine midpoint quadrature (independent oracle)."""
    length = basis.length  # type: ignore[attr-defined]
    n = basis.dimension  # type: ignore[attr-defined]
    points = [(i + 0.5) * length / panels for i in range(panels)]
    phi = basis.evaluation_operator(points).matrix  # type: ignore[attr-defined]
    w = length / panels
    return [
        [w * sum(phi[q][i] * phi[q][j] for q in range(panels)) for j in range(n)]
        for i in range(n)
    ]


class BasisTests(unittest.TestCase):
    """Mass matrices, evaluation and the integral operator."""

    def test_sine_basis_orthonormal(self) -> None:
        """Identity mass matrix agrees with independent quadrature."""
        basis = SineBasis(5, 2.0)
        gram = _midpoint_gram(basis, 2000)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(gram[i][j], 1.0 if i == j else 0.0, delta=1e-6)
                self.assertEqual(basis.mass_matrix()[i][j], 1.0 if i == j else 0.0)

    def test_hat_mass_matrix_closed_form_and_quadrature(self) -> None:
        """Mass is h/6 [1 4 1] and matches quadrature."""
        basis = HatBasis(5, 3.0)
        h = 3.0 / 6.0
        mass = basis.mass_matrix()
        gram = _midpoint_gram(basis, 3000)
        for i in range(5):
            for j in range(5):
                expected = {0: 4 * h / 6, 1: h / 6}.get(abs(i - j), 0.0)
                self.assertAlmostEqual(mass[i][j], expected, places=14)
                self.assertAlmostEqual(gram[i][j], expected, delta=1e-5)

    def test_hat_is_nodal_and_vanishes_at_ends(self) -> None:
        """Hat coefficients are nodal values; the ends are zero."""
        basis = HatBasis(3, 1.0)
        x = (2.0, -1.0, 4.0)
        values = basis.evaluate(x, [0.0, 0.25, 0.5, 0.75, 1.0, 0.375])
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[4], 0.0)
        for got, want in zip(values[1:4], x, strict=True):
            self.assertAlmostEqual(got, want, places=14)
        self.assertAlmostEqual(values[5], 0.5 * (2.0 - 1.0), places=14)

    def test_piecewise_constant_mass_and_evaluation(self) -> None:
        """Mass is h I; the last cell includes the right end."""
        basis = PiecewiseConstantBasis(4, 2.0)
        mass = basis.mass_matrix()
        for i in range(4):
            for j in range(4):
                self.assertEqual(mass[i][j], 0.5 if i == j else 0.0)
        gram = _midpoint_gram(basis, 2000)
        self.assertAlmostEqual(gram[2][2], 0.5, delta=1e-9)
        values = basis.evaluate((1.0, 2.0, 3.0, 4.0), [0.0, 0.49, 0.5, 2.0])
        self.assertEqual(values, (1.0, 1.0, 2.0, 4.0))

    def test_integral_operator_polynomial_kernels_piecewise_constant(self) -> None:
        """Aligned panels integrate K phi_j exactly (closed form)."""
        n, length = 4, 2.0
        basis = PiecewiseConstantBasis(n, length)
        kernels = [lambda s: 1.0, lambda s: s, lambda s: s**3 - 2.0 * s]
        op = basis.integral_operator(kernels, 8)
        h = length / n

        def antiderivative(i: int, a: float) -> float:
            return (a, a * a / 2, a**4 / 4 - a * a)[i]

        for i in range(3):
            for j in range(n):
                exact = antiderivative(i, (j + 1) * h) - antiderivative(i, j * h)
                self.assertAlmostEqual(op.matrix[i][j], exact, places=13)

    def test_integral_operator_linear_basis_and_misaligned_grid(self) -> None:
        """Hat integrals: K=1 gives h; a misaligned grid converges."""
        n, length = 3, 1.0
        basis = HatBasis(n, length)
        h = length / (n + 1)
        aligned = basis.integral_operator([lambda s: 1.0, lambda s: s * s], 4)
        for j in range(n):
            self.assertAlmostEqual(aligned.matrix[0][j], h, places=14)
            # int s^2 hat_j = h (s_j^2 + h^2 / 6)
            sj = (j + 1) * h
            self.assertAlmostEqual(aligned.matrix[1][j], h * (sj * sj + h * h / 6), 13)
        errors = []
        for panels in (3, 30, 300):
            op = basis.integral_operator([lambda s: 1.0], panels)
            errors.append(max(abs(op.matrix[0][j] - h) for j in range(n)))
        self.assertLess(errors[2], errors[0])
        self.assertLess(errors[2], 1e-3)

    def test_integral_operator_sine_closed_form(self) -> None:
        """int_0^L phi_k ds = sqrt(2L) (1 - cos(k pi)) / (k pi)."""
        length = 1.5
        op = SineBasis(6, length).integral_operator([lambda s: 1.0], 20)
        for k in range(1, 7):
            exact = (
                math.sqrt(2.0 * length) * (1.0 - math.cos(k * math.pi)) / (k * math.pi)
            )
            self.assertAlmostEqual(op.matrix[0][k - 1], exact, places=12)

    def test_explicit_breakpoints_and_adjoint_shape(self) -> None:
        """Breakpoint grids work and invalid grids raise."""
        basis = SineBasis(3, 1.0)
        op = basis.integral_operator([lambda s: s], [0.0, 0.3, 1.0])
        self.assertEqual(op.shape, (1, 3))
        for bad in ([0.0, 0.5], [0.0, 0.7, 0.6, 1.0], [0.1, 1.0], 0):
            with self.assertRaises(ValueError):
                basis.integral_operator([lambda s: s], bad)
        with self.assertRaises(TypeError):
            basis.integral_operator([lambda s: s], "abc")  # type: ignore[arg-type]

    def test_invalid_inputs_raise(self) -> None:
        """Bad constructor, point and coefficient arguments raise."""
        with self.assertRaises(ValueError):
            SineBasis(0, 1.0)
        with self.assertRaises(ValueError):
            HatBasis(3, -1.0)
        with self.assertRaises(TypeError):
            SineBasis(2.5, 1.0)  # type: ignore[arg-type]
        basis = SineBasis(3, 1.0)
        with self.assertRaises(ValueError):
            basis.evaluate((1.0, 2.0), [0.5])
        with self.assertRaises(ValueError):
            basis.evaluate((1.0, 2.0, 3.0), [1.5])
        with self.assertRaises(ValueError):
            basis.evaluation_operator([])
        with self.assertRaises(ValueError):
            basis.integral_operator([], 4)
        with self.assertRaises(TypeError):
            basis.integral_operator([3.0], 4)  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            basis.integral_operator([lambda s: math.inf], 4)


class PriorTests(unittest.TestCase):
    """Spectral Gaussian prior."""

    def setUp(self) -> None:
        """Create a reference prior."""
        self.prior = SpectralGaussianPrior(1.0, 2.0, 1.5, 1.5, 64)

    def test_coefficient_variances_independent_of_truncation(self) -> None:
        """First 16 variances agree for 16 and 64 modes and match the formula."""
        small = self.prior.coefficient_covariance(16).standard_deviations
        large = self.prior.coefficient_covariance(64).standard_deviations
        self.assertEqual(small, large[:16])
        for k in range(1, 17):
            expected = 4.0 * (1.5**2 + (k * math.pi) ** 2) ** -1.5
            self.assertAlmostEqual(small[k - 1] ** 2, expected, places=14)

    def test_trace_and_tail_bound(self) -> None:
        """Partial traces increase and the tail bound dominates the true tail."""
        prior = SpectralGaussianPrior(1.0, 1.0, 0.5, 2.0, 8)
        reference = prior.trace(20000)
        for n in (4, 8, 16, 32):
            tail = reference - prior.trace(n)
            self.assertLessEqual(tail, prior.remaining_trace_bound(n))
            self.assertGreater(tail, 0.0)

    def test_pointwise_variance_converges(self) -> None:
        """Differences shrink with modes and respect the tail bound."""
        point = [0.3]
        values = [self.prior.pointwise_variance(point, m)[0] for m in (4, 16, 64, 256)]
        diffs = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
        self.assertTrue(all(d2 < d1 for d1, d2 in zip(diffs, diffs[1:], strict=False)))
        for m, a, b in zip((4, 16, 64), values, values[1:], strict=False):
            self.assertLessEqual(abs(b - a), 2.0 * self.prior.remaining_trace_bound(m))
        # Independent direct sum.
        direct = sum(
            self.prior.eigenvalue(k) * 2.0 * math.sin(k * math.pi * 0.3) ** 2
            for k in range(1, 65)
        )
        self.assertAlmostEqual(values[2], direct, places=13)

    def test_smoothness_not_trace_class_raises(self) -> None:
        """Smoothness <= 1/2 is rejected; other bad parameters raise."""
        with self.assertRaises(ValueError):
            SpectralGaussianPrior(1.0, 1.0, 1.0, 0.5, 10)
        with self.assertRaises(ValueError):
            SpectralGaussianPrior(1.0, 1.0, 1.0, 0.3, 10)
        with self.assertRaises(ValueError):
            SpectralGaussianPrior(1.0, 0.0, 1.0, 1.5, 10)
        with self.assertRaises(ValueError):
            SpectralGaussianPrior(1.0, 1.0, -1.0, 1.5, 10)
        with self.assertRaises(ValueError):
            SpectralGaussianPrior(1.0, 1.0, 1.0, 1.5, 0)
        with self.assertRaises(TypeError):
            SpectralGaussianPrior(1.0, 1.0, 1.0, 1.5, 2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.prior.sample(0, 1, [0.5])
        with self.assertRaises(TypeError):
            self.prior.sample(3, "x", [0.5])  # type: ignore[arg-type]

    def test_kl_sample_covariance_matches_theory(self) -> None:
        """Sample variance and covariance at two points within 5 MCSE."""
        points = [0.3, 0.7]
        count = 3000
        draws = self.prior.sample(count, 20240601, points)
        basis = SineBasis(64, 1.0)
        phi = basis.evaluation_operator(points).matrix
        lam = [self.prior.eigenvalue(k) for k in range(1, 65)]

        def cov(a: int, b: int) -> float:
            return sum(lk * phi[a][k] * phi[b][k] for k, lk in enumerate(lam))

        v1, v2, c12 = cov(0, 0), cov(1, 1), cov(0, 1)
        m1 = sum(d[0] for d in draws) / count
        m2 = sum(d[1] for d in draws) / count
        s11 = sum((d[0] - m1) ** 2 for d in draws) / (count - 1)
        s22 = sum((d[1] - m2) ** 2 for d in draws) / (count - 1)
        s12 = sum((d[0] - m1) * (d[1] - m2) for d in draws) / (count - 1)
        self.assertLess(abs(s11 - v1), 5 * v1 * math.sqrt(2.0 / count))
        self.assertLess(abs(s22 - v2), 5 * v2 * math.sqrt(2.0 / count))
        self.assertLess(abs(s12 - c12), 5 * math.sqrt((v1 * v2 + c12**2) / count))
        # Reproducible for a fixed seed.
        self.assertEqual(draws, self.prior.sample(count, 20240601, points))


class FunctionalInverseDemo(unittest.TestCase):
    """Illustration of a functional inverse problem (not a coverage study)."""

    def test_gaussian_kernel_inversion(self) -> None:
        """32-mode inversion of data generated from a finer 128-mode truth."""
        length = 1.0
        prior_fine = SpectralGaussianPrior(length, 10.0, 2.0, 2.0, 128)
        centres = [0.05 + 0.9 * i / 19 for i in range(20)]
        width = 0.05
        kernels = [
            (lambda s, c=c: math.exp(-0.5 * ((s - c) / width) ** 2)) for c in centres
        ]
        # Truth: one fixed draw of the 128-mode prior (finer than inversion).
        rng = random.Random(7)
        true_coeff = [
            math.sqrt(prior_fine.eigenvalue(k)) * rng.gauss(0.0, 1.0)
            for k in range(1, 129)
        ]
        fine = SineBasis(128, length)
        # Data from a finer quadrature grid (300 panels) than the inversion's (100).
        clean = DenseOperator(fine.integral_operator(kernels, 300).matrix).matvec(
            true_coeff
        )
        sigma = 0.005
        noise_rng = random.Random(11)
        y = tuple(v + sigma * noise_rng.gauss(0.0, 1.0) for v in clean)

        n = 32
        prior = SpectralGaussianPrior(length, 10.0, 2.0, 2.0, n)
        a_op = SineBasis(n, length).integral_operator(kernels, 100)
        post = linear_gaussian_posterior(
            a_op,
            y,
            noise=ScaledIdentityCovariance(len(y), sigma**2),
            prior_covariance=prior.coefficient_covariance(),
            method="parameter_space",
        )
        prior_var = [prior.eigenvalue(k) for k in range(1, n + 1)]
        post_var = post.marginal_variances()
        for k in range(3):
            self.assertLess(
                abs(post.mean[k] - true_coeff[k]), 0.2 * math.sqrt(prior_var[k])
            )
        for k in range(n - 4, n):
            self.assertGreater(post_var[k] / prior_var[k], 0.9)
        self.assertLess(post_var[0] / prior_var[0], 0.5)

        # Integrated QoI int_a^b f ds, a=0.2, b=0.6, closed form per mode.
        a, b = 0.2, 0.6

        def qoi_weight(k: int) -> float:
            return (
                math.sqrt(2.0 / length)
                * (length / (k * math.pi))
                * (
                    math.cos(k * math.pi * a / length)
                    - math.cos(k * math.pi * b / length)
                )
            )

        q = tuple(qoi_weight(k) for k in range(1, n + 1))
        truth = sum(qoi_weight(k) * true_coeff[k - 1] for k in range(1, 129))
        estimate = sum(w * m for w, m in zip(q, post.mean, strict=True))
        variance = sum(w * c for w, c in zip(q, post.covariance_action(q), strict=True))
        self.assertGreater(variance, 0.0)
        self.assertLess(abs(estimate - truth), 4.0 * math.sqrt(variance))


class CredibleBandTests(unittest.TestCase):
    """Pointwise and simultaneous bands."""

    def test_simultaneous_wider_than_pointwise(self) -> None:
        """Simultaneous band contains the pointwise one for Gaussian draws."""
        prior = SpectralGaussianPrior(1.0, 1.0, 1.0, 1.5, 32)
        grid = [(i + 1) / 41 for i in range(40)]
        samples = prior.sample(2000, 5, grid)
        bands = credible_bands(samples, 0.9)
        self.assertGreater(bands.scaling_factor, 1.645)
        low, high = bands.scaling_factor_interval
        self.assertLessEqual(low, bands.scaling_factor)
        self.assertGreaterEqual(high, bands.scaling_factor)
        for i in range(40):
            self.assertGreaterEqual(
                bands.simultaneous_upper[i] - bands.simultaneous_lower[i],
                bands.upper[i] - bands.lower[i],
            )
        # At least `level` of the samples lie inside the simultaneous band.
        inside = sum(
            all(
                lo <= v <= hi
                for v, lo, hi in zip(
                    s, bands.simultaneous_lower, bands.simultaneous_upper, strict=True
                )
            )
            for s in samples
        )
        self.assertGreaterEqual(inside / 2000, 0.9 - 1e-12)

    def test_invalid_inputs_raise(self) -> None:
        """Bad level, shape and degenerate points raise."""
        good = [[0.0, 1.0], [1.0, 2.0], [2.0, 0.5]]
        for level in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                credible_bands(good, level)
        with self.assertRaises(ValueError):
            credible_bands([[1.0, 2.0]], 0.9)
        with self.assertRaises(ValueError):
            credible_bands([[1.0, 2.0], [1.0]], 0.9)
        with self.assertRaises(ValueError):
            credible_bands([[0.0, 1.0], [0.0, 2.0]], 0.9)
        with self.assertRaises(ValueError):
            credible_bands([[0.0, math.nan], [1.0, 2.0]], 0.9)


if __name__ == "__main__":
    unittest.main()
