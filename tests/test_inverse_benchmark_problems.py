"""Tests for the inverse benchmark catalog (roadmap task 19, L01-L06).

The checks are structural (shapes, exact spectra, adjoint identity, seed
separation, the inverse-crime flag) plus one solver run per case that only
asserts improvement over the zero estimate, never a pinned accuracy.
"""

from __future__ import annotations

import math
import unittest

from probstat.inverse import (
    LinearInverseProblem,
    krylov_regularization,
    l1_regularized_least_squares,
    linear_gaussian_posterior,
    spectral_diagnostics,
)
from probstat.inverse.benchmark_problems import (
    INVERSE_BENCHMARK_CATALOG,
    correlated_gaussian_benchmark,
    deconvolution_benchmark,
    diagonal_spectrum_benchmark,
    sparse_sensing_benchmark,
)
from probstat.inverse.iterative import expected_noise_norm
from probstat.linalg import adjoint_test


def distance(left: object, right: object) -> float:
    """Return the Euclidean distance of two vectors.

    References:
        [EQ-INV-BENCH-001].

    """
    return math.hypot(*(a - b for a, b in zip(left, right, strict=True)))  # type: ignore[call-overload]


class BenchmarkCatalogTests(unittest.TestCase):
    """Benchmark instances declare and honour their provenance [EQ-INV-BENCH-001]."""

    def test_catalog_lists_six_cases(self) -> None:
        """L01-L06 are present with methods and pathologies."""
        identifiers = [case.identifier for case in INVERSE_BENCHMARK_CATALOG]
        self.assertEqual(identifiers, ["L01", "L02", "L03", "L04", "L05", "L06"])
        for case in INVERSE_BENCHMARK_CATALOG:
            self.assertTrue(case.methods)
            self.assertTrue(case.pathologies)

    def test_diagonal_spectrum(self) -> None:
        """L01 has the declared spectrum, nullity and a seed-stable truth."""
        case = diagonal_spectrum_benchmark(12, decay="exponential", rate=0.5, nullity=2)
        diagnostics = spectral_diagnostics(case.operator)
        assert case.singular_values is not None
        self.assertEqual(diagnostics.singular_values, case.singular_values)
        self.assertEqual(diagnostics.nullspace_dimension, 2)
        other = diagonal_spectrum_benchmark(
            12, decay="exponential", rate=0.5, nullity=2, seed=99
        )
        self.assertEqual(case.truth, other.truth)
        self.assertNotEqual(case.data, other.data)
        self.assertAlmostEqual(
            case.realized_noise_norm, distance(case.data, case.exact_data), places=12
        )
        with self.assertRaises(ValueError):
            diagonal_spectrum_benchmark(4, nullity=4)

    def test_deconvolution_inverse_crime_flag(self) -> None:
        """Refinement 1 is an inverse crime; refinement 3 has a discretization error."""
        crime = deconvolution_benchmark(32, refinement=1)
        self.assertTrue(crime.same_discretization)
        self.assertLess(
            distance(crime.exact_data, crime.operator.matvec(crime.truth)), 1e-12
        )
        honest = deconvolution_benchmark(32, refinement=3)
        self.assertFalse(honest.same_discretization)
        gap = distance(honest.exact_data, honest.operator.matvec(honest.truth))
        self.assertGreater(gap, 1e-6)
        self.assertLess(gap, 0.2 * math.hypot(*honest.exact_data))
        self.assertTrue(
            adjoint_test(honest.operator, list(honest.truth), list(honest.data)).passed
        )
        for boundary in ("periodic", "reflect"):
            case = deconvolution_benchmark(16, boundary=boundary, truth="smooth")
            self.assertEqual(case.operator.shape, (16, 16))

    def test_deconvolution_early_stopping_improves(self) -> None:
        """Early-stopped LSQR on independent data beats the zero estimate."""
        case = deconvolution_benchmark(48, refinement=2, relative_noise=0.01)
        problem = LinearInverseProblem.create(
            case.operator, case.data, noise=case.noise
        )
        result = krylov_regularization(
            problem, noise_norm=expected_noise_norm(48), truth=case.truth
        )
        self.assertEqual(result.stop_reason, "discrepancy")
        self.assertLess(
            distance(result.estimate, case.truth), 0.7 * math.hypot(*case.truth)
        )

    def test_sparse_sensing(self) -> None:
        """L05 is underdetermined with the declared support; the lasso helps."""
        case = sparse_sensing_benchmark(20, 50, sparsity=3, relative_noise=1e-3)
        self.assertEqual(case.operator.shape, (20, 50))
        self.assertEqual(sum(1 for v in case.truth if v != 0.0), 3)
        result = l1_regularized_least_squares(
            case.operator, case.data, 0.01, max_iterations=4000
        )
        self.assertLess(
            distance(result.solution, case.truth), 0.5 * math.hypot(*case.truth)
        )
        with self.assertRaises(ValueError):
            sparse_sensing_benchmark(10, 8)

    def test_correlated_gaussian(self) -> None:
        """L06 supplies an SPD prior; its exact posterior uses only solver inputs."""
        case = correlated_gaussian_benchmark(16, 8)
        assert case.prior_covariance is not None
        posterior = linear_gaussian_posterior(
            case.operator,
            case.data,
            noise=case.noise,
            prior_covariance=case.prior_covariance,
        )
        self.assertEqual(len(posterior.mean), 16)
        self.assertTrue(math.isfinite(posterior.log_evidence))
        again = correlated_gaussian_benchmark(16, 8, seed=5)
        self.assertNotEqual(case.truth, again.truth)
        with self.assertRaises(ValueError):
            correlated_gaussian_benchmark(6, 5)


if __name__ == "__main__":
    unittest.main()
