"""Tests for spectral diagnostics, iterative regularization and parameter selection.

Roadmap tasks 04, 06 and 07. Oracles: diagonal operators with closed-form
singular values, Landweber iterates and filter factors; the closed-form SVD
of the backward heat problem; the legacy ``tikhonov`` (stacked QR) for the
GSVD and the hybrid method; the explicit influence-matrix formulas of GCV
and UPRE for a diagonal problem; and the Gaussian marginal density for the
evidence. Synthetic truths only enter error histories, never the rules.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse import backward_heat_problem, second_derivative_problem, tikhonov
from probstat.inverse.iterative import (
    expected_noise_norm,
    hybrid_lsqr,
    krylov_regularization,
    landweber,
)
from probstat.inverse.linear_problem import LinearInverseProblem, solve_regularized
from probstat.inverse.parameter_selection import (
    grouped_cross_validation,
    logarithmic_grid,
    regularization_path,
    select_regularization,
)
from probstat.inverse.spectral import (
    generalized_svd,
    gsvd_tikhonov,
    spectral_diagnostics,
)
from probstat.linalg import (
    DenseCovariance,
    DiagonalCovariance,
    ScaledIdentityCovariance,
    matvec,
)

SEED = 424242


def diagonal_matrix(values: list[float]) -> list[list[float]]:
    """Return ``diag(values)``.

    References:
        [EQ-INV-SPEC-001].

    """
    return [
        [v if i == j else 0.0 for j in range(len(values))] for i, v in enumerate(values)
    ]


def max_diff(left: object, right: object) -> float:
    """Return the largest absolute entrywise difference.

    References:
        [EQ-INV-SPEC-001].

    """
    return max(abs(a - b) for a, b in zip(left, right, strict=True))  # type: ignore[call-overload]


def noisy(
    data: tuple[float, ...], level: float, seed: int
) -> tuple[list[float], float]:
    """Add iid Gaussian noise with standard deviation ``level ||b|| / sqrt(m)``.

    References:
        [EQ-INV-ITER-002].

    """
    generator = random.Random(seed)
    sigma = level * math.hypot(*data) / math.sqrt(len(data))
    return [b + sigma * generator.gauss(0.0, 1.0) for b in data], sigma


class SpectralDiagnosticsTests(unittest.TestCase):
    """Rank, nullspace and Picard quantities [EQ-INV-SPEC-001]."""

    def test_graded_diagonal_spectrum(self) -> None:
        """``diag(1, 1e-5, 1e-12, 0)``: exact values, rank 3, one null direction."""
        diagnostics = spectral_diagnostics(
            diagonal_matrix([1.0, 1e-5, 1e-12, 0.0]), data=[1.0, 1.0, 1.0, 1.0]
        )
        self.assertEqual(diagnostics.singular_values, (1.0, 1e-5, 1e-12, 0.0))
        self.assertEqual(diagnostics.rank, 3)
        self.assertEqual(diagnostics.nullspace_dimension, 1)
        self.assertEqual(diagnostics.condition_number, math.inf)
        self.assertAlmostEqual(diagnostics.effective_condition_number, 1e12, delta=1.0)
        assert diagnostics.solution_coefficients is not None
        self.assertEqual(diagnostics.solution_coefficients[3], math.inf)
        self.assertAlmostEqual(diagnostics.solution_coefficients[1], 1e5, delta=1e-6)
        self.assertEqual(diagnostics.detectable(1.0), (True, False, False, False))
        coarse = spectral_diagnostics(
            diagonal_matrix([1.0, 1e-5, 1e-12, 0.0]), rtol=1e-8
        )
        self.assertEqual(coarse.rank, 2)

    def test_wide_operator_full_nullspace(self) -> None:
        """A 3x6 operator has a completed 3-dimensional nullspace with ``A N' = 0``."""
        generator = random.Random(SEED)
        a = [[generator.gauss(0.0, 1.0) for _ in range(6)] for _ in range(3)]
        diagnostics = spectral_diagnostics(a)
        self.assertEqual(diagnostics.rank, 3)
        self.assertEqual(len(diagnostics.nullspace_basis), 3)
        for vector in diagnostics.nullspace_basis:
            self.assertLess(max(abs(v) for v in matvec(a, vector)), 1e-12)
            self.assertAlmostEqual(math.hypot(*vector), 1.0, places=12)
        x = [generator.gauss(0.0, 1.0) for _ in range(6)]
        projected = diagnostics.null_projection(x)
        twice = diagnostics.null_projection(projected)
        self.assertLess(max_diff(projected, twice), 1e-12)
        self.assertLess(max(abs(v) for v in matvec(a, projected)), 1e-12)

    def test_backward_heat_closed_form_and_whitening(self) -> None:
        """The whitened spectrum scales by ``1/s``; it matches the closed form."""
        problem = backward_heat_problem(12)
        assert problem.singular_values is not None
        diagnostics = spectral_diagnostics(
            problem.operator, noise=ScaledIdentityCovariance(12, 0.25)
        )
        expected = tuple(2.0 * s for s in problem.singular_values)
        self.assertLess(
            max_diff(diagnostics.singular_values, expected), 1e-12 * expected[0]
        )
        assert diagnostics.out_of_range_norm is None
        consistent = spectral_diagnostics(problem.operator, data=problem.exact_data)
        assert consistent.out_of_range_norm is not None
        self.assertLess(consistent.out_of_range_norm, 1e-12)


class GeneralizedSVDTests(unittest.TestCase):
    """GSVD identities and general-form Tikhonov [EQ-INV-SPEC-002]."""

    def test_gsvd_identities_and_tikhonov(self) -> None:
        """``A X = U C``, ``L X = Z S``, ``c^2 + s^2 = 1`` and stacked-QR agreement."""
        generator = random.Random(SEED + 1)
        a = [[generator.gauss(0.0, 1.0) for _ in range(5)] for _ in range(8)]
        penalty = [
            [(-1.0 if j == i else 1.0 if j == i + 1 else 0.0) for j in range(5)]
            for i in range(4)
        ]
        decomposition = generalized_svd(a, penalty)
        for c, s in zip(decomposition.c, decomposition.s, strict=True):
            self.assertAlmostEqual(c * c + s * s, 1.0, places=12)
        x = decomposition.x
        for i in range(8):
            for j in range(5):
                ax = math.fsum(a[i][k] * x[k][j] for k in range(5))
                self.assertAlmostEqual(
                    ax, decomposition.u[i][j] * decomposition.c[j], places=12
                )
        for i in range(4):
            for j in range(5):
                lx = math.fsum(penalty[i][k] * x[k][j] for k in range(5))
                self.assertAlmostEqual(
                    lx, decomposition.z[i][j] * decomposition.s[j], places=12
                )
        self.assertEqual(
            sum(1 for s in decomposition.s if s < 1e-12), 1
        )  # constant mode
        b = [generator.gauss(0.0, 1.0) for _ in range(8)]
        for lam in (1e-4, 1e-2, 1.0, 1e2):
            expected = tikhonov(a, b, lam, penalty=penalty).solution
            self.assertLess(
                max_diff(gsvd_tikhonov(decomposition, b, lam), expected), 1e-10
            )

    def test_common_nullspace_raises(self) -> None:
        """A shared null direction of ``A`` and ``L`` is refused."""
        with self.assertRaises(ValueError):
            generalized_svd([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]], [[1.0, 1.0]])
        with self.assertRaises(ValueError):
            generalized_svd([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])


class IterativeRegularizationTests(unittest.TestCase):
    """Landweber, early-stopped Krylov and hybrid LSQR [EQ-INV-ITER-001..003]."""

    def test_landweber_filter_factors(self) -> None:
        """On ``diag(sigma)``: ``x_k = (1 - (1 - w s^2)^k) b / s`` exactly."""
        sigma = [2.0, 0.5, 0.1]
        b = [1.0, -1.0, 0.5]
        problem = LinearInverseProblem.create(diagonal_matrix(sigma), b)
        omega = 0.2
        result = landweber(problem, step=omega, max_iterations=7)
        expected = [
            (1.0 - (1.0 - omega * s * s) ** 7) * bi / s
            for s, bi in zip(sigma, b, strict=True)
        ]
        self.assertLess(max_diff(result.estimate, expected), 1e-13)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertFalse(result.converged)
        # Seven iterations plus one forward application for W (y - A x_ref).
        self.assertEqual(result.counts.forward, 8)
        self.assertEqual(result.counts.transpose, 7)
        bound = landweber(problem, operator_norm_bound=2.0, max_iterations=3)
        self.assertEqual(bound.iterations, 3)
        with self.assertRaises(ValueError):
            landweber(problem)
        with self.assertRaises(ValueError):
            landweber(problem, step=0.1, operator_norm_bound=2.0)

    def test_semi_convergence_and_discrepancy_stop(self) -> None:
        """Error is not monotone in the iteration while the misfit decreases."""
        base = second_derivative_problem(32)
        y, sigma = noisy(base.exact_data, 0.01, SEED + 2)
        problem = LinearInverseProblem.create(
            base.operator, y, noise=ScaledIdentityCovariance(32, sigma * sigma)
        )
        delta = expected_noise_norm(32)
        long_run = krylov_regularization(
            problem, method="cgls", max_iterations=30, truth=base.exact_solution
        )
        assert long_run.error_history is not None
        assert long_run.oracle_iteration is not None
        self.assertGreater(
            long_run.error_history[-1], 1.5 * min(long_run.error_history)
        )
        misfits = long_run.misfit_history
        self.assertTrue(
            all(b <= a * (1 + 1e-9) for a, b in zip(misfits, misfits[1:], strict=False))
        )
        for method in ("cgls", "lsqr", "lsmr"):
            result = krylov_regularization(
                problem, method=method, noise_norm=delta, truth=base.exact_solution
            )
            with self.subTest(method=method):
                self.assertEqual(result.stop_reason, "discrepancy")
                self.assertLessEqual(
                    result.misfit_history[-1], 1.01 * delta * (1 + 1e-12)
                )
                self.assertGreater(result.misfit_history[-2], 1.01 * delta)
                assert result.error_history is not None
                # The discrepancy iterate is far better than the over-fitted one.
                self.assertLess(result.error_history[-1], long_run.error_history[-1])

    def test_cgls_lsqr_iterates_agree(self) -> None:
        """CGLS and LSQR produce the same Krylov iterates in exact arithmetic."""
        base = second_derivative_problem(16)
        problem = LinearInverseProblem.create(base.operator, base.exact_data)
        a = krylov_regularization(problem, method="cgls", max_iterations=4)
        b = krylov_regularization(problem, method="lsqr", max_iterations=4)
        self.assertLess(
            max_diff(a.estimate, b.estimate), 1e-9 * max(abs(v) for v in a.estimate)
        )

    def test_hybrid_fixed_equals_tikhonov(self) -> None:
        """With a fixed ``lambda`` and enough steps, hybrid LSQR equals Tikhonov."""
        base = second_derivative_problem(16)
        y, sigma = noisy(base.exact_data, 0.01, SEED + 3)
        problem = LinearInverseProblem.create(
            base.operator, y, noise=ScaledIdentityCovariance(16, sigma * sigma)
        )
        lam = 0.5
        hybrid = hybrid_lsqr(
            problem,
            rule="fixed",
            regularization=lam,
            max_iterations=16,
            stabilization_tolerance=1e-15,
        )
        exact = solve_regularized(problem, lam)
        self.assertLess(
            max_diff(hybrid.estimate, exact.estimate),
            1e-7 * max(abs(v) for v in exact.estimate),
        )
        # Projected residual equals the true whitened misfit under orthogonality.
        self.assertAlmostEqual(
            hybrid.misfit_history[-1], problem.misfit(hybrid.estimate), places=8
        )

    def test_hybrid_discrepancy_meets_target(self) -> None:
        """The projected discrepancy rule ends near the noise level."""
        base = second_derivative_problem(32)
        y, sigma = noisy(base.exact_data, 0.01, SEED + 4)
        problem = LinearInverseProblem.create(
            base.operator, y, noise=ScaledIdentityCovariance(32, sigma * sigma)
        )
        delta = expected_noise_norm(32)
        result = hybrid_lsqr(
            problem, rule="discrepancy", noise_norm=delta, truth=base.exact_solution
        )
        self.assertIn(
            result.stop_reason, ("stabilized", "max_iterations", "breakdown_exact")
        )
        self.assertLessEqual(problem.misfit(result.estimate), 1.01 * delta * 1.0001)
        assert result.regularization_history is not None
        self.assertGreater(result.regularization_history[-1], 0.0)
        with self.assertRaises(ValueError):
            hybrid_lsqr(problem, rule="discrepancy")
        with self.assertRaises(ValueError):
            hybrid_lsqr(
                LinearInverseProblem.create(base.operator, y, penalty=[[1.0] * 32]),
                rule="gcv",
            )


class ParameterSelectionTests(unittest.TestCase):
    """GCV, UPRE, discrepancy, L-curve, evidence and CV [EQ-INV-SEL-001]."""

    def setUp(self) -> None:
        """Build a diagonal problem whose criteria have closed forms.

        References:
            [EQ-INV-SEL-001].

        """
        self.sigma = [3.0, 1.0, 0.3, 0.1, 0.03]
        self.y = [1.0, 0.5, -0.4, 0.2, 0.1]
        self.problem = LinearInverseProblem.create(diagonal_matrix(self.sigma), self.y)
        self.grid = logarithmic_grid(1e-3, 1e2, 15)

    def test_closed_form_criteria(self) -> None:
        """Residual, df, GCV and UPRE from the filter factors ``s^2 / (s^2 + l^2)``."""
        path = regularization_path(self.problem, self.grid)
        for i, lam in enumerate(self.grid):
            filters = [s * s / (s * s + lam * lam) for s in self.sigma]
            residual = math.hypot(
                *((1.0 - f) * y for f, y in zip(filters, self.y, strict=True))
            )
            df = math.fsum(filters)
            self.assertAlmostEqual(path.misfits[i], residual, places=12)
            self.assertAlmostEqual(path.degrees_of_freedom[i], df, places=12)
            self.assertAlmostEqual(
                path.gcv[i], 5 * residual**2 / (5 - df) ** 2, places=10
            )
            self.assertAlmostEqual(path.upre[i], residual**2 + 2 * df - 5, places=10)

    def test_evidence_matches_gaussian_density(self) -> None:
        """``y ~ N(0, Gamma + A C0 A')`` with ``C0 = I / lambda^2`` (diagonal)."""
        variances = [0.5, 1.0, 2.0, 0.25, 1.5]
        problem = LinearInverseProblem.create(
            diagonal_matrix(self.sigma), self.y, noise=DiagonalCovariance(variances)
        )
        path = regularization_path(problem, self.grid)
        assert path.log_evidence is not None
        for i, lam in enumerate(self.grid):
            total = 0.0
            for s, v, y in zip(self.sigma, variances, self.y, strict=True):
                var = v + s * s / (lam * lam)
                total += -0.5 * (y * y / var + math.log(2.0 * math.pi * var))
            self.assertAlmostEqual(path.log_evidence[i], total, places=10)
        selection = select_regularization(path, "evidence")
        self.assertEqual(selection.status, "interior")
        wide = LinearInverseProblem.create(
            diagonal_matrix(self.sigma), self.y, penalty=[[1.0, -1.0, 0.0, 0.0, 0.0]]
        )
        self.assertIsNone(regularization_path(wide, self.grid).log_evidence)

    def test_upre_is_unbiased_for_prediction_risk(self) -> None:
        """Mean UPRE matches the mean prediction loss within Monte Carlo error."""
        truth = [0.3, -0.2, 0.5, 0.1, -0.4]
        lam = 0.2
        a = diagonal_matrix(self.sigma)
        exact = [s * t for s, t in zip(self.sigma, truth, strict=True)]
        generator = random.Random(SEED + 5)
        upre_values, losses = [], []
        for _ in range(400):
            y = [e + generator.gauss(0.0, 1.0) for e in exact]
            path = regularization_path(
                LinearInverseProblem.create(a, y), (lam / 2, lam, 2 * lam)
            )
            x = path.estimates[1]
            losses.append(
                math.fsum(
                    (s * xi - e) ** 2
                    for s, xi, e in zip(self.sigma, x, exact, strict=True)
                )
            )
            upre_values.append(path.upre[1])
        differences = [u - loss for u, loss in zip(upre_values, losses, strict=True)]
        mean = math.fsum(differences) / len(differences)
        sd = math.sqrt(
            math.fsum((d - mean) ** 2 for d in differences) / (len(differences) - 1)
        )
        self.assertLess(abs(mean), 4.0 * sd / math.sqrt(len(differences)))

    def test_discrepancy_statuses(self) -> None:
        """Attainable, unattainable and always-met targets are reported honestly."""
        path = regularization_path(self.problem, self.grid, noise_norm=0.3)
        selection = select_regularization(path, "discrepancy")
        self.assertEqual(selection.status, "interior")
        assert selection.index is not None
        self.assertLessEqual(path.misfits[selection.index], 0.3)
        self.assertGreater(path.misfits[selection.index + 1], 0.3)
        tiny = select_regularization(
            regularization_path(self.problem, self.grid, noise_norm=1e-9), "discrepancy"
        )
        self.assertEqual(tiny.status, "unattainable")
        self.assertIsNone(tiny.regularization)
        huge = select_regularization(
            regularization_path(self.problem, self.grid, noise_norm=100.0),
            "discrepancy",
        )
        self.assertEqual(huge.status, "always_met")
        with self.assertRaises(ValueError):
            select_regularization(
                regularization_path(self.problem, self.grid), "discrepancy"
            )

    def test_boundary_is_flagged(self) -> None:
        """A grid that misses the optimum reports a boundary choice."""
        path = regularization_path(self.problem, logarithmic_grid(10.0, 100.0, 5))
        self.assertEqual(select_regularization(path, "gcv").status, "boundary")

    def test_l_curve_and_oracle_on_deriv2(self) -> None:
        """The L-curve corner and GCV land within a factor of the oracle error."""
        base = second_derivative_problem(24)
        y, sigma = noisy(base.exact_data, 0.01, SEED + 6)
        problem = LinearInverseProblem.create(
            base.operator, y, noise=ScaledIdentityCovariance(24, sigma * sigma)
        )
        path = regularization_path(
            problem,
            logarithmic_grid(1e-3, 1e3, 25),
            noise_norm=expected_noise_norm(24),
            truth=base.exact_solution,
        )
        oracle = path.oracle_index()
        assert oracle is not None and path.errors is not None
        for criterion in ("gcv", "upre", "discrepancy", "l_curve", "evidence"):
            selection = select_regularization(path, criterion)
            with self.subTest(criterion=criterion):
                assert selection.index is not None
                self.assertLess(path.errors[selection.index], 3.0 * path.errors[oracle])

    def test_hutchinson_degrees_of_freedom(self) -> None:
        """The matrix-free df estimate agrees with the exact trace within its MCSE."""
        base = second_derivative_problem(16)
        y, sigma = noisy(base.exact_data, 0.01, SEED + 7)
        problem = LinearInverseProblem.create(
            base.operator, y, noise=ScaledIdentityCovariance(16, sigma * sigma)
        )
        grid = (0.5, 2.0, 8.0)
        exact = regularization_path(problem, grid)
        estimate = regularization_path(
            problem, grid, trace="hutchinson", probes=40, seed=3
        )
        for d, e, err in zip(
            exact.degrees_of_freedom,
            estimate.degrees_of_freedom,
            estimate.degrees_of_freedom_errors,
            strict=True,
        ):
            self.assertLess(abs(d - e), 4.0 * err + 1e-6)
        self.assertLess(max_diff(exact.misfits, estimate.misfits), 1e-6)

    def test_grouped_cross_validation(self) -> None:
        """Leave-one-group-out refits; correlated noise is refused."""
        base = second_derivative_problem(16)
        y, sigma = noisy(base.exact_data, 0.01, SEED + 8)
        problem = LinearInverseProblem.create(
            base.operator, y, noise=ScaledIdentityCovariance(16, sigma * sigma)
        )
        result = grouped_cross_validation(
            problem, logarithmic_grid(1e-2, 1e2, 9), [i // 4 for i in range(16)]
        )
        self.assertEqual(len(result.fold_scores), 4)
        self.assertEqual(len(result.scores), 9)
        correlated = LinearInverseProblem.create(
            base.operator,
            y,
            noise=DenseCovariance(
                [[1.0 if i == j else 0.3 for j in range(16)] for i in range(16)]
            ),
        )
        with self.assertRaises(ValueError):
            grouped_cross_validation(
                correlated, (0.1, 1.0), [i // 4 for i in range(16)]
            )
        with self.assertRaises(ValueError):
            grouped_cross_validation(problem, (0.1, 1.0), [0] * 16)

    def test_invalid_grids(self) -> None:
        """Non-increasing or non-positive grids raise."""
        with self.assertRaises(ValueError):
            regularization_path(self.problem, (1.0, 0.5, 2.0))
        with self.assertRaises(ValueError):
            regularization_path(self.problem, (0.0, 1.0, 2.0))
        with self.assertRaises(ValueError):
            logarithmic_grid(1.0, 0.5)


if __name__ == "__main__":
    unittest.main()
