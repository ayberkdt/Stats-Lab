"""Tests for LIS, functional resolution and non-sine priors."""

from __future__ import annotations

import dataclasses
import math
import os
import unittest

from probstat.inverse.function_space import (
    HatBasis,
    PiecewiseConstantBasis,
    SineBasis,
)
from probstat.inverse.function_space_extended import (
    CosineBasis,
    ShiftedLegendreBasis,
    WhittleMaternPrior1D,
    _band_cholesky,
    averaging_kernel,
    cosine_spectral_prior,
    likelihood_informed_subspace,
    lis_linear_posterior,
    lis_projection_error,
    local_average_functional,
    matern_covariance,
    nonlinear_lis,
    point_functional,
    resolution_kernel_surface,
    resolution_operator,
)
from probstat.inverse.linear_gaussian import (
    linear_gaussian_posterior,
    prior_whitened_information,
)
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    DiagonalCovariance,
    IdentityOperator,
    ScaledIdentityCovariance,
    SelectionOperator,
    cholesky,
    inverse_spd,
    log_det_spd,
    matmul,
    solve_spd,
    symmetric_eigen,
    symmetrize,
    transpose,
)


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Return the Euclidean inner product."""
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _exp_cov(n: int, rho: float, scale: float = 1.0) -> DenseCovariance:
    """Return the AR(1)-type covariance ``scale * rho^|i-j|``."""
    return DenseCovariance(
        tuple(tuple(scale * rho ** abs(i - j) for j in range(n)) for i in range(n))
    )


def _forward_matrix(m: int, n: int) -> tuple[tuple[float, ...], ...]:
    """Return a fixed dense ``m x n`` matrix with smooth rows."""
    return tuple(
        tuple(
            math.sin(0.7 * (i + 1) * (j + 1)) + 0.3 * math.cos(i - j) for j in range(n)
        )
        for i in range(m)
    )


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


class LikelihoodInformedSubspaceTests(unittest.TestCase):
    """Closed form, generalized eigenproblem, exactness and error formulas."""

    def test_diagonal_closed_form_spectrum_and_rank(self) -> None:
        """H = diag(a_i^2 c_i / sigma^2): eigenvalues, rank and informed rank."""
        a = (2.0, 1.0, 0.5, 0.1)
        c = (1.0, 2.0, 3.0, 9.0)
        sigma2 = 0.25
        lis = likelihood_informed_subspace(
            [
                DenseOperator(
                    tuple(
                        tuple(a[i] if i == j else 0.0 for j in range(4))
                        for i in range(4)
                    )
                )
            ],
            noise=ScaledIdentityCovariance(4, sigma2),
            prior_factor=DiagonalCovariance(c),
            threshold=1.0,
        )
        expected = sorted(
            (ai * ai * ci / sigma2 for ai, ci in zip(a, c, strict=True)), reverse=True
        )
        for got, want in zip(expected, [16.0, 8.0, 3.0, 0.36], strict=True):
            self.assertAlmostEqual(got, want, places=13)
        for got, want in zip(lis.eigenvalues, expected, strict=True):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(lis.rank, 3)
        self.assertEqual(lis.informed_rank, 4)
        self.assertFalse(lis.capped)
        self.assertAlmostEqual(lis.gap, 3.0 - 0.36, places=12)
        # directions are P w = sqrt(c_i) e_i up to sign, ordered by lambda
        order = (0, 1, 2)
        for direction, i in zip(lis.parameter_directions, order, strict=True):
            self.assertAlmostEqual(abs(direction[i]), math.sqrt(c[i]), places=12)
        # single point: no spread
        self.assertEqual(lis.relative_sample_spread, 0.0)

    def test_generalized_eigenproblem_with_correlated_prior(self) -> None:
        """J' G^-1 J v = lambda C0^-1 v for v = P w, strongly correlated prior."""
        n, m = 5, 3
        a = _forward_matrix(m, n)
        prior = _exp_cov(n, 0.9)
        noise = DiagonalCovariance((0.2, 0.5, 0.1))
        lis = likelihood_informed_subspace(
            [DenseOperator(a)], noise=noise, prior_factor=prior, threshold=1e-8
        )
        spectrum = prior_whitened_information(a, noise, prior)
        self.assertEqual(lis.informed_rank, m)
        self.assertEqual(lis.rank, m)
        for got, want in zip(lis.eigenvalues, spectrum.eigenvalues, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-10 * max(1.0, want))
        c0 = tuple(tuple(0.9 ** abs(i - j) for j in range(n)) for i in range(n))
        c0_inv = inverse_spd(c0)
        gram = matmul(
            transpose(a),
            tuple(
                tuple(a[i][j] / (0.2, 0.5, 0.1)[i] for j in range(n)) for i in range(m)
            ),
        )
        for lam, v in zip(lis.eigenvalues[:m], lis.parameter_directions, strict=True):
            lhs = tuple(_dot(row, v) for row in gram)
            rhs = tuple(lam * _dot(row, v) for row in c0_inv)
            scale = max(abs(x) for x in lhs)
            for x, y in zip(lhs, rhs, strict=True):
                self.assertAlmostEqual(x, y, delta=1e-8 * scale)
            # C0^-1-normalization: v' C0^-1 v = ||w||^2 = 1
            self.assertAlmostEqual(_dot(v, tuple(_dot(r, v) for r in c0_inv)), 1.0, 8)

    def _problem(self) -> tuple[object, ...]:
        """Return a fixed linear-Gaussian problem with m < n."""
        n, m = 6, 3
        a = _forward_matrix(m, n)
        prior = _exp_cov(n, 0.8, 2.0)
        noise = DiagonalCovariance((0.2, 0.5, 0.1))
        truth = tuple(math.cos(0.9 * j) for j in range(n))
        y = tuple(_dot(a[i], truth) + 0.05 * (-1.0) ** i * (i + 1) for i in range(m))
        return a, y, noise, prior

    def test_rank_equal_to_informed_rank_is_exact(self) -> None:
        """The projected posterior equals the exact one at the informed rank."""
        a, y, noise, prior = self._problem()
        lis = likelihood_informed_subspace(
            [DenseOperator(a)], noise=noise, prior_factor=prior, threshold=1e-10
        )
        self.assertEqual(lis.rank, 3)
        post = lis_linear_posterior(a, y, noise=noise, prior_factor=prior, subspace=lis)
        exact = linear_gaussian_posterior(
            a, y, noise=noise, prior_covariance=prior, method="parameter_space"
        )
        for got, want in zip(post.mean, exact.mean, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-10)
        cov = post.covariance_matrix()
        for j in range(6):
            column = exact.covariance_action(
                tuple(1.0 if i == j else 0.0 for i in range(6))
            )
            for i in range(6):
                self.assertAlmostEqual(cov[i][j], column[i], delta=1e-10)
        err = lis_projection_error(a, y, noise=noise, prior_factor=prior, subspace=lis)
        self.assertLess(err.mean_error_whitened, 1e-10)
        self.assertLess(err.covariance_frobenius_error, 1e-10)
        self.assertLess(abs(err.kl_exact_to_lis), 1e-10)
        self.assertLess(err.omitted_spectrum_trace, 1e-12)
        self.assertLess(abs(err.trace_difference), 1e-10)

    def test_truncation_error_matches_omitted_spectrum_formulas(self) -> None:
        """Frobenius, operator, trace and KL errors from the omitted spectrum."""
        a, y, noise, prior = self._problem()
        full = likelihood_informed_subspace(
            [DenseOperator(a)], noise=noise, prior_factor=prior, threshold=1e-10
        )
        lam = full.eigenvalues
        threshold = math.sqrt(lam[0] * lam[1])  # keeps exactly one direction
        lis = likelihood_informed_subspace(
            [DenseOperator(a)], noise=noise, prior_factor=prior, threshold=threshold
        )
        self.assertEqual(lis.rank, 1)
        err = lis_projection_error(a, y, noise=noise, prior_factor=prior, subspace=lis)
        omitted = lam[1:3]
        shrink = [v / (1.0 + v) for v in omitted]
        self.assertAlmostEqual(
            err.covariance_frobenius_error,
            math.sqrt(sum(s * s for s in shrink)),
            delta=1e-10,
        )
        self.assertAlmostEqual(err.covariance_operator_error, max(shrink), delta=1e-10)
        self.assertAlmostEqual(err.trace_difference, sum(shrink), delta=1e-10)
        self.assertAlmostEqual(err.omitted_spectrum_trace, sum(shrink), delta=1e-10)
        self.assertAlmostEqual(err.omitted_spectrum_bound, max(shrink), delta=1e-12)
        # independent KL from the eigen-decomposition of K'K and b = K' W r
        p = cholesky(
            tuple(tuple(2.0 * 0.8 ** abs(i - j) for j in range(6)) for i in range(6))
        )
        gamma_inv = (1.0 / 0.2, 1.0 / 0.5, 1.0 / 0.1)
        k = tuple(
            tuple(
                math.sqrt(gamma_inv[i]) * sum(a[i][q] * p[q][j] for q in range(6))
                for j in range(6)
            )
            for i in range(3)
        )
        h = symmetrize(matmul(transpose(k), k))
        values, vectors = symmetric_eigen(h)
        order = sorted(range(6), key=lambda i: -values[i])
        b = tuple(
            sum(k[i][j] * math.sqrt(gamma_inv[i]) * y[i] for i in range(3))
            for j in range(6)
        )
        kl = 0.0
        for rank_index in order[1:]:
            lam_i = values[rank_index]
            if lam_i < 1e-12:
                continue
            v = tuple(vectors[r][rank_index] for r in range(6))
            kl += 0.5 * (math.log1p(lam_i) - lam_i / (1.0 + lam_i))
            kl += 0.5 * (_dot(v, b) / (1.0 + lam_i)) ** 2
        self.assertAlmostEqual(err.kl_exact_to_lis, kl, delta=1e-9)
        self.assertGreater(err.kl_exact_to_lis, 0.0)
        # the exact whitened posterior agrees with the library solver
        exact = linear_gaussian_posterior(
            a, y, noise=noise, prior_covariance=prior, method="parameter_space"
        )
        for got, want in zip(err.exact_mean, exact.whitened_mean, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-10)

    def test_prior_is_kept_in_unresolved_directions(self) -> None:
        """Rank zero returns the prior: mean mu, whitened covariance I."""
        a, y, noise, prior = self._problem()
        lis = likelihood_informed_subspace(
            [DenseOperator(a)], noise=noise, prior_factor=prior, threshold=1e12
        )
        self.assertEqual(lis.rank, 0)
        self.assertEqual(lis.gap, 0.0)
        mu = (0.5,) * 6
        post = lis_linear_posterior(
            a, y, noise=noise, prior_factor=prior, prior_mean=mu, subspace=lis
        )
        for got, want in zip(post.mean, mu, strict=True):
            self.assertEqual(got, want)
        for i in range(6):
            for j in range(6):
                self.assertEqual(post.whitened_covariance[i][j], 1.0 if i == j else 0.0)

    def test_operator_norm_bound_for_ill_conditioned_forward_map(self) -> None:
        """Singular values over eight decades; error equals the next eigenvalue term."""
        n = 6
        sv = tuple(10.0 ** (-k * 1.5) for k in range(n))  # 1 .. 3e-8
        a = tuple(tuple(sv[i] if i == j else 0.0 for j in range(n)) for i in range(n))
        noise = ScaledIdentityCovariance(n, 1e-4)
        prior = ScaledIdentityCovariance(n, 1.0)
        y = (1.0, 0.5, 0.2, 0.1, 0.05, 0.01)
        lis = likelihood_informed_subspace(
            [DenseOperator(a)], noise=noise, prior_factor=prior, threshold=1.0
        )
        lam = lis.eigenvalues
        self.assertEqual(lis.rank, sum(1 for v in lam if v > 1.0))
        err = lis_projection_error(a, y, noise=noise, prior_factor=prior, subspace=lis)
        self.assertAlmostEqual(
            err.covariance_operator_error,
            lam[lis.rank] / (1.0 + lam[lis.rank]),
            delta=1e-12,
        )
        self.assertLessEqual(err.covariance_operator_error, 0.5)

    def test_subspace_from_other_jacobians_has_positive_error(self) -> None:
        """A subspace averaged over different Jacobians is not exact: KL > 0."""
        a, y, noise, prior = self._problem()
        other = _forward_matrix(3, 6)
        other = tuple(
            tuple(v * (1.0 + 0.4 * ((i + j) % 3)) for j, v in enumerate(row))
            for i, row in enumerate(other)
        )
        lis = likelihood_informed_subspace(
            [DenseOperator(other)],
            noise=noise,
            prior_factor=prior,
            threshold=1e-10,
            max_rank=1,
        )
        self.assertTrue(lis.capped)
        err = lis_projection_error(a, y, noise=noise, prior_factor=prior, subspace=lis)
        self.assertGreater(err.kl_exact_to_lis, 1e-6)
        self.assertGreater(err.covariance_frobenius_error, 1e-6)

    def test_nonlinear_points_dependence_and_weights(self) -> None:
        """A nonlinear model gives point-dependent subspaces; spread is reported."""

        def jac(x: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
            """Jacobian of f = (x0 x1, x1 x2, sin x0)."""
            return (
                (x[1], x[0], 0.0),
                (0.0, x[2], x[1]),
                (math.cos(x[0]), 0.0, 0.0),
            )

        noise = ScaledIdentityCovariance(3, 0.1)
        prior = ScaledIdentityCovariance(3, 1.0)
        pts_a = [(1.0, 0.0, 0.0), (1.1, 0.1, 0.0)]
        pts_b = [(0.0, 1.0, 2.0), (0.1, 1.2, 1.9)]
        lis_a = nonlinear_lis(
            jac, pts_a, noise=noise, prior_factor=prior, threshold=1.0
        )
        lis_b = nonlinear_lis(
            jac, pts_b, noise=noise, prior_factor=prior, threshold=1.0
        )
        self.assertGreater(lis_a.relative_sample_spread, 0.0)
        self.assertAlmostEqual(
            lis_a.relative_standard_error,
            lis_a.relative_sample_spread / math.sqrt(2.0),
            places=14,
        )
        # leading directions differ between the point sets
        overlap = abs(_dot(lis_a.basis[0], lis_b.basis[0]))
        self.assertLess(overlap, 0.9)
        # weight on the first point only reproduces the single-point result
        single = nonlinear_lis(
            jac, pts_a[:1], noise=noise, prior_factor=prior, threshold=1.0
        )
        weighted = nonlinear_lis(
            jac,
            pts_a,
            noise=noise,
            prior_factor=prior,
            threshold=1.0,
            weights=[1.0, 0.0],
        )
        for u, v in zip(single.eigenvalues, weighted.eigenvalues, strict=True):
            self.assertAlmostEqual(u, v, delta=1e-12)
        # constant Jacobian: zero spread
        const = nonlinear_lis(
            lambda _x: ((1.0, 2.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 3.0)),
            pts_a,
            noise=noise,
            prior_factor=prior,
            threshold=1.0,
        )
        self.assertEqual(const.relative_sample_spread, 0.0)

    def test_failures(self) -> None:
        """Invalid input is rejected; a bad basis and dimension limit are caught."""
        a, y, noise, prior = self._problem()
        op = DenseOperator(a)
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                likelihood_informed_subspace(
                    [op], noise=noise, prior_factor=prior, threshold=bad
                )
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [], noise=noise, prior_factor=prior, threshold=1.0
            )
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [op, DenseOperator(a[:2])],
                noise=noise,
                prior_factor=prior,
                threshold=1.0,
            )
        with self.assertRaises(TypeError):
            likelihood_informed_subspace(
                [op], noise="x", prior_factor=prior, threshold=1.0
            )
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [op], noise=noise, prior_factor=prior, threshold=1.0, weights=[-1.0]
            )
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [op], noise=noise, prior_factor=prior, threshold=1.0, max_dimension=3
            )
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [op], noise=noise, prior_factor=prior, threshold=1.0, max_rank=0
            )
        with self.assertRaises(TypeError):
            nonlinear_lis(1.0, [(0.0,)], noise=noise, prior_factor=prior, threshold=1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            nonlinear_lis(
                lambda _x: a, [], noise=noise, prior_factor=prior, threshold=1.0
            )
        lis = likelihood_informed_subspace(
            [op], noise=noise, prior_factor=prior, threshold=1e-10
        )
        bad_basis = dataclasses.replace(
            lis, basis=tuple(tuple(2.0 * v for v in w) for w in lis.basis)
        )
        with self.assertRaises(ValueError):
            lis_linear_posterior(
                a, y, noise=noise, prior_factor=prior, subspace=bad_basis
            )
        with self.assertRaises(TypeError):
            lis_linear_posterior(a, y, noise=noise, prior_factor=prior, subspace="no")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            lis_linear_posterior(
                a, y[:2], noise=noise, prior_factor=prior, subspace=lis
            )
        with self.assertRaises(ValueError):
            lis.project((1.0, 2.0))
        projected = lis.project(tuple(1.0 for _ in range(6)))
        self.assertEqual(len(projected), 6)


class ResolutionTests(unittest.TestCase):
    """Resolution operator, averaging kernels and the mass matrix."""

    def test_piecewise_constant_exact_kernel(self) -> None:
        """G = I: R = rho I, kernel is a box of height rho/h on the cell."""
        n, length = 8, 1.0
        h = length / n
        sigma2, tau2 = 0.25, 1.0
        rho = tau2 / (tau2 + sigma2)
        basis = PiecewiseConstantBasis(n, length)
        res = resolution_operator(
            basis,
            IdentityOperator(n),
            noise=ScaledIdentityCovariance(n, sigma2),
            prior_factor=ScaledIdentityCovariance(n, tau2),
        )
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(
                    res.matrix[i][j], rho if i == j else 0.0, places=14
                )
        self.assertAlmostEqual(res.trace, rho * n, places=12)
        self.assertAlmostEqual(res.constant_defect, 1.0 - rho, places=12)
        s0 = 0.3
        ker = averaging_kernel(res, basis, point_functional(basis, s0), center=s0)
        cell = int(s0 / h)
        for j, value in enumerate(ker.coefficients):
            self.assertAlmostEqual(value, rho / h if j == cell else 0.0, places=12)
        self.assertAlmostEqual(ker.integral, rho, places=12)
        self.assertAlmostEqual(ker.target_integral, 1.0, places=12)
        centre = (cell + 0.5) * h
        self.assertAlmostEqual(ker.centroid, centre, places=12)
        self.assertAlmostEqual(ker.rms_width, h / math.sqrt(12.0), places=12)
        d = s0 - centre
        self.assertAlmostEqual(
            ker.backus_gilbert_spread, h + 12.0 * d * d / h, places=10
        )
        self.assertEqual(ker.negative_mass_fraction, 0.0)

    def test_strong_data_kernel_integrates_to_one(self) -> None:
        """With vanishing prior influence constants are resolved: integral 1."""
        n = 8
        basis = PiecewiseConstantBasis(n, 2.0)
        res = resolution_operator(
            basis,
            IdentityOperator(n),
            noise=ScaledIdentityCovariance(n, 1e-10),
            prior_factor=ScaledIdentityCovariance(n, 1.0),
        )
        self.assertLess(res.constant_defect, 1e-9)
        for s0 in (0.1, 0.77, 1.9):
            ker = averaging_kernel(res, basis, point_functional(basis, s0))
            self.assertAlmostEqual(ker.integral, 1.0, delta=1e-9)
            self.assertAlmostEqual(ker.integral, ker.target_integral, delta=1e-9)

    def test_block_average_kernel_matches_closed_form(self) -> None:
        """Block averages of 3 cells: box kernel of width 3h; BG spread W + 12 d^2/W."""
        n, k, length = 12, 3, 1.2
        h = length / n
        blocks = n // k
        sigma2, tau2 = 0.01, 4.0
        g = tuple(
            tuple(1.0 / k if j // k == b else 0.0 for j in range(n))
            for b in range(blocks)
        )
        basis = PiecewiseConstantBasis(n, length)
        res = resolution_operator(
            basis,
            DenseOperator(g),
            noise=ScaledIdentityCovariance(blocks, sigma2),
            prior_factor=ScaledIdentityCovariance(n, tau2),
        )
        rho = (tau2 / k) / (tau2 / k + sigma2)
        for i in range(n):
            for j in range(n):
                want = rho / k if i // k == j // k else 0.0
                self.assertAlmostEqual(res.matrix[i][j], want, places=13)
        self.assertAlmostEqual(res.constant_defect, 1.0 - rho, places=12)
        s0 = 0.55  # cell 5, block 1 = [0.3, 0.6), centre 0.45
        ker = averaging_kernel(res, basis, point_functional(basis, s0), center=s0)
        width = k * h
        self.assertAlmostEqual(ker.integral, rho, places=12)
        self.assertAlmostEqual(ker.centroid, 0.45, places=12)
        self.assertAlmostEqual(ker.rms_width, width / math.sqrt(12.0), places=12)
        d = s0 - 0.45
        self.assertAlmostEqual(
            ker.backus_gilbert_spread, width + 12.0 * d * d / width, places=10
        )
        # local-average functional over the whole block: weight 1/width on [0.3, 0.6)
        weight = local_average_functional(
            basis, lambda s: (1.0 / width) if 0.3 <= s < 0.6 else 0.0
        )
        ker2 = averaging_kernel(res, basis, weight)
        self.assertAlmostEqual(ker2.integral, rho, delta=1e-10)
        self.assertAlmostEqual(ker2.target_integral, 1.0, delta=1e-10)

    def test_matches_coefficient_space_and_perturbation(self) -> None:
        """R equals (G'G^-1G + C^-1)^-1 G'G^-1G and the noise-free response."""
        n = 10
        basis = HatBasis(n, 1.0)
        points = (0.15, 0.4, 0.5, 0.83)
        g_op = basis.evaluation_operator(points)
        g = tuple(tuple(r) for r in g_op.matrix)
        noise_var = (0.01, 0.02, 0.01, 0.05)
        c = tuple(tuple(1.5 * 0.5 ** abs(i - j) for j in range(n)) for i in range(n))
        res = resolution_operator(
            basis,
            g_op,
            noise=DiagonalCovariance(noise_var),
            prior_factor=DenseCovariance(c),
        )
        # independent coefficient-space oracle with explicit inverses
        weighted = tuple(
            tuple(g[i][j] / noise_var[i] for j in range(n)) for i in range(len(points))
        )
        information = matmul(transpose(g), weighted)
        precision = tuple(
            tuple(information[i][j] + inverse_spd(c)[i][j] for j in range(n))
            for i in range(n)
        )
        reference = matmul(inverse_spd(symmetrize(precision)), information)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(res.matrix[i][j], reference[i][j], delta=1e-8)
        # finite perturbation of the true field through the exact posterior mean
        truth = tuple(math.sin(2.0 * j) for j in range(n))
        base = tuple(_dot(row, truth) for row in g)

        def posterior_mean(y: tuple[float, ...]) -> tuple[float, ...]:
            """Exact posterior mean for data y."""
            return linear_gaussian_posterior(
                g_op,
                y,
                noise=DiagonalCovariance(noise_var),
                prior_covariance=DenseCovariance(c),
            ).mean

        mean0 = posterior_mean(base)
        for j in (0, 4, 9):
            bumped = tuple(base[i] + g[i][j] for i in range(len(points)))
            mean1 = posterior_mean(bumped)
            for i in range(n):
                self.assertAlmostEqual(
                    mean1[i] - mean0[i], res.matrix[i][j], delta=1e-9
                )
        # mass matrix: hat constants are not representable -> positive defect
        self.assertGreater(res.constant_defect, 1e-3)

    def test_mass_matrix_enters_the_kernel(self) -> None:
        """Int a f = l' R x holds with M; the naive R' l fails for hat functions."""
        n = 9
        basis = HatBasis(n, 2.0)
        g_op = basis.evaluation_operator((0.3, 0.9, 1.5))
        res = resolution_operator(
            basis,
            g_op,
            noise=ScaledIdentityCovariance(3, 0.04),
            prior_factor=_exp_cov(n, 0.6),
        )
        s0 = 0.77
        ell = point_functional(basis, s0)
        ker = averaging_kernel(res, basis, ell)
        x = tuple(math.cos(1.3 * j) + 0.2 * j for j in range(n))
        target = _dot(ell, tuple(_dot(row, x) for row in res.matrix))
        panels = 4000
        pts = [(q + 0.5) * 2.0 / panels for q in range(panels)]
        kv = basis.evaluate(ker.coefficients, pts)
        fv = basis.evaluate(x, pts)
        quad = sum(a * b for a, b in zip(kv, fv, strict=True)) * 2.0 / panels
        self.assertAlmostEqual(quad, target, delta=1e-5)
        naive = tuple(_dot(row, ell) for row in transpose(res.matrix))
        nv = basis.evaluate(naive, pts)
        quad_naive = sum(a * b for a, b in zip(nv, fv, strict=True)) * 2.0 / panels
        self.assertGreater(abs(quad_naive - target), 1e-3)
        # kernel surface agrees with the point kernels
        surface = resolution_kernel_surface(res, basis, [s0], pts[:5])
        for q in range(5):
            self.assertAlmostEqual(surface[0][q], kv[q], delta=1e-9)
        # the integral is l' R c with c = M^-1 int phi
        mass = basis.mass_matrix()
        c = solve_spd(mass, res.integral_vector)
        self.assertAlmostEqual(
            ker.integral,
            _dot(ell, tuple(_dot(row, c) for row in res.matrix)),
            places=10,
        )

    def test_failures(self) -> None:
        """Dimension mismatches and invalid arguments are rejected."""
        basis = PiecewiseConstantBasis(4, 1.0)
        noise = ScaledIdentityCovariance(4, 0.1)
        prior = ScaledIdentityCovariance(4, 1.0)
        with self.assertRaises(ValueError):
            resolution_operator(
                basis,
                IdentityOperator(5),
                noise=ScaledIdentityCovariance(5, 1.0),
                prior_factor=ScaledIdentityCovariance(5, 1.0),
            )
        with self.assertRaises(ValueError):
            resolution_operator(
                basis,
                IdentityOperator(4),
                noise=ScaledIdentityCovariance(3, 1.0),
                prior_factor=prior,
            )
        with self.assertRaises(ValueError):
            resolution_operator(
                basis,
                IdentityOperator(4),
                noise=noise,
                prior_factor=prior,
                max_dimension=2,
            )
        with self.assertRaises(ValueError):
            resolution_operator(
                basis, IdentityOperator(4), noise=noise, prior_factor=prior, panels=0
            )
        res = resolution_operator(
            basis, IdentityOperator(4), noise=noise, prior_factor=prior
        )
        with self.assertRaises(ValueError):
            averaging_kernel(res, basis, (1.0, 0.0))
        with self.assertRaises(ValueError):
            averaging_kernel(res, basis, point_functional(basis, 0.5), center=2.0)
        with self.assertRaises(ValueError):
            point_functional(basis, 1.5)
        with self.assertRaises(ValueError):
            resolution_kernel_surface(res, SineBasis(3, 1.0), [0.5], [0.5])

    def test_sine_basis_does_not_resolve_constants(self) -> None:
        """Sine functions cannot represent 1: kernel integral differs from target."""
        n = 6
        basis = SineBasis(n, 1.0)
        res = resolution_operator(
            basis,
            IdentityOperator(n),
            noise=ScaledIdentityCovariance(n, 1e-12),
            prior_factor=ScaledIdentityCovariance(n, 1.0),
        )
        ker = averaging_kernel(res, basis, point_functional(basis, 0.4))
        # R = I: the kernel is the L2 projection of the delta; its integral is the
        # projection of the constant 1 evaluated at 0.4
        proj = _dot(basis.evaluate(res.constant_coefficients, [0.4]), (1.0,))
        self.assertAlmostEqual(ker.integral, ker.target_integral, delta=1e-9)
        self.assertAlmostEqual(ker.target_integral, proj, delta=1e-12)
        self.assertGreater(abs(1.0 - ker.integral), 0.02)


class WhittleMaternTests(unittest.TestCase):
    """Banded SPDE prior against dense oracles and continuum covariances."""

    def test_precision_closed_form_and_banded_cholesky(self) -> None:
        """Q = h (kappa^2 + T/h^2)^alpha; covariance columns equal dense inverse."""
        n, kappa, length = 7, 2.0, 3.0
        for boundary in ("dirichlet", "neumann"):
            for alpha in (1, 2):
                prior = WhittleMaternPrior1D(
                    n,
                    length,
                    kappa,
                    alpha=alpha,
                    boundary=boundary,  # type: ignore[arg-type]
                )
                h = prior.spacing
                self.assertAlmostEqual(
                    h, length / (n + 1 if boundary == "dirichlet" else n)
                )
                lap = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    lap[i][i] = kappa**2 + 2.0 / h**2
                    if i > 0:
                        lap[i][i - 1] = -1.0 / h**2
                    if i < n - 1:
                        lap[i][i + 1] = -1.0 / h**2
                if boundary == "neumann":
                    lap[0][0] -= 1.0 / h**2
                    lap[-1][-1] -= 1.0 / h**2
                lmat = tuple(tuple(r) for r in lap)
                q = lmat if alpha == 1 else matmul(lmat, lmat)
                q = tuple(tuple(h * v for v in row) for row in q)
                got = prior.precision_matrix()
                for i in range(n):
                    for j in range(n):
                        self.assertAlmostEqual(
                            got[i][j], q[i][j], delta=1e-12 * max(1.0, abs(q[i][j]))
                        )
                self.assertEqual(prior.bandwidth, alpha)
                cov = inverse_spd(q)
                for j in range(n):
                    col = prior.covariance_column(j)
                    for i in range(n):
                        self.assertAlmostEqual(
                            col[i], cov[i][j], delta=1e-10 * cov[i][i]
                        )
                self.assertAlmostEqual(
                    prior.log_det_precision(), log_det_spd(q), places=9
                )

    def test_continuum_matern_values(self) -> None:
        """Closed forms of the Matern covariance, variance and decay."""
        self.assertAlmostEqual(matern_covariance(0.0, 2.0, 1), 0.25, places=14)
        self.assertAlmostEqual(
            matern_covariance(0.5, 2.0, 1), math.exp(-1.0) / 4.0, places=14
        )
        self.assertAlmostEqual(matern_covariance(0.0, 2.0, 2), 1.0 / 32.0, places=14)
        self.assertAlmostEqual(
            matern_covariance(-0.5, 2.0, 2, 3.0),
            9.0 * 2.0 * math.exp(-1.0) / 32.0,
            places=14,
        )
        # alpha=2 covariance is the convolution of two alpha=1 covariances:
        # int e^{-k|u|}/(2k) e^{-k|r-u|}/(2k) du = (1 + k r) e^{-k r} / (4 k^3)
        k, r, steps = 1.3, 0.8, 200000
        lim = 40.0
        du = 2.0 * lim / steps
        conv = (
            sum(
                math.exp(-k * abs(-lim + (q + 0.5) * du))
                * math.exp(-k * abs(r - (-lim + (q + 0.5) * du)))
                for q in range(steps)
            )
            * du
            / (4.0 * k * k)
        )
        self.assertAlmostEqual(conv, matern_covariance(r, k, 2), delta=1e-6)

    def test_interior_covariance_matches_matern_and_boundary_effect(self) -> None:
        """Neumann, alpha=1: variance doubling at the boundary, 1% decay distance."""
        length, kappa, n = 10.0, 1.0, 400
        prior = WhittleMaternPrior1D(n, length, kappa, alpha=1, boundary="neumann")
        rep = prior.boundary_effect(0.01)
        h = prior.spacing
        self.assertAlmostEqual(rep.stationary_variance, 0.5, places=14)
        # images boundary variance C(0) + C(2 s) with 2 s = h: ratio 1 + exp(-kappa h)
        self.assertAlmostEqual(
            rep.exact_boundary_variance_ratio, 1.0 + math.exp(-kappa * h), delta=1e-7
        )
        self.assertAlmostEqual(
            rep.boundary_variance_ratio, rep.exact_boundary_variance_ratio, delta=2e-3
        )
        self.assertGreater(rep.boundary_variance_ratio, 1.9)
        self.assertTrue(rep.interior_reached)
        self.assertAlmostEqual(rep.center_variance_ratio, 1.0, delta=2e-3)
        # ratio - 1 = exp(-2 kappa d) crosses 1% at d = ln(100) / (2 kappa)
        expected = math.log(100.0) / (2.0 * kappa)
        self.assertAlmostEqual(rep.decay_distance, expected, delta=2.0 * h)
        # the only center-column error left is the far boundary image, e^{-kappa L/2}
        self.assertAlmostEqual(
            rep.center_column_error, math.exp(-kappa * length / 2.0), delta=2e-4
        )
        self.assertGreater(rep.boundary_column_error, 0.3)
        self.assertLess(rep.discretization_error, 3e-3)

    def test_dirichlet_variance_vanishes_at_boundary(self) -> None:
        """Dirichlet alpha=1: variance (1 - exp(-2 kappa s)) / (2 kappa) near 0."""
        length, kappa, n = 10.0, 1.0, 400
        prior = WhittleMaternPrior1D(n, length, kappa, alpha=1, boundary="dirichlet")
        rep = prior.boundary_effect(0.01)
        h = prior.spacing
        self.assertAlmostEqual(
            rep.exact_boundary_variance_ratio,
            1.0 - math.exp(-2.0 * kappa * h),
            delta=1e-9,
        )
        self.assertAlmostEqual(
            rep.boundary_variance_ratio, rep.exact_boundary_variance_ratio, delta=2e-3
        )
        self.assertLess(rep.boundary_variance_ratio, 0.1)
        self.assertAlmostEqual(
            rep.decay_distance, math.log(100.0) / (2.0 * kappa), delta=2.0 * h
        )

    def test_alpha_two_matches_images_and_converges_second_order(self) -> None:
        """alpha=2: discretization error against the images formula is O(h^2)."""
        errors = []
        for n in (50, 100, 200):
            prior = WhittleMaternPrior1D(n, 1.0, 6.0, alpha=2, boundary="neumann")
            col = prior.covariance_column(n // 4)
            s = prior.nodes[n // 4]
            err = max(
                abs(col[j] - prior.continuum_covariance(s, prior.nodes[j]))
                for j in range(n)
            ) / prior.stationary_covariance(0.0)
            errors.append(err)
        self.assertLess(errors[2], 1e-3)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
            self.assertGreater(coarse / fine, 3.0)
            self.assertLess(coarse / fine, 5.0)

    def test_mesh_independence_of_functionals(self) -> None:
        """Same continuum prior at n=50,100,200: error to the continuum is O(h^2)."""
        errors = []
        for n in (50, 100, 200):
            prior = WhittleMaternPrior1D(n, 1.0, 5.0, alpha=1, boundary="neumann")
            index = min(range(n), key=lambda i: abs(prior.nodes[i] - 0.5))
            s = prior.nodes[index]
            discrete = prior.covariance_column(index)[index]
            # the continuum variance at s includes the boundary effect (~1e-3 here)
            continuum = prior.continuum_covariance(s, s)
            self.assertGreater(continuum - 0.1, 5e-4)
            errors.append(abs(discrete - continuum))
        self.assertLess(errors[2], 1e-4)
        self.assertGreater(errors[0] / errors[2], 8.0)

    def test_sampler_covariance_within_mcse(self) -> None:
        """Sample covariance of banded-Cholesky draws agrees with Q^-1."""
        n, draws = 12, 3000
        prior = WhittleMaternPrior1D(n, 1.0, 4.0, alpha=2, boundary="dirichlet")
        samples = prior.sample(draws, seed=17)
        again = prior.sample(draws, seed=17)
        self.assertEqual(samples[:3], again[:3])
        cov = inverse_spd(prior.precision_matrix())
        for i, j in ((0, 0), (5, 5), (5, 6), (2, 9), (11, 11)):
            est = sum(s[i] * s[j] for s in samples) / draws
            mcse = math.sqrt((cov[i][i] * cov[j][j] + cov[i][j] ** 2) / draws)
            self.assertLess(abs(est - cov[i][j]), 5.0 * mcse)

    def test_prior_factor_in_linear_gaussian_inversion(self) -> None:
        """Banded factor operator equals the dense-covariance posterior."""
        n = 40
        prior = WhittleMaternPrior1D(n, 1.0, 8.0, alpha=2, boundary="neumann")
        indices = (3, 11, 20, 27, 36)
        forward = SelectionOperator(n, indices)
        truth = prior.sample(1, seed=3)[0]
        y = tuple(truth[i] + 0.01 * (k - 2) for k, i in enumerate(indices))
        noise = ScaledIdentityCovariance(len(indices), 0.05**2)
        banded = linear_gaussian_posterior(
            forward,
            y,
            noise=noise,
            prior_factor=prior.prior_factor(),
            method="parameter_space",
        )
        dense_cov = symmetrize(inverse_spd(prior.precision_matrix()))
        dense = linear_gaussian_posterior(
            forward, y, noise=noise, prior_covariance=DenseCovariance(dense_cov)
        )
        for u, v in zip(banded.mean, dense.mean, strict=True):
            self.assertAlmostEqual(u, v, delta=1e-7 * max(1.0, abs(v)))
        var_b = banded.marginal_variances()
        var_d = dense.marginal_variances()
        for u, v in zip(var_b, var_d, strict=True):
            self.assertAlmostEqual(u, v, delta=1e-7 * v)
        self.assertAlmostEqual(banded.log_evidence, dense.log_evidence, delta=1e-6)

    def test_failures(self) -> None:
        """Invalid parameters, non-SPD band and image-sum overflow are rejected."""
        with self.assertRaises(ValueError):
            WhittleMaternPrior1D(10, 1.0, 0.0)
        with self.assertRaises(ValueError):
            WhittleMaternPrior1D(10, 1.0, -1.0)
        with self.assertRaises(ValueError):
            WhittleMaternPrior1D(10, 1.0, 1.0, alpha=3)
        with self.assertRaises(TypeError):
            WhittleMaternPrior1D(10, 1.0, 1.0, alpha=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            WhittleMaternPrior1D(10, 1.0, 1.0, boundary="robin")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            WhittleMaternPrior1D(3, 1.0, 1.0)
        with self.assertRaises(ValueError):
            WhittleMaternPrior1D(10, 1.0, 1.0, amplitude=0.0)
        with self.assertRaises(ValueError):
            _band_cholesky([[1.0, 0.0], [1.0, 3.0]], 1)  # Q = [[1, 3], [3, 1]]
        with self.assertRaises(ValueError):
            _band_cholesky([[0.0, 0.0]], 1)
        prior = WhittleMaternPrior1D(10, 1.0, 1.0)
        with self.assertRaises(ValueError):
            prior.continuum_covariance(0.5, 2.0)
        with self.assertRaises(ValueError):
            prior.covariance_column(10)
        with self.assertRaises(ValueError):
            prior.solve_precision((1.0, 2.0))
        with self.assertRaises(ValueError):
            prior.boundary_effect(1.5)
        with self.assertRaises(ValueError):
            prior.sample(0)
        wide = WhittleMaternPrior1D(10, 1.0, 1e-6)
        with self.assertRaises(ValueError):
            wide.continuum_covariance(0.1, 0.2)
        with self.assertRaises(ValueError):
            matern_covariance(1.0, 1.0, 3)
        # a prior much shorter than the correlation length never reaches stationarity
        short = WhittleMaternPrior1D(40, 1.0, 1.0, alpha=1, boundary="neumann")
        self.assertFalse(short.boundary_effect(0.01).interior_reached)


class CosineAndLegendreTests(unittest.TestCase):
    """Exact mass matrices of the new bases and the Neumann spectral prior."""

    def test_cosine_orthonormal_and_integrals(self) -> None:
        """Gram matrix by quadrature, constants contained, integrals closed form."""
        basis = CosineBasis(6, 2.0)
        gram = _midpoint_gram(basis, 2000)
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(gram[i][j], 1.0 if i == j else 0.0, delta=1e-6)
                self.assertEqual(basis.mass_matrix()[i][j], 1.0 if i == j else 0.0)
        integrals = basis.integral_operator([lambda _s: 1.0], 50).matrix[0]
        self.assertAlmostEqual(integrals[0], math.sqrt(2.0), places=12)
        for value in integrals[1:]:
            self.assertAlmostEqual(value, 0.0, places=12)
        # the constant 1 lies in the span: coefficient sqrt(L) on phi_0
        values = basis.evaluate((math.sqrt(2.0),) + (0.0,) * 5, [0.0, 0.7, 2.0])
        for v in values:
            self.assertAlmostEqual(v, 1.0, places=14)
        # derivative of cos modes vanishes at the ends: phi_k'(0) ~ 0 by symmetry
        h = 1e-6
        d = basis.evaluate((0.0, 1.0, 0.0, 0.0, 0.0, 0.0), [h, 0.0])
        self.assertAlmostEqual((d[0] - d[1]) / h, 0.0, delta=1e-5)

    def test_legendre_values_mass_and_integrals(self) -> None:
        """Shifted Legendre: explicit P_2, P_3, endpoint values, exact mass."""
        length = 3.0
        raw = ShiftedLegendreBasis(5, length, normalized=False)
        s = [0.0, 0.4, 1.5, 2.9, 3.0]
        for point in s:
            t = 2.0 * point / length - 1.0
            row = raw.evaluate((0.0, 0.0, 1.0, 0.0, 0.0), [point])[0]
            self.assertAlmostEqual(row, 0.5 * (3.0 * t * t - 1.0), places=13)
            row3 = raw.evaluate((0.0, 0.0, 0.0, 1.0, 0.0), [point])[0]
            self.assertAlmostEqual(row3, 0.5 * (5.0 * t**3 - 3.0 * t), places=13)
            row4 = raw.evaluate((0.0, 0.0, 0.0, 0.0, 1.0), [point])[0]
            self.assertAlmostEqual(
                row4, (35.0 * t**4 - 30.0 * t * t + 3.0) / 8.0, places=13
            )
        gram = _midpoint_gram(raw, 3000)
        mass = raw.mass_matrix()
        for i in range(5):
            for j in range(5):
                want = length / (2 * i + 1) if i == j else 0.0
                self.assertAlmostEqual(mass[i][j], want, places=14)
                self.assertAlmostEqual(gram[i][j], want, delta=2e-5)
        norm = ShiftedLegendreBasis(5, length)
        gram_n = _midpoint_gram(norm, 3000)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(gram_n[i][j], 1.0 if i == j else 0.0, delta=2e-5)
        # endpoint values P_k(1) = 1, P_k(-1) = (-1)^k
        ends = raw.evaluation_operator([0.0, length]).matrix
        for k in range(5):
            self.assertAlmostEqual(ends[0][k], (-1.0) ** k, places=13)
            self.assertAlmostEqual(ends[1][k], 1.0, places=13)
        ones = raw.integral_operator([lambda _s: 1.0], 30).matrix[0]
        self.assertAlmostEqual(ones[0], length, places=12)
        for v in ones[1:]:
            self.assertAlmostEqual(v, 0.0, places=11)
        with self.assertRaises(TypeError):
            ShiftedLegendreBasis(3, 1.0, normalized=1)  # type: ignore[arg-type]
        single = ShiftedLegendreBasis(1, 2.0)
        self.assertAlmostEqual(
            single.evaluate((1.0,), [0.3])[0], 1.0 / math.sqrt(2.0), places=14
        )

    def test_neumann_spectral_prior_matches_images_with_tail_bound(self) -> None:
        """Cosine-spectral covariance equals the Neumann images up to the tail."""
        length, kappa, n = 1.0, 5.0, 400
        basis = CosineBasis(n, length)
        variances = tuple(
            sd * sd
            for sd in cosine_spectral_prior(n, length, kappa, s=1.0).standard_deviations
        )
        prior = WhittleMaternPrior1D(20, length, kappa, alpha=1, boundary="neumann")
        tail = 2.0 * length / (math.pi**2 * (n - 1))
        for s, t in ((0.0, 0.0), (0.2, 0.7), (0.5, 0.5), (1.0, 0.9)):
            row_s = basis.evaluation_operator([s]).matrix[0]
            row_t = basis.evaluation_operator([t]).matrix[0]
            spectral = sum(
                v * a * b for v, a, b in zip(variances, row_s, row_t, strict=True)
            )
            images = prior.continuum_covariance(s, t)
            self.assertLessEqual(abs(spectral - images), tail)
            self.assertLess(
                spectral, images + 1e-15
            )  # truncation only removes variance
        with self.assertRaises(ValueError):
            cosine_spectral_prior(5, 1.0, -1.0)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudies(unittest.TestCase):
    """Larger sweeps: boundary-effect tables and sampler accuracy."""

    def test_boundary_effect_sweep_and_large_sampler(self) -> None:
        """Print measured boundary effect for both boundaries and alpha; check MC."""
        for boundary in ("neumann", "dirichlet"):
            for alpha in (1, 2):
                prior = WhittleMaternPrior1D(
                    800,
                    10.0,
                    1.0,
                    alpha=alpha,
                    boundary=boundary,  # type: ignore[arg-type]
                )
                rep = prior.boundary_effect(0.01)
                print(
                    f"SLOW {boundary} alpha={alpha}: boundary ratio "
                    f"{rep.boundary_variance_ratio:.6f} (images "
                    f"{rep.exact_boundary_variance_ratio:.6f}), decay "
                    f"{rep.decay_distance:.4f}, centre col err "
                    f"{rep.center_column_error:.3e}, disc err "
                    f"{rep.discretization_error:.3e}"
                )
                self.assertTrue(rep.interior_reached)
                self.assertLess(rep.discretization_error, 1e-3)
        prior = WhittleMaternPrior1D(60, 1.0, 5.0, alpha=2, boundary="neumann")
        draws = 20000
        samples = prior.sample(draws, seed=5)
        cov = inverse_spd(prior.precision_matrix())
        worst = 0.0
        for i, j in ((0, 0), (30, 30), (30, 33), (10, 50), (59, 59)):
            est = sum(s[i] * s[j] for s in samples) / draws
            mcse = math.sqrt((cov[i][i] * cov[j][j] + cov[i][j] ** 2) / draws)
            worst = max(worst, abs(est - cov[i][j]) / mcse)
        print(f"SLOW sampler worst |z| over 5 entries: {worst:.3f}")
        self.assertLess(worst, 5.0)


if __name__ == "__main__":
    unittest.main()
