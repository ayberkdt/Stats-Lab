"""Tests for exact and low-rank linear-Gaussian Bayesian inversion."""

from __future__ import annotations

import math
import unittest

from probstat.inverse.linear_gaussian import (
    GaussianLinearPosterior,
    linear_gaussian_posterior,
    low_rank_posterior,
    prior_whitened_information,
)
from probstat.inverse.regularization import tikhonov
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    DiagonalCovariance,
    ZeroOperator,
    add,
    inverse_spd,
    log_det_spd,
    lu_solve,
    matmul,
    matvec,
    transpose,
)

LOG_2PI = math.log(2.0 * math.pi)


def _normal_logpdf(x: float, mean: float, variance: float) -> float:
    """Return the scalar normal log density."""
    return -0.5 * ((x - mean) ** 2 / variance + math.log(variance) + LOG_2PI)


def _dense_oracle(a, gamma, c0, mu, y):
    """Precision-form posterior and evidence, independent of the module."""
    g_inv = inverse_spd(gamma)
    precision = add(matmul(transpose(a), matmul(g_inv, a)), inverse_spd(c0))
    cov = inverse_spd(precision)
    resid = tuple(yi - ai for yi, ai in zip(y, matvec(a, mu), strict=True))
    gain = matvec(cov, matvec(transpose(a), matvec(g_inv, resid)))
    mean = tuple(m + g for m, g in zip(mu, gain, strict=True))
    s = add(matmul(a, matmul(c0, transpose(a))), gamma)
    s_inv = inverse_spd(s)
    quad = sum(r * q for r, q in zip(resid, matvec(s_inv, resid), strict=True))
    evidence = -0.5 * (quad + log_det_spd(s) + len(y) * LOG_2PI)
    return mean, cov, evidence


def _diff(left, right) -> float:
    """Return the max absolute difference of two vectors."""
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


A32 = ((1.0, 0.5), (-0.3, 2.0), (0.7, -1.1))
G3 = ((0.5, 0.1, 0.0), (0.1, 0.8, -0.2), (0.0, -0.2, 0.6))
C2 = ((1.2, 0.4), (0.4, 0.7))
MU2 = (0.3, -0.5)
Y3 = (1.0, -0.4, 0.9)

A23 = ((1.0, -0.5, 0.2), (0.3, 1.5, -0.8))
G2 = ((0.4, -0.1), (-0.1, 0.9))
C3 = ((1.0, 0.3, 0.1), (0.3, 0.8, -0.2), (0.1, -0.2, 0.6))
MU3 = (0.1, 0.2, -0.3)
Y2 = (0.7, -0.6)


class ScalarConjugateTests(unittest.TestCase):
    """Closed-form scalar conjugate case."""

    def test_closed_form(self) -> None:
        """Closed form."""
        a, mu, s0, s, y = 2.0, 1.0, 0.5, 0.3, 3.0
        precision = a * a / s**2 + 1.0 / s0**2
        variance = 1.0 / precision
        mean = mu + variance * a / s**2 * (y - a * mu)
        evidence = _normal_logpdf(y, a * mu, a * a * s0 * s0 + s * s)
        for method in ("data_space", "parameter_space"):
            post = linear_gaussian_posterior(
                ((a,),),
                (y,),
                noise=DiagonalCovariance((s * s,)),
                prior_mean=(mu,),
                prior_covariance=DiagonalCovariance((s0 * s0,)),
                method=method,
            )
            self.assertAlmostEqual(post.mean[0], mean, places=12)
            self.assertAlmostEqual(post.covariance_action((1.0,))[0], variance, 12)
            self.assertAlmostEqual(post.marginal_variances()[0], variance, 12)
            self.assertAlmostEqual(post.log_evidence, evidence, places=12)


class DenseCorrelatedTests(unittest.TestCase):
    """Independent precision-form oracle on correlated dense problems."""

    def _check(self, a, gamma, c0, mu, y) -> None:
        mean, cov, evidence = _dense_oracle(a, gamma, c0, mu, y)
        n = len(mu)
        posts = {
            method: linear_gaussian_posterior(
                a,
                y,
                noise=DenseCovariance(gamma),
                prior_mean=mu,
                prior_covariance=DenseCovariance(c0),
                method=method,
            )
            for method in ("data_space", "parameter_space")
        }
        for post in posts.values():
            self.assertIsInstance(post, GaussianLinearPosterior)
            self.assertLess(_diff(post.mean, mean), 1e-11)
            self.assertAlmostEqual(post.log_evidence, evidence, places=11)
            for j in range(n):
                unit = tuple(1.0 if i == j else 0.0 for i in range(n))
                column = tuple(cov[i][j] for i in range(n))
                self.assertLess(_diff(post.covariance_action(unit), column), 1e-11)
            self.assertLess(
                _diff(post.marginal_variances(), tuple(cov[i][i] for i in range(n))),
                1e-11,
            )
        data, param = posts["data_space"], posts["parameter_space"]
        self.assertLess(_diff(data.mean, param.mean), 1e-11)
        self.assertAlmostEqual(data.log_evidence, param.log_evidence, places=11)
        self.assertAlmostEqual(
            data.log_det_data_covariance, param.log_det_data_covariance, places=11
        )

    def test_overdetermined(self) -> None:
        """Overdetermined."""
        self._check(A32, G3, C2, MU2, Y3)

    def test_underdetermined(self) -> None:
        """Underdetermined."""
        self._check(A23, G2, C3, MU3, Y2)

    def test_predictive(self) -> None:
        """Predictive."""
        _, cov, _ = _dense_oracle(A32, G3, C2, MU2, Y3)
        h = ((1.0, 2.0), (0.5, -1.0))
        noise = DiagonalCovariance((0.1, 0.2))
        for method in ("data_space", "parameter_space"):
            post = linear_gaussian_posterior(
                A32,
                Y3,
                noise=DenseCovariance(G3),
                prior_mean=MU2,
                prior_covariance=DenseCovariance(C2),
                method=method,
            )
            pred = post.predictive(h)
            expected = matmul(h, matmul(cov, transpose(h)))
            self.assertLess(
                max(
                    _diff(r, e) for r, e in zip(pred.covariance, expected, strict=True)
                ),
                1e-11,
            )
            self.assertLess(_diff(pred.mean, matvec(h, post.mean)), 1e-12)
            noisy = post.predictive(h, noise=noise)
            self.assertTrue(noisy.includes_noise)
            self.assertAlmostEqual(
                noisy.covariance[1][1] - pred.covariance[1][1], 0.2, places=12
            )


class SpecialCaseTests(unittest.TestCase):
    """Zero operator, singular prior, vanishing noise."""

    @staticmethod
    def _null_vector(p):
        """Return a unit-ish vector v with P' v = 0 for the 5 x 2 factor p."""
        # solve for v = (x, y, 1, 0, 0): columns of P give two equations
        lhs = ((p[0][0], p[1][0]), (p[0][1], p[1][1]))
        rhs = (-p[2][0], -p[2][1])
        x, y = lu_solve(lhs, rhs)
        return (x, y, 1.0, 0.0, 0.0)

    def test_zero_operator_returns_prior(self) -> None:
        """Zero operator returns prior."""
        y = (0.4, -0.2, 0.9)
        gamma = DenseCovariance(G3)
        c0 = DenseCovariance(C2)
        expected = -0.5 * (
            sum(a * b for a, b in zip(y, matvec(inverse_spd(G3), y), strict=True))
            + log_det_spd(G3)
            + 3 * LOG_2PI
        )
        for method in ("data_space", "parameter_space"):
            post = linear_gaussian_posterior(
                ZeroOperator(3, 2),
                y,
                noise=gamma,
                prior_mean=MU2,
                prior_covariance=c0,
                method=method,
            )
            self.assertEqual(post.mean, MU2)
            self.assertLess(_diff(post.covariance_action((1.0, 0.0)), C2[0]), 1e-13)
            self.assertAlmostEqual(post.log_evidence, expected, places=12)

    def test_singular_prior_factor_with_fewer_data_than_unknowns(self) -> None:
        """Singular prior factor with fewer data than unknowns."""
        p = (
            (1.0, 0.0),
            (0.5, 1.0),
            (0.0, -1.0),
            (2.0, 0.3),
            (-1.0, 0.8),
        )
        a = ((1.0, 0.2, -0.4, 0.0, 0.5), (0.0, 1.0, 0.3, -0.7, 0.1))
        gamma = G2
        y = (0.5, -0.3)
        mu = (0.1, 0.0, -0.2, 0.3, 0.0)
        c0 = matmul(p, transpose(p))
        # Oracle: S-form posterior with explicit singular C0.
        s = add(matmul(a, matmul(c0, transpose(a))), gamma)
        gain = matmul(c0, matmul(transpose(a), inverse_spd(s)))
        resid = tuple(yi - ai for yi, ai in zip(y, matvec(a, mu), strict=True))
        mean = tuple(m + g for m, g in zip(mu, matvec(gain, resid), strict=True))
        cov = add(c0, matmul(gain, matmul(a, c0)), scale=-1.0)
        for method in ("data_space", "parameter_space"):
            post = linear_gaussian_posterior(
                DenseOperator(a),
                y,
                noise=DenseCovariance(gamma),
                prior_mean=mu,
                prior_factor=DenseOperator(p),
                method=method,
            )
            self.assertLess(_diff(post.mean, mean), 1e-12)
            for j in range(5):
                unit = tuple(1.0 if i == j else 0.0 for i in range(5))
                column = tuple(cov[i][j] for i in range(5))
                self.assertLess(_diff(post.covariance_action(unit), column), 1e-12)
        # a direction in the null space of P' has zero prior and posterior variance
        post = linear_gaussian_posterior(
            DenseOperator(a),
            y,
            noise=DenseCovariance(gamma),
            prior_factor=DenseOperator(p),
        )
        # P' v = 0 for v = (0, 0, 0, 0, 0) is trivial; build a real null vector:
        # v = e_0 - ... solve P' v = 0 with v in span(e_3, e_4, e_2)
        null = self._null_vector(p)
        self.assertLess(max(abs(c) for c in post.covariance_action(null)), 1e-13)
        self.assertGreater(max(abs(c) for c in null), 0.5)

    def test_vanishing_noise_fits_observed_subspace(self) -> None:
        """Vanishing noise fits observed subspace."""
        post = linear_gaussian_posterior(
            A32,
            Y3[:3],
            noise=DiagonalCovariance((1e-12,) * 3),
            prior_covariance=DiagonalCovariance((1.0, 1.0)),
            method="parameter_space",
        )
        # 3 data, 2 unknowns: least-squares limit; residual of the exact LS fit
        normal = matmul(transpose(A32), A32)
        ls = matvec(inverse_spd(normal), matvec(transpose(A32), Y3))
        self.assertLess(_diff(post.mean, ls), 1e-8)
        # the posterior variances collapse by the noise scale
        self.assertLess(max(post.marginal_variances()), 1e-11)


class ConditionalSimulationTests(unittest.TestCase):
    """Exact sampling within Monte Carlo error bands."""

    def test_sample_moments_within_mcse_bands(self) -> None:
        """Sample moments within mcse bands."""
        a = ((1.0, 0.5), (-0.3, 2.0))
        gamma = ((0.5, 0.1), (0.1, 0.8))
        c0 = ((1.2, 0.4), (0.4, 0.7))
        mu = (0.3, -0.5)
        y = (1.0, -0.4)
        mean, cov, _ = _dense_oracle(a, gamma, c0, mu, y)
        count = 4000
        for method in ("data_space", "parameter_space"):
            post = linear_gaussian_posterior(
                a,
                y,
                noise=DenseCovariance(gamma),
                prior_mean=mu,
                prior_covariance=DenseCovariance(c0),
                method=method,
            )
            draws = post.sample(count, seed=20240607)
            self.assertEqual(draws[:5], post.sample(5, seed=20240607))
            sample_mean = [sum(d[i] for d in draws) / count for i in range(2)]
            for i in range(2):
                band = 5.0 * math.sqrt(cov[i][i] / count)
                self.assertLess(abs(sample_mean[i] - mean[i]), band)
            for i in range(2):
                for j in range(i, 2):
                    sample_cov = sum(
                        (d[i] - sample_mean[i]) * (d[j] - sample_mean[j]) for d in draws
                    ) / (count - 1)
                    # Var of a sample covariance entry of a Gaussian
                    se = math.sqrt((cov[i][i] * cov[j][j] + cov[i][j] ** 2) / count)
                    self.assertLess(abs(sample_cov - cov[i][j]), 5.0 * se)


class LowRankTests(unittest.TestCase):
    """Low-rank posterior against the exact solution."""

    @staticmethod
    def _problem():
        n, m, rank = 30, 12, 5
        a = [[0.0] * n for _ in range(m)]
        for k in range(rank):
            weight = 3.0 / (k + 1)
            for i in range(m):
                for j in range(n):
                    a[i][j] += (
                        weight
                        * math.sin(0.7 * (i + 1) * (k + 1))
                        * math.cos(0.3 * (j + 1) * (k + 2) + k)
                    )
        a_matrix = tuple(tuple(row) for row in a)
        prior = DiagonalCovariance(tuple(1.0 + 0.1 * (j % 7) for j in range(n)))
        noise = DiagonalCovariance(tuple(0.5 + 0.05 * i for i in range(m)))
        y = tuple(math.cos(1.3 * i) for i in range(m))
        mu = tuple(0.1 * math.sin(j) for j in range(n))
        return a_matrix, prior, noise, y, mu

    def test_all_nonzero_modes_equals_exact(self) -> None:
        """All nonzero modes equals exact."""
        a, prior, noise, y, mu = self._problem()
        exact = linear_gaussian_posterior(
            DenseOperator(a),
            y,
            noise=noise,
            prior_mean=mu,
            prior_covariance=prior,
            method="parameter_space",
        )
        probes = [
            tuple(math.sin(0.9 * (j + 1) * (q + 1)) for j in range(30))
            for q in range(3)
        ]
        for method, rank in (("randomized", 5), ("randomized", 7), ("lanczos", 5)):
            low = low_rank_posterior(
                DenseOperator(a),
                y,
                noise=noise,
                prior_mean=mu,
                prior_factor=prior,
                rank=rank,
                seed=3,
                method=method,
            )
            self.assertLess(_diff(low.mean, exact.mean), 1e-8, (method, rank))
            for probe in probes:
                reference = exact.covariance_action(probe)
                scale = max(abs(v) for v in reference)
                self.assertLess(
                    _diff(low.covariance_action(probe), reference) / scale,
                    1e-8,
                    (method, rank),
                )
        info = prior_whitened_information(DenseOperator(a), noise, prior)
        low = low_rank_posterior(
            DenseOperator(a), y, noise=noise, prior_factor=prior, rank=5, seed=1
        )
        self.assertAlmostEqual(low.information_gain, info.mutual_information, 8)
        self.assertAlmostEqual(low.effective_dimension, info.effective_dimension, 8)
        # information gain = 1/2 (log det S - log det Gamma)
        self.assertAlmostEqual(
            info.mutual_information,
            0.5 * (exact.log_det_data_covariance - noise.log_det()),
            places=9,
        )
        self.assertEqual(len(info.eigenvalues), 30)
        self.assertEqual(sum(1 for v in info.eigenvalues if v > 1e-9), 5)
        self.assertAlmostEqual(info.shrinkage_factors[-1], 1.0, places=12)

    def test_truncation_preserves_prior_in_unresolved_directions(self) -> None:
        """Truncation preserves prior in unresolved directions."""
        a = (
            (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        )
        noise = DiagonalCovariance((1.0, 1.0, 1.0))
        prior = DiagonalCovariance((1.0,) * 6)
        y = (1.0, 1.0, 1.0)
        low = low_rank_posterior(
            DenseOperator(a), y, noise=noise, prior_factor=prior, rank=2
        )
        e = [tuple(1.0 if i == j else 0.0 for i in range(6)) for j in range(6)]
        self.assertAlmostEqual(low.eigenvalues[0], 9.0, places=9)
        self.assertAlmostEqual(low.eigenvalues[1], 4.0, places=9)
        self.assertAlmostEqual(low.smallest_retained_eigenvalue, 4.0, places=9)
        variances = [low.covariance_action(v)[j] for j, v in enumerate(e)]
        self.assertAlmostEqual(variances[0], 1.0 / 10.0, places=9)
        self.assertAlmostEqual(variances[1], 1.0 / 5.0, places=9)
        # unresolved: weakest observed direction and unobserved ones keep prior
        for j in range(2, 6):
            self.assertAlmostEqual(variances[j], 1.0, places=9)
        # the exact variance of direction 2 is smaller (truncation is conservative)
        full = low_rank_posterior(
            DenseOperator(a), y, noise=noise, prior_factor=prior, rank=3
        )
        self.assertAlmostEqual(full.covariance_action(e[2])[2], 0.5, places=9)
        self.assertLess(low.information_gain, full.information_gain)

    def test_low_rank_samples_match_covariance_action(self) -> None:
        """Low rank samples match covariance action."""
        a = (
            (3.0, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
        )
        low = low_rank_posterior(
            DenseOperator(a),
            (1.0, -1.0),
            noise=DiagonalCovariance((1.0, 1.0)),
            prior_factor=DiagonalCovariance((1.0,) * 4),
            rank=2,
            method="lanczos",
            seed=5,
        )
        count = 4000
        draws = low.sample(count, seed=11)
        for i, expected in enumerate((0.1, 0.2, 1.0, 1.0)):
            mean = sum(d[i] for d in draws) / count
            variance = sum((d[i] - mean) ** 2 for d in draws) / (count - 1)
            self.assertLess(
                abs(variance - expected), 5.0 * expected * math.sqrt(2.0 / count)
            )
        self.assertAlmostEqual(
            sum(d[0] for d in draws) / count, low.mean[0], delta=5 * 0.32 / 63
        )


class TikhonovEquivalenceTests(unittest.TestCase):
    """The Gaussian posterior mean is the Tikhonov solution."""

    def test_posterior_mean_equals_tikhonov(self) -> None:
        """Posterior mean equals tikhonov."""
        a = (
            (1.0, 0.3, 0.0),
            (0.2, 1.0, 0.4),
            (0.0, 0.5, 1.0),
            (1.0, -1.0, 0.5),
            (0.3, 0.2, -0.6),
        )
        lower = ((1.5, 0.0, 0.0), (0.4, 1.1, 0.0), (-0.2, 0.3, 0.9))
        lam = 0.7
        y = (1.0, 0.5, -0.3, 0.2, 0.8)
        inv_columns = [
            lu_solve(lower, tuple(1.0 if i == j else 0.0 for i in range(3)))
            for j in range(3)
        ]
        factor = tuple(
            tuple(inv_columns[j][i] / lam for j in range(3)) for i in range(3)
        )  # P = L^-1 / lambda, C0 = (lambda^2 L'L)^-1
        c0 = matmul(factor, transpose(factor))
        reference = tikhonov(a, y, lam, penalty=lower).solution
        for method in ("data_space", "parameter_space"):
            via_factor = linear_gaussian_posterior(
                DenseOperator(a),
                y,
                noise=DiagonalCovariance((1.0,) * 5),
                prior_factor=DenseOperator(factor),
                method=method,
            )
            self.assertLess(_diff(via_factor.mean, reference), 1e-10)
        via_cov = linear_gaussian_posterior(
            a,
            y,
            noise=DiagonalCovariance((1.0,) * 5),
            prior_covariance=DenseCovariance(c0),
        )
        self.assertLess(_diff(via_cov.mean, reference), 1e-10)
        # posterior covariance is (A'A + lambda^2 L'L)^-1
        penalty = matmul(transpose(lower), lower)
        precision = add(matmul(transpose(a), a), penalty, scale=lam * lam)
        expected = inverse_spd(precision)
        self.assertLess(
            _diff(via_cov.covariance_action((1.0, 0.0, 0.0)), expected[0]), 1e-10
        )


class InvalidInputTests(unittest.TestCase):
    """Adversarial and malformed inputs."""

    def setUp(self) -> None:
        """Build shared valid inputs."""
        self.noise = DiagonalCovariance((1.0, 1.0))
        self.prior = DiagonalCovariance((1.0, 1.0, 1.0))
        self.a = DenseOperator(A23)

    def test_dimension_mismatch(self) -> None:
        """Dimension mismatch."""
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a, (1.0,), noise=self.noise, prior_covariance=self.prior
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=DiagonalCovariance((1.0,) * 3),
                prior_covariance=self.prior,
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=self.noise,
                prior_covariance=DiagonalCovariance((1.0, 1.0)),
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=self.noise,
                prior_covariance=self.prior,
                prior_mean=(0.0,),
            )

    def test_non_spd_covariance_rejected(self) -> None:
        """Non spd covariance rejected."""
        with self.assertRaises(ValueError):
            DenseCovariance(((1.0, 2.0), (2.0, 1.0)))
        with self.assertRaises(ValueError):
            DiagonalCovariance((1.0, 0.0))

    def test_type_and_option_errors(self) -> None:
        """Type and option errors."""
        with self.assertRaises(TypeError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=((1.0, 0.0), (0.0, 1.0)),
                prior_covariance=self.prior,
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(self.a, (1.0, 2.0), noise=self.noise)
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=self.noise,
                prior_covariance=self.prior,
                prior_factor=DenseOperator(C3),
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=self.noise,
                prior_covariance=self.prior,
                method="bogus",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, 2.0),
                noise=self.noise,
                prior_covariance=self.prior,
                max_dense_dimension=1,
            )
        with self.assertRaises(ValueError):
            linear_gaussian_posterior(
                self.a,
                (1.0, float("nan")),
                noise=self.noise,
                prior_covariance=self.prior,
            )
        post = linear_gaussian_posterior(
            self.a, (1.0, 2.0), noise=self.noise, prior_covariance=self.prior
        )
        with self.assertRaises(ValueError):
            post.sample(0)
        with self.assertRaises(TypeError):
            post.sample(2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            post.covariance_action((1.0, 2.0))
        with self.assertRaises(ValueError):
            low_rank_posterior(
                self.a, (1.0, 2.0), noise=self.noise, prior_factor=self.prior, rank=3
            )
        with self.assertRaises(ValueError):
            low_rank_posterior(
                self.a,
                (1.0, 2.0),
                noise=self.noise,
                prior_factor=self.prior,
                rank=1,
                method="bogus",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
