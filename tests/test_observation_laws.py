"""Tests of the observation laws and the Fisher-scoring estimator.

Oracles are independent of the implementation: hand-evaluated 2x2 and 3x3
Gaussian densities, ``probstat.distributions`` mass and density functions,
``statistics.NormalDist``, asymptotic tail series, quadrature of the expected
squared score, a direct Newton solve and ``solve_regularized``.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable, Sequence
from statistics import NormalDist

from probstat.distributions import Binomial, Poisson, StudentT
from probstat.inverse.linear_problem import LinearInverseProblem, solve_regularized
from probstat.inverse.observation_laws import (
    BinomialLaw,
    CensoredGaussianLaw,
    GaussianLaw,
    MissingMask,
    ObservationLaw,
    PoissonLaw,
    StudentTLaw,
    maximum_likelihood_linear,
)
from probstat.linalg import (
    BlockDiagonalCovariance,
    DenseCovariance,
    DiagonalCovariance,
    inverse_spd,
)

LOG_2PI = math.log(2.0 * math.pi)
STD = NormalDist()


def central_difference(
    law: ObservationLaw,
    y: Sequence[float],
    eta: Sequence[float],
    step: float = 1e-6,
) -> list[float]:
    """Return the central finite-difference gradient of ``log_likelihood``."""
    grad = []
    for i in range(len(eta)):
        up = list(eta)
        down = list(eta)
        up[i] += step
        down[i] -= step
        grad.append(
            (law.log_likelihood(y, up) - law.log_likelihood(y, down)) / (2 * step)
        )
    return grad


def simpson(f: Callable[[float], float], lo: float, hi: float, n: int = 4000) -> float:
    """Composite Simpson rule with ``n`` (even) panels."""
    h = (hi - lo) / n
    total = f(lo) + f(hi)
    for k in range(1, n):
        total += (4 if k % 2 else 2) * f(lo + k * h)
    return total * h / 3.0


class GaussianLawTests(unittest.TestCase):
    """Normalization, correlation, score and simulation of the Gaussian law."""

    def test_two_by_two_correlated_density_by_hand(self) -> None:
        """Match the hand-expanded bivariate normal log-density."""
        g11, g12, g22 = 2.0, 0.6, 1.0
        law = GaussianLaw(DenseCovariance([[g11, g12], [g12, g22]]))
        y, eta = (1.5, -0.5), (1.0, 0.5)
        r1, r2 = y[0] - eta[0], y[1] - eta[1]
        det = g11 * g22 - g12 * g12
        quad = (g22 * r1 * r1 - 2 * g12 * r1 * r2 + g11 * r2 * r2) / det
        expected = -0.5 * (quad + math.log(det) + 2 * LOG_2PI)
        self.assertAlmostEqual(law.log_likelihood(y, eta), expected, places=13)

    def test_block_diagonal_equals_sum_of_blocks(self) -> None:
        """A block-diagonal covariance gives the sum of the block densities."""
        b1 = DenseCovariance([[2.0, 0.6], [0.6, 1.0]])
        b2 = DiagonalCovariance([0.5])
        full = GaussianLaw(BlockDiagonalCovariance([b1, b2]))
        y, eta = (1.5, -0.5, 0.2), (1.0, 0.5, -0.1)
        parts = GaussianLaw(b1).log_likelihood(y[:2], eta[:2]) + GaussianLaw(
            b2
        ).log_likelihood(y[2:], eta[2:])
        self.assertAlmostEqual(full.log_likelihood(y, eta), parts, places=13)

    def test_score_is_finite_difference_of_log_likelihood(self) -> None:
        """The score equals the central difference of the log-likelihood."""
        law = GaussianLaw(DenseCovariance([[2.0, 0.6], [0.6, 1.0]]))
        y, eta = (1.5, -0.5), (1.0, 0.5)
        fd = central_difference(law, y, eta)
        for a, b in zip(law.score(y, eta), fd, strict=True):
            self.assertAlmostEqual(a, b, places=7)

    def test_fisher_information_is_precision_diagonal(self) -> None:
        """The diagonal information is diag(Gamma^-1)."""
        cov = [[2.0, 0.6], [0.6, 1.0]]
        law = GaussianLaw(DenseCovariance(cov))
        precision = inverse_spd(cov)
        info = law.fisher_information((0.0, 0.0))
        self.assertAlmostEqual(info[0], precision[0][0], places=12)
        self.assertAlmostEqual(info[1], precision[1][1], places=12)

    def test_simulation_moments(self) -> None:
        """Simulated mean and covariance match within Monte Carlo bands."""
        cov = [[2.0, 0.6], [0.6, 1.0]]
        law = GaussianLaw(DenseCovariance(cov))
        eta = (1.0, -2.0)
        n = 4000
        draws = [law.simulate(eta, seed=11 + k) for k in range(n)]
        for j in range(2):
            mean = sum(d[j] for d in draws) / n
            self.assertLess(abs(mean - eta[j]), 4 * math.sqrt(cov[j][j] / n))
        c01 = sum((d[0] - eta[0]) * (d[1] - eta[1]) for d in draws) / n
        mcse = math.sqrt((cov[0][0] * cov[1][1] + cov[0][1] ** 2) / n)
        self.assertLess(abs(c01 - cov[0][1]), 4 * mcse)

    def test_not_a_covariance_factor_rejected(self) -> None:
        """A raw matrix is not a covariance factor."""
        with self.assertRaises(TypeError):
            GaussianLaw([[1.0]])  # type: ignore[arg-type]

    def test_length_mismatch_rejected(self) -> None:
        """Mismatched lengths raise."""
        law = GaussianLaw(DiagonalCovariance([1.0, 1.0]))
        with self.assertRaises(ValueError):
            law.log_likelihood((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            law.log_likelihood((1.0, 2.0), (1.0, 2.0, 3.0))


class MissingMaskTests(unittest.TestCase):
    """Masks use the marginal submatrix, not a sub-selection of the precision."""

    GAMMA = [[2.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.5]]

    def test_marginal_submatrix_three_by_three(self) -> None:
        """Observed entries (0, 2) use [[2, .2], [.2, 1.5]] by hand."""
        law = MissingMask(GaussianLaw(DenseCovariance(self.GAMMA)), [True, False, True])
        y = (1.0, math.nan, -0.5)
        eta = (0.2, 9.0, 0.1)
        r1, r2 = 0.8, -0.6
        g11, g12, g22 = 2.0, 0.2, 1.5
        det = g11 * g22 - g12 * g12
        quad = (g22 * r1 * r1 - 2 * g12 * r1 * r2 + g11 * r2 * r2) / det
        expected = -0.5 * (quad + math.log(det) + 2 * LOG_2PI)
        self.assertAlmostEqual(law.log_likelihood(y, eta), expected, places=12)
        # The sub-selection of the precision is a different (conditional) law.
        prec = inverse_spd(self.GAMMA)
        p11, p12, p22 = prec[0][0], prec[0][2], prec[2][2]
        wrong_quad = p11 * r1 * r1 + 2 * p12 * r1 * r2 + p22 * r2 * r2
        wrong_det = p11 * p22 - p12 * p12
        wrong = -0.5 * (wrong_quad - math.log(wrong_det) + 2 * LOG_2PI)
        self.assertGreater(abs(wrong - expected), 1e-3)

    def test_score_and_information_zero_at_missing(self) -> None:
        """Masked entries get zero score and information; simulate gives nan."""
        law = MissingMask(PoissonLaw("log"), [True, False, True])
        y = (2.0, math.nan, 0.0)
        eta = (0.5, 0.1, -0.3)
        self.assertEqual(law.score(y, eta)[1], 0.0)
        self.assertEqual(law.fisher_information(eta)[1], 0.0)
        self.assertTrue(math.isnan(law.simulate(eta, seed=3)[1]))
        fd = central_difference(law, (2.0, 0.0, 0.0), eta)
        for a, b in zip(law.score(y, eta), fd, strict=True):
            self.assertAlmostEqual(a, b, places=6)

    def test_all_masked_and_bad_masks_raise(self) -> None:
        """No observed entry, or a wrong-length mask, is an error."""
        gauss = GaussianLaw(DiagonalCovariance([1.0, 1.0]))
        with self.assertRaises(ValueError):
            MissingMask(gauss, [False, False])
        with self.assertRaises(ValueError):
            MissingMask(gauss, [True, True, True])
        with self.assertRaises(TypeError):
            MissingMask(gauss, [1, 0])  # type: ignore[list-item]


class CountLawTests(unittest.TestCase):
    """Poisson and binomial laws against probstat.distributions."""

    def test_poisson_identity_exact_small_probabilities(self) -> None:
        """Match closed forms and the distribution mass function."""
        law = PoissonLaw("identity")
        y, mu = (0.0, 3.0, 7.0), (0.5, 2.0, 6.0)
        expected = -0.5 + (3 * math.log(2.0) - 2.0 - math.log(6.0))
        expected += 7 * math.log(6.0) - 6.0 - math.log(5040.0)
        self.assertAlmostEqual(law.log_likelihood(y, mu), expected, places=12)
        ref = sum(Poisson(m).log_pmf(int(k)) for k, m in zip(y, mu, strict=True))
        self.assertAlmostEqual(law.log_likelihood(y, mu), ref, places=12)

    def test_poisson_log_link_matches_identity(self) -> None:
        """exp(eta) under the log link reproduces the identity-link value."""
        y, mu = (0.0, 3.0, 7.0), (0.5, 2.0, 6.0)
        eta = tuple(math.log(m) for m in mu)
        self.assertAlmostEqual(
            PoissonLaw("log").log_likelihood(y, eta),
            PoissonLaw("identity").log_likelihood(y, mu),
            places=12,
        )

    def test_poisson_extreme_rates_without_overflow(self) -> None:
        """Tiny and huge rates give finite values and scores."""
        log = PoissonLaw("log")
        self.assertAlmostEqual(log.log_likelihood((0.0,), (-700.0,)), -math.exp(-700.0))
        value = log.log_likelihood((5.0,), (-700.0,))
        self.assertAlmostEqual(value, -3500.0 - math.lgamma(6.0), places=9)
        self.assertTrue(math.isfinite(log.log_likelihood((900.0,), (700.0,))))
        ident = PoissonLaw("identity")
        self.assertAlmostEqual(ident.log_likelihood((0.0,), (1e-300,)), -1e-300)
        near_zero = ident.log_likelihood((1.0,), (1e-300,))
        self.assertAlmostEqual(near_zero, math.log(1e-300), places=9)
        values = ident.score((0.0,), (1e-300,)) + ident.fisher_information((1e-300,))
        for value in values:
            self.assertFalse(math.isnan(value))

    def test_poisson_score_fisher_and_simulation(self) -> None:
        """Score = finite difference, information = E[score^2], moments match."""
        for link, eta in (("identity", (0.7, 2.5, 4.0)), ("log", (-0.3, 0.9, 1.5))):
            law = PoissonLaw(link)
            y = (0.0, 3.0, 5.0)
            fd = central_difference(law, y, eta)
            for a, b in zip(law.score(y, eta), fd, strict=True):
                self.assertAlmostEqual(a, b, places=6)
            for e, info in zip(eta, law.fisher_information(eta), strict=True):
                mu = e if link == "identity" else math.exp(e)
                total = 0.0
                for k in range(0, 80):
                    s = law.score((float(k),), (e,))[0]
                    total += Poisson(mu).pmf(k) * s * s
                self.assertAlmostEqual(total, info, places=9)
        law = PoissonLaw("identity")
        mu = (0.8, 12.0)
        n = 3000
        draws = [law.simulate(mu, seed=100 + k) for k in range(n)]
        for j in range(2):
            mean = sum(d[j] for d in draws) / n
            self.assertLess(abs(mean - mu[j]), 4 * math.sqrt(mu[j] / n))
            var = sum((d[j] - mean) ** 2 for d in draws) / (n - 1)
            self.assertLess(abs(var - mu[j]), 6 * mu[j] * math.sqrt(2.0 / n) * 1.5)

    def test_binomial_matches_distribution_and_extremes(self) -> None:
        """Match Binomial.log_pmf; stay finite at |eta| = 800."""
        law = BinomialLaw([5, 10, 1])
        y, eta = (2.0, 10.0, 0.0), (0.3, -1.0, 4.0)
        ref = 0.0
        for k, n, e in zip(y, (5, 10, 1), eta, strict=True):
            ref += Binomial(n, 1.0 / (1.0 + math.exp(-e))).log_pmf(int(k))
        self.assertAlmostEqual(law.log_likelihood(y, eta), ref, places=11)
        big = BinomialLaw([7])
        self.assertAlmostEqual(big.log_likelihood((7.0,), (800.0,)), 0.0, places=12)
        self.assertAlmostEqual(big.log_likelihood((0.0,), (-800.0,)), 0.0, places=12)
        self.assertAlmostEqual(big.log_likelihood((0.0,), (800.0,)), -5600.0, places=6)
        self.assertEqual(big.fisher_information((800.0,))[0], 0.0)

    def test_binomial_score_fisher_simulation(self) -> None:
        """Score, expected information and moments of the binomial law."""
        trials = [5, 12]
        law = BinomialLaw(trials)
        eta = (0.4, -1.2)
        y = (3.0, 2.0)
        fd = central_difference(law, y, eta)
        for a, b in zip(law.score(y, eta), fd, strict=True):
            self.assertAlmostEqual(a, b, places=6)
        for n, e, info in zip(trials, eta, law.fisher_information(eta), strict=True):
            p = 1.0 / (1.0 + math.exp(-e))
            total = sum(Binomial(n, p).pmf(k) * (k - n * p) ** 2 for k in range(n + 1))
            self.assertAlmostEqual(total, info, places=10)
        count = 3000
        draws = [law.simulate(eta, seed=7 + k) for k in range(count)]
        for j, (n, e) in enumerate(zip(trials, eta, strict=True)):
            p = 1.0 / (1.0 + math.exp(-e))
            mean = sum(d[j] for d in draws) / count
            self.assertLess(abs(mean - n * p), 4 * math.sqrt(n * p * (1 - p) / count))

    def test_invalid_inputs_raise(self) -> None:
        """Negative counts, eta <= 0, wrong links and trial violations raise."""
        with self.assertRaises(ValueError):
            PoissonLaw("identity").log_likelihood((-1.0,), (1.0,))
        with self.assertRaises(ValueError):
            PoissonLaw("identity").log_likelihood((1.5,), (1.0,))
        with self.assertRaises(ValueError):
            PoissonLaw("identity").log_likelihood((1.0,), (0.0,))
        with self.assertRaises(ValueError):
            PoissonLaw("identity").log_likelihood((1.0,), (-2.0,))
        with self.assertRaises(ValueError):
            PoissonLaw("log").log_likelihood((1.0,), (800.0,))
        with self.assertRaises(ValueError):
            PoissonLaw("sqrt")
        with self.assertRaises(ValueError):
            PoissonLaw("log").log_likelihood((1.0, 2.0), (0.0,))
        with self.assertRaises(ValueError):
            BinomialLaw([5]).log_likelihood((6.0,), (0.0,))
        with self.assertRaises(ValueError):
            BinomialLaw([5], link="probit")
        with self.assertRaises(ValueError):
            BinomialLaw([-1])


class StudentTLawTests(unittest.TestCase):
    """Independent Student-t errors."""

    def test_matches_student_t_distribution(self) -> None:
        """Sum of univariate StudentT log-densities."""
        scale = (0.5, 2.0, 1.0)
        law = StudentTLaw(scale, 3.5)
        y, eta = (0.3, -1.0, 4.0), (0.0, 1.0, 1.0)
        ref = sum(
            StudentT(3.5, e, s).log_pdf(v)
            for v, e, s in zip(y, eta, scale, strict=True)
        )
        self.assertAlmostEqual(law.log_likelihood(y, eta), ref, places=12)

    def test_score_fisher_and_weights(self) -> None:
        """Score = finite difference; information = E[score^2] by quadrature."""
        nu, s = 5.0, 1.7
        law = StudentTLaw((s,), nu)
        y, eta = (1.1,), (0.2,)
        fd = central_difference(law, y, eta)
        self.assertAlmostEqual(law.score(y, eta)[0], fd[0], places=7)
        r = 0.9
        self.assertAlmostEqual(
            law.irls_weights((r,), (0.0,))[0], (nu + 1) / (nu * s * s + r * r)
        )
        dist = StudentT(nu, 0.0, s)

        def integrand(theta: float) -> float:
            t = s * math.tan(theta)
            score = law.score((t,), (0.0,))[0]
            return dist.pdf(t) * score * score * s / math.cos(theta) ** 2

        limit = 0.5 * math.pi * (1 - 1e-9)
        total = simpson(integrand, -limit, limit, 4000)
        self.assertAlmostEqual(total, law.fisher_information((0.0,))[0], places=5)

    def test_simulation_variance(self) -> None:
        """Sample variance approaches s^2 nu/(nu-2) for nu = 8."""
        nu, s = 8.0, 1.5
        law = StudentTLaw((s,), nu)
        n = 5000
        draws = [law.simulate((0.0,), seed=500 + k)[0] for k in range(n)]
        var = sum(d * d for d in draws) / n
        theory = s * s * nu / (nu - 2.0)
        kappa = 6.0 / (nu - 4.0)
        mcse = theory * math.sqrt((2.0 + kappa) / n)
        self.assertLess(abs(var - theory), 5 * mcse)

    def test_invalid_inputs(self) -> None:
        """A non-positive nu, non-positive scales and length mismatches raise."""
        with self.assertRaises(ValueError):
            StudentTLaw((1.0,), 0.0)
        with self.assertRaises(ValueError):
            StudentTLaw((1.0,), -2.0)
        with self.assertRaises(ValueError):
            StudentTLaw((0.0,), 3.0)
        with self.assertRaises(ValueError):
            StudentTLaw((1.0, 1.0), 3.0).log_likelihood((1.0,), (1.0,))


class CensoredLawTests(unittest.TestCase):
    """Detection limits via stable log-CDF and log-survival."""

    def test_matches_normal_cdf_closed_form(self) -> None:
        """Left, right and exact entries against statistics.NormalDist."""
        law = CensoredGaussianLaw(
            (1.0, 2.0, 0.5), lower=(0.0, None, None), upper=(None, 3.0, None)
        )
        y, eta = (0.0, 3.0, 0.4), (0.5, 1.0, 0.1)
        expected = math.log(STD.cdf((0.0 - 0.5) / 1.0))
        expected += math.log(1.0 - STD.cdf((3.0 - 1.0) / 2.0))
        expected += math.log(NormalDist(0.1, 0.5).pdf(0.4))
        self.assertAlmostEqual(law.log_likelihood(y, eta), expected, places=12)

    def test_extreme_tails_without_overflow(self) -> None:
        """Forty-sd censoring matches the asymptotic Mills series."""
        z = 40.0
        law = CensoredGaussianLaw((1.0,), lower=(-z,), upper=None)
        series = (
            -0.5 * z * z
            - math.log(z)
            - 0.5 * LOG_2PI
            + math.log(1 - 1 / z**2 + 3 / z**4 - 15 / z**6)
        )
        self.assertAlmostEqual(law.log_likelihood((-z,), (0.0,)), series, places=9)
        right = CensoredGaussianLaw((1.0,), lower=None, upper=(z,))
        self.assertAlmostEqual(right.log_likelihood((z,), (0.0,)), series, places=9)
        far = law.log_likelihood((-z,), (1000.0,))
        self.assertTrue(math.isfinite(far))
        for value in law.score((-z,), (0.0,)) + law.score((-z,), (1000.0,)):
            self.assertTrue(math.isfinite(value))
        # The score at 40 sd follows the inverse Mills ratio, about -z.
        self.assertAlmostEqual(law.score((-z,), (0.0,))[0], -z, delta=0.05)

    def test_score_matches_finite_differences(self) -> None:
        """Left, right and exact entries together."""
        law = CensoredGaussianLaw(
            (1.0, 2.0, 0.5), lower=(0.0, None, None), upper=(None, 3.0, None)
        )
        y, eta = (0.0, 3.0, 0.4), (0.5, 1.0, 0.1)
        fd = central_difference(law, y, eta)
        for a, b in zip(law.score(y, eta), fd, strict=True):
            self.assertAlmostEqual(a, b, places=6)

    def test_fisher_information_is_expected_squared_score(self) -> None:
        """Closed form against quadrature of the mixed observation."""
        s, lo, up, e = 1.3, -1.0, 2.0, 0.4
        law = CensoredGaussianLaw((s,), lower=(lo,), upper=(up,))
        dist = NormalDist(e, s)

        def exact_term(y: float) -> float:
            score = (y - e) / (s * s)
            return dist.pdf(y) * score * score

        interior = simpson(exact_term, lo, up, 4000)
        left = dist.cdf(lo) * law.score((lo,), (e,))[0] ** 2
        right = (1.0 - dist.cdf(up)) * law.score((up,), (e,))[0] ** 2
        self.assertAlmostEqual(
            law.fisher_information((e,))[0], interior + left + right, places=9
        )
        plain = CensoredGaussianLaw((s,))
        self.assertAlmostEqual(plain.fisher_information((e,))[0], 1.0 / (s * s))

    def test_simulation_censoring_fractions(self) -> None:
        """Recorded values sit on the limits with the Gaussian probabilities."""
        law = CensoredGaussianLaw((1.0,), lower=(-0.5,), upper=(1.0,))
        n = 4000
        draws = [law.simulate((0.3,), seed=900 + k)[0] for k in range(n)]
        p_left = STD.cdf(-0.8)
        p_right = 1.0 - STD.cdf(0.7)
        left = sum(d == -0.5 for d in draws) / n
        right = sum(d == 1.0 for d in draws) / n
        self.assertLess(abs(left - p_left), 4 * math.sqrt(p_left * (1 - p_left) / n))
        self.assertLess(
            abs(right - p_right), 4 * math.sqrt(p_right * (1 - p_right) / n)
        )

    def test_invalid_inputs(self) -> None:
        """Values beyond a limit, crossed limits and bad scales raise."""
        law = CensoredGaussianLaw((1.0,), lower=(0.0,))
        with self.assertRaises(ValueError):
            law.log_likelihood((-1.0,), (0.0,))
        with self.assertRaises(ValueError):
            CensoredGaussianLaw((1.0,), lower=(2.0,), upper=(1.0,))
        with self.assertRaises(ValueError):
            CensoredGaussianLaw((-1.0,))
        with self.assertRaises(ValueError):
            CensoredGaussianLaw((1.0, 1.0), lower=(0.0,))


class MaximumLikelihoodTests(unittest.TestCase):
    """Fisher scoring / IRLS estimator."""

    @staticmethod
    def design(n: int) -> list[list[float]]:
        """Return the intercept-slope design on [-1, 1]."""
        return [[1.0, -1.0 + 2.0 * i / (n - 1)] for i in range(n)]

    def test_poisson_log_link_recovers_coefficients_and_matches_newton(self) -> None:
        """Estimate within 4 asymptotic sd and equal to a direct Newton solve."""
        a = self.design(200)
        truth = (1.0, 0.8)
        law = PoissonLaw("log")
        eta = [r[0] * truth[0] + r[1] * truth[1] for r in a]
        y = law.simulate(eta, seed=2024)
        fit = maximum_likelihood_linear(a, y, law, tolerance=1e-15)
        self.assertTrue(fit.converged, fit.stop_reason)
        mu = [math.exp(e) for e in eta]
        i00 = sum(m * r[0] * r[0] for m, r in zip(mu, a, strict=True))
        i01 = sum(m * r[0] * r[1] for m, r in zip(mu, a, strict=True))
        i11 = sum(m * r[1] * r[1] for m, r in zip(mu, a, strict=True))
        det = i00 * i11 - i01 * i01
        sd = (math.sqrt(i11 / det), math.sqrt(i00 / det))
        for est, t, s in zip(fit.estimate, truth, sd, strict=True):
            self.assertLess(abs(est - t), 4 * s)
        # Direct 2-parameter Newton iteration written independently.
        b0, b1 = 0.0, 0.0
        for _ in range(60):
            m = [math.exp(b0 * r[0] + b1 * r[1]) for r in a]
            g0 = sum((yy - mm) * r[0] for yy, mm, r in zip(y, m, a, strict=True))
            g1 = sum((yy - mm) * r[1] for yy, mm, r in zip(y, m, a, strict=True))
            h00 = sum(mm * r[0] * r[0] for mm, r in zip(m, a, strict=True))
            h01 = sum(mm * r[0] * r[1] for mm, r in zip(m, a, strict=True))
            h11 = sum(mm * r[1] * r[1] for mm, r in zip(m, a, strict=True))
            d = h00 * h11 - h01 * h01
            b0 += (h11 * g0 - h01 * g1) / d
            b1 += (h00 * g1 - h01 * g0) / d
        self.assertAlmostEqual(fit.estimate[0], b0, places=7)
        self.assertAlmostEqual(fit.estimate[1], b1, places=7)
        self.assertLess(fit.gradient_norm, 1e-6)
        objectives = [h.objective for h in fit.history]
        self.assertTrue(
            all(b <= a_ for a_, b in zip(objectives, objectives[1:], strict=False))
        )

    def test_poisson_identity_link_mean(self) -> None:
        """A constant rate estimated under the identity link is the sample mean."""
        y = (2.0, 0.0, 5.0, 3.0, 1.0)
        fit = maximum_likelihood_linear(
            [[1.0]] * 5, y, PoissonLaw("identity"), initial=(1.0,)
        )
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.estimate[0], sum(y) / 5.0, places=8)
        with self.assertRaises(ValueError):
            maximum_likelihood_linear([[1.0]] * 5, y, PoissonLaw("identity"))

    def test_gaussian_law_reproduces_solve_regularized(self) -> None:
        """Weighted affine Tikhonov with correlated noise is reproduced."""
        a = [
            [1.0, 0.5, 0.0],
            [0.3, 1.0, 0.2],
            [0.0, 0.4, 1.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.0, 0.7],
            [0.5, 0.5, 0.1],
        ]
        y = (1.0, 0.2, -0.4, 1.5, 0.3, 0.9)
        cov = [
            [1.0 if i == j else 0.3 ** abs(i - j) for j in range(6)] for i in range(6)
        ]
        noise = DenseCovariance(cov)
        pen = [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]
        ref = (0.1, -0.2, 0.3)
        lam = 0.7
        problem = LinearInverseProblem.create(
            a, y, noise=noise, penalty=pen, reference=ref
        )
        exact = solve_regularized(problem, lam, method="qr")
        fit = maximum_likelihood_linear(
            a, y, GaussianLaw(noise), regularization=lam, penalty=pen, reference=ref
        )
        self.assertTrue(fit.converged)
        for u, v in zip(fit.estimate, exact.estimate, strict=True):
            self.assertAlmostEqual(u, v, places=8)
        self.assertLessEqual(fit.iterations, 2)
        self.assertLess(fit.gradient_norm, 1e-7)

    def test_student_t_resists_two_gross_outliers(self) -> None:
        """Constructed example only: not a universal claim about robustness."""
        xs = [float(i) for i in range(12)]
        wiggle = (0.05, -0.04, 0.03, -0.06, 0.02, 0.04)
        wiggle += (-0.03, 0.05, -0.02, 0.01, -0.05, 0.03)
        truth = (2.0, 0.5)
        y = [truth[0] + truth[1] * x + w for x, w in zip(xs, wiggle, strict=True)]
        y[3] += 25.0
        y[8] -= 25.0
        a = [[1.0, x] for x in xs]
        gauss = maximum_likelihood_linear(
            a, y, GaussianLaw(DiagonalCovariance([0.01] * 12))
        )
        robust = maximum_likelihood_linear(
            a, y, StudentTLaw((0.1,) * 12, 3.0), max_iterations=200
        )
        self.assertTrue(robust.converged, robust.stop_reason)
        err_t = max(abs(e - t) for e, t in zip(robust.estimate, truth, strict=True))
        err_g = max(abs(e - t) for e, t in zip(gauss.estimate, truth, strict=True))
        self.assertLess(err_t, 0.1)
        self.assertGreater(err_g, 1.0)
        objectives = [h.objective for h in robust.history]
        self.assertTrue(
            all(
                b <= a_ + 1e-12
                for a_, b in zip(objectives, objectives[1:], strict=False)
            )
        )

    def test_censored_mean_matches_brute_force(self) -> None:
        """A censored-Gaussian location estimate equals a golden-section search."""
        law = CensoredGaussianLaw([1.0] * 40, lower=[0.0] * 40)
        y = law.simulate([0.2] * 40, seed=77)
        self.assertGreater(sum(v == 0.0 for v in y), 5)
        fit = maximum_likelihood_linear([[1.0]] * 40, y, law, tolerance=1e-15)
        self.assertTrue(fit.converged, fit.stop_reason)

        def negative(m: float) -> float:
            return -law.log_likelihood(y, [m] * 40)

        lo, hi = -3.0, 3.0
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(100):
            c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
            if negative(c) < negative(d):
                hi = d
            else:
                lo = c
        self.assertAlmostEqual(fit.estimate[0], 0.5 * (lo + hi), places=6)
        info = sum(law.fisher_information([0.2] * 40))
        self.assertLess(abs(fit.estimate[0] - 0.2), 4 / math.sqrt(info))

    def test_masked_equals_dropped_rows(self) -> None:
        """Masking rows is identical to deleting them."""
        a = self.design(12)
        law = PoissonLaw("log")
        y = law.simulate([0.3 * r[1] + 0.5 for r in a], seed=5)
        observed = [i % 3 != 0 for i in range(12)]
        masked_y = [v if k else math.nan for v, k in zip(y, observed, strict=True)]
        fit_masked = maximum_likelihood_linear(a, masked_y, MissingMask(law, observed))
        rows = [r for r, k in zip(a, observed, strict=True) if k]
        kept = [v for v, k in zip(y, observed, strict=True) if k]
        fit_drop = maximum_likelihood_linear(rows, kept, law)
        for u, v in zip(fit_masked.estimate, fit_drop.estimate, strict=True):
            self.assertAlmostEqual(u, v, places=8)

    def test_masked_correlated_gaussian_uses_marginal(self) -> None:
        """Dropping an observation equals the GLS of the marginal submatrix."""
        a = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        y = (1.0, math.nan, 2.5)
        gamma = [[2.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.5]]
        law = MissingMask(GaussianLaw(DenseCovariance(gamma)), [True, False, True])
        fit = maximum_likelihood_linear(a, y, law)
        sub = DenseCovariance([[2.0, 0.2], [0.2, 1.5]])
        problem = LinearInverseProblem.create([a[0], a[2]], (1.0, 2.5), noise=sub)
        exact = solve_regularized(problem, 0.0, method="qr")
        for u, v in zip(fit.estimate, exact.estimate, strict=True):
            self.assertAlmostEqual(u, v, places=8)

    def test_invalid_arguments(self) -> None:
        """Bad dimensions, parameters and data raise."""
        a = [[1.0], [1.0]]
        law = PoissonLaw("log")
        with self.assertRaises(ValueError):
            maximum_likelihood_linear(a, (1.0,), law)
        with self.assertRaises(ValueError):
            maximum_likelihood_linear(a, (1.0, -2.0), law)
        with self.assertRaises(ValueError):
            maximum_likelihood_linear(a, (1.0, 2.0), law, regularization=-1.0)
        with self.assertRaises(ValueError):
            maximum_likelihood_linear(a, (1.0, 2.0), law, tolerance=0.0)
        with self.assertRaises(ValueError):
            maximum_likelihood_linear(
                a, (1.0, 2.0), GaussianLaw(DiagonalCovariance([1.0] * 3))
            )
        with self.assertRaises(TypeError):
            maximum_likelihood_linear(a, (1.0, 2.0), "poisson")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
