"""Tests of the extended observation laws.

Oracles are independent of the implementation: ``probstat.distributions``
(negative binomial, Poisson, Student-t, normal), ``probstat.multivariate``
(multivariate t), the existing laws of ``observation_laws``, composite Simpson
quadrature of the defining integrals, finite differences, direct summation of
probability mass functions and the dense routines of ``probstat.linalg``.
Stochastic tests use fixed seeds and bands of several Monte Carlo standard
errors.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from collections.abc import Callable, Sequence

from probstat.distributions import NegativeBinomial, Normal, Poisson, StudentT
from probstat.foundations.random_streams import resolve_stream
from probstat.inverse.observation_laws import (
    CensoredGaussianLaw,
    GaussianLaw,
    MissingMask,
    ObservationLaw,
    PoissonLaw,
    StudentTLaw,
    maximum_likelihood_linear,
)
from probstat.inverse.observation_laws_extended import (
    GaussianMixtureOutlierLaw,
    HeteroscedasticGaussianLaw,
    HuberLaw,
    IntervalCensoredGaussianLaw,
    MarginalScaleGaussianLaw,
    MultivariateStudentTLaw,
    NegativeBinomialLaw,
    ProfiledScaleGaussianLaw,
    StateDependentCovarianceLaw,
    huber_sandwich_covariance,
)
from probstat.linalg import (
    DenseCovariance,
    DiagonalCovariance,
    ScaledIdentityCovariance,
    inverse_spd,
    solve_spd,
)
from probstat.multivariate.student_t import MultivariateStudentT

LOG_2PI = math.log(2.0 * math.pi)
STD = Normal(0.0, 1.0)


def fd_gradient(
    law: ObservationLaw,
    y: Sequence[float],
    eta: Sequence[float],
    step: float = 1e-6,
) -> list[float]:
    """Return the central finite-difference gradient of ``log_likelihood``."""
    out = []
    for i in range(len(eta)):
        up, down = list(eta), list(eta)
        up[i] += step
        down[i] -= step
        out.append(
            (law.log_likelihood(y, up) - law.log_likelihood(y, down)) / (2 * step)
        )
    return out


def simpson(f: Callable[[float], float], lo: float, hi: float, n: int = 4000) -> float:
    """Composite Simpson rule with ``n`` (even) panels."""
    h = (hi - lo) / n
    total = f(lo) + f(hi)
    for k in range(1, n):
        total += (4 if k % 2 else 2) * f(lo + k * h)
    return total * h / 3.0


def close(
    test: unittest.TestCase, a: Sequence[float], b: Sequence[float], tol: float
) -> None:
    """Assert two vectors agree entrywise within ``tol``."""
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b, strict=True):
        test.assertAlmostEqual(x, y, delta=tol)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return sample mean and standard deviation."""
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# 1. Negative binomial
# ---------------------------------------------------------------------------


class NegativeBinomialTests(unittest.TestCase):
    """Normalization, oracle agreement, Poisson limit and scoring."""

    def test_matches_distribution_module_for_integer_dispersion(self) -> None:
        """Integer theta: NB(mu, theta) is NegativeBinomial(theta, theta/(theta+mu))."""
        for theta in (1, 2, 5, 30):
            law = NegativeBinomialLaw(theta)
            for mu in (0.05, 0.7, 3.0, 40.0):
                p = theta / (theta + mu)
                for k in (0, 1, 4, 25):
                    expected = NegativeBinomial(theta, p).log_pmf(k)
                    got = law.log_likelihood([k], [math.log(mu)])
                    self.assertAlmostEqual(
                        got, expected, delta=1e-11 * (1 + abs(expected))
                    )

    def test_normalization_mean_and_variance_for_real_dispersion(self) -> None:
        """The pmf sums to one with mean mu and variance mu + mu^2/theta."""
        theta, mu = 2.5, 3.0
        law = NegativeBinomialLaw(theta)
        pmf = [math.exp(law.log_likelihood([k], [math.log(mu)])) for k in range(400)]
        self.assertAlmostEqual(math.fsum(pmf), 1.0, delta=1e-12)
        mean = math.fsum(k * p for k, p in enumerate(pmf))
        var = math.fsum((k - mean) ** 2 * p for k, p in enumerate(pmf))
        self.assertAlmostEqual(mean, mu, delta=1e-10)
        self.assertAlmostEqual(var, mu + mu * mu / theta, delta=1e-9)

    def test_poisson_limit(self) -> None:
        """Theta -> infinity reproduces PoissonLaw (log link); math.inf is exact."""
        y, eta = [0.0, 3.0, 9.0], [math.log(0.4), math.log(2.5), math.log(8.0)]
        reference = PoissonLaw("log").log_likelihood(y, eta)
        self.assertAlmostEqual(
            NegativeBinomialLaw(math.inf).log_likelihood(y, eta), reference, delta=1e-13
        )
        gaps = [
            abs(NegativeBinomialLaw(t).log_likelihood(y, eta) - reference)
            for t in (1e2, 1e4, 1e6, 1e9)
        ]
        self.assertTrue(all(a > b for a, b in zip(gaps, gaps[1:], strict=False)))
        self.assertLess(gaps[-1], 1e-7)
        close(
            self,
            NegativeBinomialLaw(math.inf).score(y, eta),
            PoissonLaw("log").score(y, eta),
            1e-13,
        )
        close(
            self,
            NegativeBinomialLaw(math.inf).fisher_information(eta),
            PoissonLaw("log").fisher_information(eta),
            1e-13,
        )

    def test_score_matches_finite_difference_and_information_is_score_variance(
        self,
    ) -> None:
        """Score = FD gradient; I = E[score^2] by exact summation of the pmf."""
        law = NegativeBinomialLaw(1.7)
        y, eta = [2.0, 0.0, 11.0], [0.3, -1.2, 2.0]
        close(self, law.score(y, eta), fd_gradient(law, y, eta), 1e-6)
        for e in (-1.0, 0.4, 2.2):
            total = math.fsum(
                math.exp(law.log_likelihood([k], [e])) * law.score([k], [e])[0] ** 2
                for k in range(1500)
            )
            self.assertAlmostEqual(total, law.fisher_information([e])[0], delta=1e-8)

    def test_extreme_rates_do_not_overflow_or_cancel(self) -> None:
        """Rates near the bounds stay finite and agree with the closed form."""
        law = NegativeBinomialLaw(3.0)
        self.assertTrue(math.isfinite(law.log_likelihood([0.0], [-700.0])))
        self.assertTrue(math.isfinite(law.log_likelihood([5.0], [-700.0])))
        # For mu << theta, p(0) = (1 + mu/theta)^-theta ~ 1 - mu.
        self.assertAlmostEqual(
            law.log_likelihood([0.0], [-30.0]),
            -3.0 * math.log1p(math.exp(-30.0) / 3.0),
            delta=1e-20,
        )
        big = law.log_likelihood([1000.0], [math.log(1000.0)])
        self.assertTrue(math.isfinite(big) and big < 0.0)
        # Huge theta with large counts: gamma ratio must not cancel.
        huge = NegativeBinomialLaw(1e12).log_likelihood([500.0], [math.log(480.0)])
        self.assertAlmostEqual(huge, Poisson(480.0).log_pmf(500), delta=1e-6)

    def test_simulation_matches_law(self) -> None:
        """Simulated counts have the NB mean and variance within MCSE bands."""
        theta, mu, n = 2.5, 3.0, 2500
        law = NegativeBinomialLaw(theta)
        draws = law.simulate([math.log(mu)] * n, seed=7)
        mean, sd = mean_sd(draws)
        self.assertLess(abs(mean - mu), 5.0 * math.sqrt((mu + mu * mu / theta) / n))
        self.assertLess(
            abs(sd * sd - (mu + mu * mu / theta)) / (mu + mu * mu / theta), 0.2
        )
        self.assertEqual(
            law.simulate([0.5, 1.0], seed=3), law.simulate([0.5, 1.0], seed=3)
        )

    def test_maximum_likelihood_linear_log_link(self) -> None:
        """Fisher scoring converges to a stationary point of the NB likelihood."""
        law = NegativeBinomialLaw(4.0)
        xs = [i / 10.0 for i in range(40)]
        design = [[1.0, x] for x in xs]
        truth = [0.5, 0.8]
        data = law.simulate([truth[0] + truth[1] * x for x in xs], seed=11)
        result = maximum_likelihood_linear(
            design, data, law, initial=[0.0, 0.0], max_iterations=100
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(result.gradient_norm, 1e-4)
        self.assertLess(abs(result.estimate[1] - truth[1]), 0.3)
        for delta in ((1e-3, 0.0), (0.0, 1e-3), (-1e-3, 1e-3)):
            nudged = [e + d for e, d in zip(result.estimate, delta, strict=True)]
            eta = [nudged[0] + nudged[1] * x for x in xs]
            self.assertLess(
                law.log_likelihood(data, eta), result.log_likelihood + 1e-12
            )

    def test_failures(self) -> None:
        """Invalid parameters and out-of-support data are rejected."""
        for bad in (0.0, -1.0, math.nan):
            with self.assertRaises(ValueError):
                NegativeBinomialLaw(bad)
        with self.assertRaises(TypeError):
            NegativeBinomialLaw("2")  # type: ignore[arg-type]
        law = NegativeBinomialLaw(2.0)
        for y in ([-1.0], [1.5], [math.nan]):
            with self.assertRaises(ValueError):
                law.log_likelihood(y, [0.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([1.0], [701.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([1.0], [math.inf])
        with self.assertRaises(ValueError):
            law.log_likelihood([1.0, 2.0], [0.0])


# ---------------------------------------------------------------------------
# 2. Interval-censored Gaussian
# ---------------------------------------------------------------------------


def interval_mass_oracle(a: float, b: float, n: int = 20000) -> float:
    """Return ``log int_a^b phi`` by scaled Simpson quadrature (independent of CDFs)."""
    peak = min(max(0.0, a), b)
    log_peak = -0.5 * peak * peak - 0.5 * LOG_2PI
    integral = simpson(lambda x: math.exp(-0.5 * (x * x - peak * peak)), a, b, n)
    return log_peak + math.log(integral)


class IntervalCensoredTests(unittest.TestCase):
    """Interval likelihood, tails, narrow intervals, derivatives and failures."""

    def test_moderate_interval_against_normal_distribution(self) -> None:
        """log(Phi(b) - Phi(a)) matches statistics-free CDF differences."""
        law = IntervalCensoredGaussianLaw([2.0], [1.0], [4.0])
        eta = 0.5
        expected = math.log(STD.cdf((4.0 - eta) / 2.0) - STD.cdf((1.0 - eta) / 2.0))
        self.assertAlmostEqual(law.log_likelihood([2.5], [eta]), expected, delta=1e-14)

    def test_one_sided_and_exact_match_existing_censored_law(self) -> None:
        """Left, right and exact entries equal the existing censored law terms."""
        sd = [1.5, 0.7, 2.0]
        eta = [0.2, 1.0, -0.4]
        interval = IntervalCensoredGaussianLaw(sd, [None, 2.0, 0.9], [-1.0, None, 0.9])
        # entry 0: Y <= -1 (left censored at -1); entry 1: Y >= 2; entry 2: exact 0.9
        reference = CensoredGaussianLaw(
            sd, lower=[-1.0, None, None], upper=[None, 2.0, None]
        )
        y = [-1.0, 2.0, 0.9]
        self.assertAlmostEqual(
            interval.log_likelihood(y, eta),
            reference.log_likelihood(y, eta),
            delta=1e-13,
        )
        close(self, interval.score(y, eta), reference.score(y, eta), 1e-12)
        exact = IntervalCensoredGaussianLaw([2.0], [0.9], [0.9])
        self.assertAlmostEqual(
            exact.log_likelihood([0.9], [-0.4]),
            GaussianLaw(DiagonalCovariance([4.0])).log_likelihood([0.9], [-0.4]),
            delta=1e-14,
        )

    def test_uninformative_entry_contributes_nothing(self) -> None:
        """(-inf, inf) has probability one: zero value, score and information."""
        law = IntervalCensoredGaussianLaw([1.0, 1.0], [None, 0.0], [None, 1.0])
        y = law.representative()
        full = law.log_likelihood(y, [3.0, 0.5])
        single = IntervalCensoredGaussianLaw([1.0], [0.0], [1.0]).log_likelihood(
            [0.5], [0.5]
        )
        self.assertAlmostEqual(full, single, delta=1e-15)
        self.assertEqual(law.score(y, [3.0, 0.5])[0], 0.0)
        self.assertEqual(law.fisher_information([3.0, 0.5])[0], 0.0)

    def test_far_tail_intervals_against_quadrature(self) -> None:
        """Log-difference of tails stays accurate where erfc underflows."""
        for lo, hi in (
            (6.0, 9.0),
            (-30.0, -28.5),
            (37.0, 40.0),
            (-1.0, 8.0),
            (-4.0, -3.5),
        ):
            law = IntervalCensoredGaussianLaw([1.0], [lo], [hi])
            got = law.log_likelihood([0.5 * (lo + hi)], [0.0])
            self.assertAlmostEqual(
                got, interval_mass_oracle(lo, hi), delta=1e-9 * (1 + abs(got))
            )

    def test_narrow_interval_limit_is_density_times_width(self) -> None:
        """Width h -> 0: log mass = log h + log phi(mid) to O(h^2), no cancellation."""
        for h in (1e-3, 1e-6, 1e-9, 1e-12):
            law = IntervalCensoredGaussianLaw([1.5], [0.4], [0.4 + h])
            width = (0.4 + h) - 0.4  # the representable width
            mid = 0.4 + 0.5 * width
            expected = math.log(width) + Normal(0.1, 1.5).log_pdf(mid)
            got = law.log_likelihood([mid], [0.1])
            self.assertAlmostEqual(got, expected, delta=1e-9 + 0.1 * h * h)

    def test_threshold_between_quadrature_and_log_difference_is_continuous(
        self,
    ) -> None:
        """Both evaluation paths agree on either side of the narrow-interval switch."""
        for centre in (0.0, 1.5, 6.0):
            for width in (0.0999 / (1.0 + abs(centre)), 0.1001 / (1.0 + abs(centre))):
                lo, hi = centre - width / 2, centre + width / 2
                law = IntervalCensoredGaussianLaw([1.0], [lo], [hi])
                got = law.log_likelihood([centre], [0.0])
                self.assertAlmostEqual(
                    got, interval_mass_oracle(lo, hi, 2000), delta=1e-9 * (1 + abs(got))
                )

    def test_score_and_information_against_finite_differences(self) -> None:
        """Score = FD gradient; information = -FD derivative of the score."""
        law = IntervalCensoredGaussianLaw(
            [1.0, 0.8, 1.2, 0.9, 1.0, 1.1],
            [-1.0, None, 1.0, 2.0, 0.3, None],
            [2.0, 0.5, None, 2.0000001, 0.3 + 1e-4, None],
        )
        y = law.representative()
        for eta in ([0.1, 0.4, -0.7, 1.9, 0.31, 0.0], [3.0, -2.0, 4.0, 0.2, 2.0, 1.0]):
            close(self, law.score(y, eta), fd_gradient(law, y, eta), 2e-6)
            info = law.fisher_information(eta)
            for i in range(len(eta)):
                up, down = list(eta), list(eta)
                up[i] += 1e-5
                down[i] -= 1e-5
                numeric = -(law.score(y, up)[i] - law.score(y, down)[i]) / 2e-5
                self.assertAlmostEqual(
                    info[i], max(numeric, 0.0), delta=2e-6 * (1 + abs(numeric))
                )

    def test_information_is_nonnegative_in_far_tails(self) -> None:
        """Log-concavity: the observed information is nonnegative and finite."""
        law = IntervalCensoredGaussianLaw([1.0, 1.0], [30.0, -40.0], [31.0, -39.0])
        info = law.fisher_information([0.0, 0.0])
        self.assertTrue(all(math.isfinite(v) and v > 0.0 for v in info))
        score = law.score([30.0, -40.0], [0.0, 0.0])
        self.assertTrue(all(math.isfinite(v) for v in score))

    def test_simulation_is_truncated_normal(self) -> None:
        """Latent draws lie in the interval and have the truncated-normal mean."""
        a, b, n = 0.5, 2.0, 3000
        law = IntervalCensoredGaussianLaw([1.0] * n, [a] * n, [b] * n)
        draws = law.simulate([0.0] * n, seed=5)
        self.assertTrue(all(a <= d <= b for d in draws))
        mass = STD.cdf(b) - STD.cdf(a)
        mean = (math.exp(-0.5 * a * a) - math.exp(-0.5 * b * b)) / (
            math.sqrt(2 * math.pi) * mass
        )
        got, sd = mean_sd(draws)
        self.assertLess(abs(got - mean), 5.0 * sd / math.sqrt(n))
        far = IntervalCensoredGaussianLaw([1.0] * 200, [20.0] * 200, [21.0] * 200)
        tail = far.simulate([0.0] * 200, seed=1)
        self.assertTrue(all(20.0 <= d <= 21.0 for d in tail))
        self.assertLess(sum(tail) / 200 - 20.0, 0.1)

    def test_maximum_likelihood_with_mixed_censoring(self) -> None:
        """Newton scoring on a log-concave likelihood reaches a stationary point."""
        truth = (1.0, 0.5)
        xs = [i / 5.0 for i in range(30)]
        rng = resolve_stream(2).generator()
        latent = [truth[0] + truth[1] * x + rng.gauss(0.0, 0.7) for x in xs]
        lower: list[float | None] = []
        upper: list[float | None] = []
        for v in latent:
            if v < 0.5:
                lower.append(None)
                upper.append(0.5)
            elif v > 3.0:
                lower.append(3.0)
                upper.append(None)
            else:
                lower.append(math.floor(v * 2) / 2)
                upper.append(math.floor(v * 2) / 2 + 0.5)
        law = IntervalCensoredGaussianLaw([0.7] * 30, lower, upper)
        design = [[1.0, x] for x in xs]
        result = maximum_likelihood_linear(
            design, law.representative(), law, max_iterations=100
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(result.gradient_norm, 1e-4)
        self.assertLess(abs(result.estimate[1] - truth[1]), 0.25)

    def test_failures(self) -> None:
        """Bad limits, scales, data outside the interval and non-finite values."""
        with self.assertRaises(ValueError):
            IntervalCensoredGaussianLaw([1.0], [2.0], [1.0])
        with self.assertRaises(ValueError):
            IntervalCensoredGaussianLaw([0.0], [0.0], [1.0])
        with self.assertRaises(ValueError):
            IntervalCensoredGaussianLaw([1.0], [math.nan], [1.0])
        with self.assertRaises(ValueError):
            IntervalCensoredGaussianLaw([1.0], [0.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            IntervalCensoredGaussianLaw([1.0], [math.inf], [math.inf])
        with self.assertRaises(TypeError):
            IntervalCensoredGaussianLaw([1.0], ["a"], [1.0])  # type: ignore[list-item]
        law = IntervalCensoredGaussianLaw([1.0], [0.0], [1.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([1.5], [0.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([0.5], [math.nan])
        with self.assertRaises(ValueError):
            law.log_likelihood([0.5, 0.5], [0.0])


# ---------------------------------------------------------------------------
# 3. Gaussian-mixture outliers
# ---------------------------------------------------------------------------


def gaussian_mean_fit(data: Sequence[float]) -> float:
    """Return the Gaussian (least-squares) location estimate."""
    return math.fsum(data) / len(data)


def contaminated_sample(
    n: int, truth: float, n_out: int, shift: float, seed: int
) -> list[float]:
    """Return ``n`` unit-noise observations with ``n_out`` shifted by ``shift``."""
    rng = resolve_stream(seed).generator()
    values = [truth + rng.gauss(0.0, 1.0) for _ in range(n)]
    for i in range(n_out):
        values[i] += shift
    return values


def robustness_study(replicates: int, n_out: int, shift: float) -> dict[str, float]:
    """Return RMS location error of Gaussian, mixture, Student-t and Huber fits."""
    n, truth = 40, 2.0
    design = [[1.0]] * n
    mixture = GaussianMixtureOutlierLaw([1.0] * n, 0.1, 5.0)
    student = StudentTLaw([1.0] * n, 3.0)
    huber = HuberLaw([1.0] * n, 1.345)
    errors: dict[str, list[float]] = {
        "gaussian": [],
        "mixture": [],
        "student_t": [],
        "huber": [],
    }
    for r in range(replicates):
        data = contaminated_sample(n, truth, n_out, shift, 1000 + r)
        median = sorted(data)[n // 2]
        errors["gaussian"].append(gaussian_mean_fit(data) - truth)
        for name, law in (
            ("mixture", mixture),
            ("student_t", student),
            ("huber", huber),
        ):
            fit = maximum_likelihood_linear(
                design, data, law, initial=[median], max_iterations=200
            )
            errors[name].append(fit.estimate[0] - truth)
    return {
        k: math.sqrt(math.fsum(e * e for e in v) / len(v)) for k, v in errors.items()
    }


class GaussianMixtureOutlierTests(unittest.TestCase):
    """Mixture density, responsibilities, derivatives and measured robustness."""

    def test_density_by_hand_and_normalization(self) -> None:
        """One entry equals the hand-evaluated mixture; it integrates to 1."""
        eps, kappa, s = 0.15, 4.0, 0.8
        law = GaussianMixtureOutlierLaw([s], eps, kappa)
        r = 1.3
        density = (1 - eps) * Normal(0.0, s).pdf(r) + eps * Normal(0.0, kappa * s).pdf(
            r
        )
        self.assertAlmostEqual(
            law.log_likelihood([r], [0.0]), math.log(density), delta=1e-13
        )
        total = simpson(
            lambda x: math.exp(law.log_likelihood([x], [0.0])), -60.0, 60.0, 24000
        )
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_responsibilities(self) -> None:
        """Small residuals are inliers; gross outliers have responsibility ~ 1."""
        law = GaussianMixtureOutlierLaw([1.0, 1.0, 1.0], 0.1, 10.0)
        rho = law.responsibilities([0.1, 3.0, 40.0], [0.0, 0.0, 0.0])
        self.assertLess(rho[0], 0.05)
        self.assertGreater(rho[2], 1.0 - 1e-12)
        self.assertTrue(rho[0] < rho[1] < rho[2])
        eps, kappa = 0.1, 10.0
        n_in = (1 - eps) * Normal(0.0, 1.0).pdf(3.0)
        n_out = eps * Normal(0.0, kappa).pdf(3.0)
        self.assertAlmostEqual(rho[1], n_out / (n_in + n_out), delta=1e-13)

    def test_score_weights_and_finite_difference(self) -> None:
        """Score = FD gradient and equals r * EM weight."""
        law = GaussianMixtureOutlierLaw([1.0, 0.5, 2.0], 0.2, 6.0)
        y, eta = [0.4, -3.0, 25.0], [0.0, 0.2, 1.0]
        score = law.score(y, eta)
        close(self, score, fd_gradient(law, y, eta), 1e-6)
        weights = law.irls_weights(y, eta)
        close(
            self,
            score,
            [(a - b) * w for a, b, w in zip(y, eta, weights, strict=True)],
            1e-14,
        )

    def test_fisher_information_two_independent_quadratures(self) -> None:
        """J = int g'^2/g = E[observed information], both by Simpson."""
        eps, kappa = 0.1, 3.0
        law = GaussianMixtureOutlierLaw([2.0], eps, kappa)
        unit = GaussianMixtureOutlierLaw([1.0], eps, kappa)

        def density(u: float) -> float:
            return math.exp(unit.log_likelihood([u], [0.0]))

        def score_sq(u: float) -> float:
            return density(u) * unit.score([u], [0.0])[0] ** 2

        def observed(u: float) -> float:
            rho = unit.responsibilities([u], [0.0])[0]
            w = (1 - rho) + rho / kappa**2
            return density(u) * (w - u * u * rho * (1 - rho) * (1 - kappa**-2) ** 2)

        first = simpson(score_sq, -45.0, 45.0, 9000)
        second = simpson(observed, -45.0, 45.0, 9000)
        self.assertAlmostEqual(first, second, delta=1e-8)
        self.assertAlmostEqual(unit.information_factor(), first, delta=1e-8)
        self.assertAlmostEqual(
            law.fisher_information([0.0])[0], first / 4.0, delta=1e-8
        )
        # eps -> 0 gives the Gaussian information 1/s^2.
        near = GaussianMixtureOutlierLaw([1.0], 1e-12, 3.0)
        self.assertAlmostEqual(near.information_factor(), 1.0, delta=1e-9)

    def test_simulation_has_mixture_variance(self) -> None:
        """Simulated variance is s^2 (1 - eps + eps kappa^2) within a band."""
        eps, kappa, n = 0.1, 3.0, 6000
        law = GaussianMixtureOutlierLaw([1.0] * n, eps, kappa)
        draws = law.simulate([0.0] * n, seed=9)
        var = math.fsum(d * d for d in draws) / n
        expected = 1 - eps + eps * kappa**2
        self.assertLess(abs(var - expected) / expected, 0.12)

    def test_robustness_against_gross_outliers_measured(self) -> None:
        """With 4/40 gross outliers the mixture beats the Gaussian mean."""
        stats = robustness_study(replicates=12, n_out=4, shift=20.0)
        print(
            f"\nROBUSTNESS fast (12 reps, 4/40 outliers at +20): {stats}",
            file=sys.stderr,
        )
        self.assertGreater(stats["gaussian"], 1.5)
        self.assertLess(stats["mixture"], 0.5 * stats["gaussian"])
        self.assertLess(stats["mixture"], 0.6)
        self.assertLess(stats["student_t"], 0.6)
        self.assertLess(stats["huber"], 0.6)

    def test_not_universally_better_on_clean_data(self) -> None:
        """Without outliers the robust fits are no better than the Gaussian mean."""
        stats = robustness_study(replicates=12, n_out=0, shift=0.0)
        print(f"\nROBUSTNESS clean (12 reps): {stats}", file=sys.stderr)
        self.assertLessEqual(stats["gaussian"], stats["mixture"] * 1.2)

    def test_mixture_via_missing_mask(self) -> None:
        """The elementwise law works under MissingMask."""
        law = GaussianMixtureOutlierLaw([1.0, 1.0, 1.0], 0.1, 4.0)
        masked = MissingMask(law, [True, False, True])
        value = masked.log_likelihood([0.2, math.nan, -0.5], [0.0, 0.0, 0.0])
        direct = law.restrict([0, 2]).log_likelihood([0.2, -0.5], [0.0, 0.0])
        self.assertAlmostEqual(value, direct, delta=1e-14)

    def test_failures(self) -> None:
        """Invalid mixture parameters and data are rejected."""
        for eps in (0.0, 1.0, -0.1, 1.5, math.nan):
            with self.assertRaises(ValueError):
                GaussianMixtureOutlierLaw([1.0], eps, 3.0)
        for kappa in (1.0, 0.5, math.inf):
            with self.assertRaises(ValueError):
                GaussianMixtureOutlierLaw([1.0], 0.1, kappa)
        with self.assertRaises(ValueError):
            GaussianMixtureOutlierLaw([-1.0], 0.1, 3.0)
        law = GaussianMixtureOutlierLaw([1.0], 0.1, 3.0)
        with self.assertRaises(ValueError):
            law.log_likelihood([math.nan], [0.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([1.0, 2.0], [0.0])


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class RobustnessSlowTests(unittest.TestCase):
    """The robustness comparison with 200 replicates."""

    def test_robustness_200_replicates(self) -> None:
        """Measured RMS errors over 200 replicates."""
        for n_out, shift in ((4, 20.0), (2, 50.0), (0, 0.0)):
            stats = robustness_study(replicates=200, n_out=n_out, shift=shift)
            print(
                f"\nROBUSTNESS slow (200 reps, {n_out}/40 at +{shift}): {stats}",
                file=sys.stderr,
            )
            if n_out:
                self.assertLess(stats["mixture"], 0.5 * stats["gaussian"])


# ---------------------------------------------------------------------------
# 4. Huber
# ---------------------------------------------------------------------------


class HuberTests(unittest.TestCase):
    """Density normalization, M-estimation criterion, IRLS and sandwich covariance."""

    def test_normalizing_constant_by_quadrature(self) -> None:
        """Z_k = int exp(-rho_k) du and the normalized density integrates to one."""
        for k in (0.3, 1.345, 3.0):
            law = HuberLaw([1.0], k)
            reach = 60.0 / k + 20.0

            def weight(u: float, k: float = k) -> float:
                return math.exp(
                    -(0.5 * u * u if abs(u) <= k else k * abs(u) - 0.5 * k * k)
                )

            # The integrand has kinks at +-k, so integrate piecewise.
            z = (
                simpson(weight, -reach, -k, 20000)
                + simpson(weight, -k, k, 4000)
                + simpson(weight, k, reach, 20000)
            )
            self.assertAlmostEqual(law.normalizing_constant(), z, delta=1e-10)
            self.assertAlmostEqual(z / law.normalizing_constant(), 1.0, delta=1e-10)

    def test_scale_enters_the_density_correctly(self) -> None:
        """Density of y = eta + s u equals (1/s) f(u)."""
        a, b = HuberLaw([1.0], 1.2), HuberLaw([2.5], 1.2)
        self.assertAlmostEqual(
            b.log_likelihood([3.0], [0.5]),
            a.log_likelihood([1.0], [0.0]) - math.log(2.5),
            delta=1e-13,
        )

    def test_large_threshold_recovers_gaussian(self) -> None:
        """K -> infinity gives N(eta, s^2) (Z -> sqrt(2 pi))."""
        law = HuberLaw([1.7, 0.6], 40.0)
        gauss = GaussianLaw(DiagonalCovariance([1.7**2, 0.6**2]))
        y, eta = [0.3, -0.2], [0.0, 0.1]
        self.assertAlmostEqual(
            law.log_likelihood(y, eta), gauss.log_likelihood(y, eta), delta=1e-12
        )

    def test_pseudo_likelihood_is_unnormalized_and_cannot_estimate_scale(self) -> None:
        """The loss alone prefers s -> infinity; the density has an interior optimum."""
        rng = resolve_stream(4).generator()
        data = [rng.gauss(0.0, 1.0) for _ in range(60)]
        zeros = [0.0] * 60
        pseudo = [
            HuberLaw([s] * 60, 1.345).pseudo_log_likelihood(data, zeros)
            for s in (0.5, 1.0, 5.0, 100.0)
        ]
        self.assertTrue(all(a < b for a, b in zip(pseudo, pseudo[1:], strict=False)))
        normalized = {
            s: HuberLaw([s] * 60, 1.345).log_likelihood(data, zeros)
            for s in (0.2, 1.0, 100.0)
        }
        self.assertGreater(normalized[1.0], normalized[0.2])
        self.assertGreater(normalized[1.0], normalized[100.0])
        law = HuberLaw([1.0], 1.0)
        gap = law.log_likelihood([0.3], [0.0]) - law.pseudo_log_likelihood([0.3], [0.0])
        self.assertAlmostEqual(gap, -math.log(law.normalizing_constant()), delta=1e-14)

    def test_score_is_clipped_residual_and_matches_finite_difference(self) -> None:
        """psi_k clips; the gradient matches differences away from the kinks."""
        law = HuberLaw([1.0, 2.0, 0.5], 1.345)
        y, eta = [0.4, 9.0, -7.0], [0.0, 0.0, 0.0]
        score = law.score(y, eta)
        close(self, score, [0.4, 1.345 / 2.0, -1.345 / 0.5], 1e-14)
        close(self, score, fd_gradient(law, y, eta), 1e-7)

    def test_information_equals_score_second_moment(self) -> None:
        """P(|u| <= k) = E psi^2 = E[curvature] under the Huber density."""
        k = 1.345
        law = HuberLaw([1.0], k)
        z = law.normalizing_constant()

        def density(u: float) -> float:
            return (
                math.exp(-(0.5 * u * u if abs(u) <= k else k * abs(u) - 0.5 * k * k))
                / z
            )

        def integrand(u: float) -> float:
            return density(u) * max(-k, min(k, u)) ** 2

        second = (
            simpson(integrand, -60.0, -k, 20000)
            + simpson(integrand, -k, k, 4000)
            + simpson(integrand, k, 60.0, 20000)
        )
        self.assertAlmostEqual(second, law.fisher_information([0.0])[0], delta=1e-7)
        self.assertAlmostEqual(
            law.central_probability(), simpson(density, -k, k, 4000), delta=1e-10
        )

    def test_simulation_matches_density(self) -> None:
        """Central mass and variance of simulated draws match the density."""
        k, n = 1.0, 6000
        law = HuberLaw([1.0] * n, k)
        draws = law.simulate([0.0] * n, seed=21)
        share = sum(1 for d in draws if abs(d) <= k) / n
        p = law.central_probability()
        self.assertLess(abs(share - p), 5.0 * math.sqrt(p * (1 - p) / n))
        z = law.normalizing_constant()
        variance = simpson(
            lambda u: (
                u
                * u
                * math.exp(-(0.5 * u * u if abs(u) <= k else k * abs(u) - 0.5 * k * k))
                / z
            ),
            -60.0,
            60.0,
            60000,
        )
        got = math.fsum(d * d for d in draws) / n
        self.assertLess(abs(got - variance) / variance, 0.1)

    def test_irls_decreases_loss_and_resists_outliers(self) -> None:
        """Half-quadratic IRLS monotonically decreases the Huber loss."""
        data = contaminated_sample(40, 2.0, 4, 20.0, seed=3)
        law = HuberLaw([1.0] * 40, 1.345)
        result = maximum_likelihood_linear(
            [[1.0]] * 40, data, law, initial=[0.0], max_iterations=200
        )
        self.assertTrue(result.converged, result.stop_reason)
        objectives = [h.objective for h in result.history]
        self.assertTrue(
            all(
                b <= a + 1e-12 for a, b in zip(objectives, objectives[1:], strict=False)
            )
        )
        self.assertLess(abs(result.estimate[0] - 2.0), 0.7)
        self.assertGreater(abs(gaussian_mean_fit(data) - 2.0), 1.5)

    def test_sandwich_scalar_closed_form_and_two_parameter_algebra(self) -> None:
        """Scalar cov = M / B^2; two parameters B^-1 M B^-1 by direct algebra."""
        data = contaminated_sample(60, 1.0, 3, 15.0, seed=8)
        law = HuberLaw([1.0] * 60, 1.345)
        fit = maximum_likelihood_linear(
            [[1.0]] * 60, data, law, initial=[1.0], max_iterations=200
        )
        x = fit.estimate[0]
        psi = [max(-1.345, min(1.345, v - x)) for v in data]
        bread = sum(1.0 for v in data if abs(v - x) < 1.345)
        meat = math.fsum(p * p for p in psi)
        cov = huber_sandwich_covariance([[1.0]] * 60, data, law, [x])
        self.assertAlmostEqual(cov[0][0], meat / bread**2, delta=1e-14)
        # Two parameters.
        ts = [i / 20.0 for i in range(60)]
        design = [[1.0, t] for t in ts]
        y = [0.5 + 2.0 * t + v for t, v in zip(ts, data, strict=True)]
        theta = [0.5, 2.0]
        cov2 = huber_sandwich_covariance(design, y, law, theta)
        resid = [yi - theta[0] - theta[1] * t for yi, t in zip(y, ts, strict=True)]
        c = [1.0 if abs(r) < 1.345 else 0.0 for r in resid]
        p2 = [max(-1.345, min(1.345, r)) ** 2 for r in resid]
        bmat = [
            [
                sum(ci * a[i] * a[j] for ci, a in zip(c, design, strict=True))
                for j in range(2)
            ]
            for i in range(2)
        ]
        mmat = [
            [
                sum(pi * a[i] * a[j] for pi, a in zip(p2, design, strict=True))
                for j in range(2)
            ]
            for i in range(2)
        ]
        binv = inverse_spd(bmat)
        left = [
            [sum(binv[i][k] * mmat[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        expected = [
            [sum(left[i][k] * binv[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        for i in range(2):
            close(self, cov2[i], expected[i], 1e-12)

    def test_failures(self) -> None:
        """Bad parameters, bad arguments and an all-clipped bread."""
        with self.assertRaises(ValueError):
            HuberLaw([1.0], 0.0)
        with self.assertRaises(ValueError):
            HuberLaw([0.0], 1.0)
        with self.assertRaises(ValueError):
            HuberLaw([1.0], math.inf)
        with self.assertRaises(TypeError):
            HuberLaw([1.0], "1")  # type: ignore[arg-type]
        law = HuberLaw([1.0, 1.0], 1.0)
        with self.assertRaises(ValueError):
            law.log_likelihood([math.nan, 0.0], [0.0, 0.0])
        with self.assertRaises(TypeError):
            huber_sandwich_covariance(
                [[1.0], [1.0]], [0.0, 0.0], StudentTLaw([1.0, 1.0], 3.0), [0.0]
            )  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            huber_sandwich_covariance([[1.0], [1.0]], [100.0, -100.0], law, [0.0])


# ---------------------------------------------------------------------------
# 5. Multivariate Student-t
# ---------------------------------------------------------------------------


SCATTER3 = [[2.0, 0.6, -0.3], [0.6, 1.5, 0.4], [-0.3, 0.4, 1.0]]


class MultivariateStudentTTests(unittest.TestCase):
    """Density against probstat's multivariate t, joint dependence and derivatives."""

    def test_matches_multivariate_student_t(self) -> None:
        """Exact agreement with probstat.multivariate across nu and dimension."""
        for nu in (0.7, 2.0, 3.5, 30.0, 1e6):
            for scatter in ([[2.0]], [[2.0, 0.5], [0.5, 1.0]], SCATTER3):
                m = len(scatter)
                eta = [0.1 * i for i in range(m)]
                y = [1.0 - 0.7 * i for i in range(m)]
                got = MultivariateStudentTLaw(scatter, nu).log_likelihood(y, eta)
                expected = MultivariateStudentT(eta, scatter, nu).log_pdf(y)
                self.assertAlmostEqual(got, expected, delta=1e-11 * (1 + abs(expected)))

    def test_univariate_case_matches_student_t_distribution(self) -> None:
        """M = 1 is the scaled Student-t of the distribution module and StudentTLaw."""
        law = MultivariateStudentTLaw([[4.0]], 5.0)
        self.assertAlmostEqual(
            law.log_likelihood([1.5], [0.2]),
            StudentT(5.0, 0.2, 2.0).log_pdf(1.5),
            delta=1e-13,
        )
        self.assertAlmostEqual(
            law.fisher_information([0.0])[0],
            StudentTLaw([2.0], 5.0).fisher_information([0.0])[0],
            delta=1e-14,
        )
        close(
            self,
            law.score([1.5], [0.2]),
            StudentTLaw([2.0], 5.0).score([1.5], [0.2]),
            1e-13,
        )

    def test_joint_law_is_not_a_product_of_univariate_t(self) -> None:
        """A diagonal scatter matrix still gives dependent components."""
        joint = MultivariateStudentTLaw([[1.0, 0.0], [0.0, 1.0]], 4.0)
        product = StudentTLaw([1.0, 1.0], 4.0)
        y, eta = [2.0, -2.0], [0.0, 0.0]
        self.assertGreater(
            abs(joint.log_likelihood(y, eta) - product.log_likelihood(y, eta)), 0.1
        )
        # Independence would factorize into the product of the marginals.
        marginal = joint.restrict([0])
        self.assertAlmostEqual(
            marginal.log_likelihood([2.0], [0.0]),
            StudentT(4.0, 0.0, 1.0).log_pdf(2.0),
            delta=1e-13,
        )

    def test_covariance_is_scaled_scatter_only_above_two(self) -> None:
        """nu/(nu-2) Sigma for nu > 2; refused otherwise; matches simulation."""
        law = MultivariateStudentTLaw(SCATTER3, 12.0)
        cov = law.covariance_matrix()
        for i in range(3):
            close(self, cov[i], [12.0 / 10.0 * v for v in SCATTER3[i]], 1e-14)
        for nu in (2.0, 1.5, 0.9):
            with self.assertRaises(ValueError):
                MultivariateStudentTLaw(SCATTER3, nu).covariance_matrix()
        stream = resolve_stream(31)
        n = 8000
        samples = [law.simulate([0.0] * 3, seed=stream.child(i)) for i in range(n)]
        for i, j in ((0, 0), (1, 1), (0, 1), (1, 2)):
            est = math.fsum(s[i] * s[j] for s in samples) / n
            self.assertLess(
                abs(est - cov[i][j]), 0.12 * math.sqrt(cov[i][i] * cov[j][j])
            )

    def test_score_matches_finite_difference(self) -> None:
        """Score = FD gradient; information equals E[score^2] (simulated)."""
        law = MultivariateStudentTLaw(SCATTER3, 5.0)
        y, eta = [1.0, -2.0, 0.5], [0.2, 0.1, -0.3]
        close(self, law.score(y, eta), fd_gradient(law, y, eta), 1e-6)
        stream = resolve_stream(14)
        n = 6000
        acc = [0.0, 0.0, 0.0]
        for i in range(n):
            draw = law.simulate(eta, seed=stream.child(i))
            s = law.score(draw, eta)
            for k in range(3):
                acc[k] += s[k] * s[k] / n
        info = law.fisher_information(eta)
        for k in range(3):
            self.assertLess(abs(acc[k] - info[k]) / info[k], 0.08)

    def test_irls_decreases_objective(self) -> None:
        """EM weighting gives a monotone decrease and a stationary limit."""
        design = [[1.0, t / 5.0] for t in range(6)]
        scatter = [[1.0 if i == j else 0.3 for j in range(6)] for i in range(6)]
        law = MultivariateStudentTLaw(scatter, 3.0)
        y = [0.1, 0.4, 0.5, 9.0, 0.9, 1.2]
        result = maximum_likelihood_linear(design, y, law, max_iterations=200)
        self.assertTrue(result.converged, result.stop_reason)
        objectives = [h.objective for h in result.history]
        self.assertTrue(
            all(
                b <= a + 1e-12 for a, b in zip(objectives, objectives[1:], strict=False)
            )
        )
        self.assertLess(result.gradient_norm, 1e-4)

    def test_missing_mask_uses_marginal_t(self) -> None:
        """Masking uses the sub-scatter matrix with the same nu."""
        law = MultivariateStudentTLaw(SCATTER3, 4.0)
        masked = MissingMask(law, [True, False, True])
        value = masked.log_likelihood([1.0, math.nan, -1.0], [0.0, 0.0, 0.2])
        sub = [[SCATTER3[0][0], SCATTER3[0][2]], [SCATTER3[2][0], SCATTER3[2][2]]]
        expected = MultivariateStudentT([0.0, 0.2], sub, 4.0).log_pdf([1.0, -1.0])
        self.assertAlmostEqual(value, expected, delta=1e-12)

    def test_failures(self) -> None:
        """Non-symmetric, indefinite, wrong-shape scatter and invalid nu or data."""
        with self.assertRaises(ValueError):
            MultivariateStudentTLaw([[1.0, 0.5], [0.0, 1.0]], 3.0)
        with self.assertRaises(ValueError):
            MultivariateStudentTLaw([[1.0, 2.0], [2.0, 1.0]], 3.0)
        with self.assertRaises(ValueError):
            MultivariateStudentTLaw([[1.0, 0.0]], 3.0)
        with self.assertRaises(ValueError):
            MultivariateStudentTLaw([], 3.0)
        for nu in (0.0, -2.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                MultivariateStudentTLaw([[1.0]], nu)
        law = MultivariateStudentTLaw([[1.0]], 3.0)
        with self.assertRaises(ValueError):
            law.log_likelihood([math.inf], [0.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([1.0, 2.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# 6. State-dependent noise
# ---------------------------------------------------------------------------


def correlated_state_law(a: float, b: float, rho: float) -> StateDependentCovarianceLaw:
    """Return ``Gamma(eta) = D R D`` with ``D = diag(sqrt(a + b eta^2))``."""
    m = 3
    corr = [[1.0 if i == j else rho for j in range(m)] for i in range(m)]

    def cov(eta: Sequence[float]) -> tuple[list[list[float]], list[list[list[float]]]]:
        d = [math.sqrt(a + b * e * e) for e in eta]
        dd = [b * e / math.sqrt(a + b * e * e) for e in eta]  # d d_i / d eta_i
        gamma = [[d[i] * corr[i][j] * d[j] for j in range(m)] for i in range(m)]
        derivs = []
        for k in range(m):
            g = [[0.0] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    left = dd[k] if i == k else 0.0
                    right = dd[k] if j == k else 0.0
                    g[i][j] = left * corr[i][j] * d[j] + d[i] * corr[i][j] * right
            derivs.append(g)
        return gamma, derivs

    return StateDependentCovarianceLaw(m, cov)


class HeteroscedasticTests(unittest.TestCase):
    """Gradient with log-determinant and covariance-derivative terms."""

    def test_density_and_constant_variance_limit(self) -> None:
        """V = a + b eta^2 matches Normal.log_pdf; b = 0 gives the Gaussian law."""
        law = HeteroscedasticGaussianLaw.quadratic(0.5, 0.2)
        y, eta = [1.0, -2.0], [0.3, 1.1]
        expected = math.fsum(
            Normal(e, math.sqrt(0.5 + 0.2 * e * e)).log_pdf(v)
            for v, e in zip(y, eta, strict=True)
        )
        self.assertAlmostEqual(law.log_likelihood(y, eta), expected, delta=1e-13)
        flat = HeteroscedasticGaussianLaw.quadratic(0.5, 0.0)
        gauss = GaussianLaw(DiagonalCovariance([0.5, 0.5]))
        self.assertAlmostEqual(
            flat.log_likelihood(y, eta), gauss.log_likelihood(y, eta), delta=1e-13
        )
        close(self, flat.score(y, eta), gauss.score(y, eta), 1e-13)

    def test_score_includes_variance_terms(self) -> None:
        """The exact gradient matches differences; the fixed-variance score not."""
        law = HeteroscedasticGaussianLaw.quadratic(0.3, 0.5)
        y, eta = [2.0, -1.0, 0.4, 5.0], [1.5, -0.4, 0.2, 2.0]
        score = law.score(y, eta)
        close(self, score, fd_gradient(law, y, eta), 2e-7)
        naive = [(v - e) / (0.3 + 0.5 * e * e) for v, e in zip(y, eta, strict=True)]
        self.assertGreater(
            max(abs(a - b) for a, b in zip(score, naive, strict=True)), 0.1
        )

    def test_information_is_score_variance(self) -> None:
        """I = 1/v + v'^2/(2 v^2) = E[score^2] by Simpson over the Gaussian."""
        law = HeteroscedasticGaussianLaw.quadratic(0.3, 0.5)
        for e in (-1.0, 0.0, 0.7, 2.0):
            v = 0.3 + 0.5 * e * e
            sd = math.sqrt(v)
            integral = simpson(
                lambda z, e=e, sd=sd: (
                    math.exp(-0.5 * z * z)
                    / math.sqrt(2 * math.pi)
                    * law.score([e + sd * z], [e])[0] ** 2
                ),
                -10.0,
                10.0,
                600,
            )
            self.assertAlmostEqual(integral, law.fisher_information([e])[0], delta=1e-9)

    def test_user_callable_and_fit(self) -> None:
        """A user callable with derivative; scoring reaches a stationary point."""
        law = HeteroscedasticGaussianLaw(
            lambda e: (0.1 + 0.01 * abs(e) ** 3, 0.03 * e * abs(e))
        )
        xs = [0.2 + 0.01 * i for i in range(200)]
        design = [[1.0, x] for x in xs]
        data = law.simulate([1.0 + 2.0 * x for x in xs], seed=17)
        result = maximum_likelihood_linear(
            design, data, law, initial=[0.5, 1.0], max_iterations=200
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(result.gradient_norm, 1e-4)
        self.assertLess(abs(result.estimate[1] - 2.0), 0.3)

    def test_failures(self) -> None:
        """Non-positive or non-finite variance, bad parameters."""
        with self.assertRaises(ValueError):
            HeteroscedasticGaussianLaw.quadratic(0.0, 1.0)
        with self.assertRaises(ValueError):
            HeteroscedasticGaussianLaw.quadratic(1.0, -1.0)
        with self.assertRaises(TypeError):
            HeteroscedasticGaussianLaw(3.0)  # type: ignore[arg-type]
        bad = HeteroscedasticGaussianLaw(lambda e: (-1.0, 0.0))
        with self.assertRaises(ValueError):
            bad.log_likelihood([0.0], [0.0])
        nan = HeteroscedasticGaussianLaw(lambda e: (1.0, math.nan))
        with self.assertRaises(ValueError):
            nan.score([0.0], [0.0])
        with self.assertRaises(ValueError):
            HeteroscedasticGaussianLaw.quadratic(1.0, 1.0).log_likelihood(
                [0.0], [math.nan]
            )


class StateDependentCovarianceTests(unittest.TestCase):
    """The full-matrix law: trace and quadratic-form terms, information, scoring."""

    def test_density_and_constant_covariance(self) -> None:
        """Constant Gamma reproduces GaussianLaw including the log-determinant."""
        gamma = [[2.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.5]]
        zero = [[[0.0] * 3 for _ in range(3)] for _ in range(3)]
        law = StateDependentCovarianceLaw(3, lambda eta: (gamma, zero))
        ref = GaussianLaw(DenseCovariance(gamma))
        y, eta = [1.0, 0.2, -0.5], [0.0, 0.5, 0.1]
        self.assertAlmostEqual(
            law.log_likelihood(y, eta), ref.log_likelihood(y, eta), delta=1e-13
        )
        close(self, law.score(y, eta), ref.score(y, eta), 1e-13)
        full = law.fisher_matrix(eta)
        for i in range(3):
            self.assertAlmostEqual(
                full[i][i], ref.fisher_information(eta)[i], delta=1e-13
            )

    def test_gradient_with_logdet_and_covariance_derivative_terms(self) -> None:
        """Exact gradient = finite differences for a correlated Gamma(eta)."""
        law = correlated_state_law(0.4, 0.3, 0.5)
        for y, eta in (
            ([1.0, -0.5, 2.0], [0.4, 0.3, 1.0]),
            ([0.2, 0.1, -3.0], [-1.0, 0.8, 0.5]),
        ):
            close(self, law.score(y, eta), fd_gradient(law, y, eta, 1e-6), 5e-7)
        # Omitting the covariance-derivative terms must be detectable.
        y, eta = [1.0, -0.5, 2.0], [0.4, 0.3, 1.0]
        gauss_part = GaussianLaw(
            DenseCovariance(
                correlated_state_law(0.4, 0.3, 0.5)._evaluate(tuple(eta))[0]
            )
        ).score(y, eta)
        gap = max(
            abs(a - b) for a, b in zip(law.score(y, eta), gauss_part, strict=True)
        )
        self.assertGreater(gap, 0.05)

    def test_diagonal_case_matches_heteroscedastic_law(self) -> None:
        """With R = I the matrix law equals the elementwise law."""
        full = correlated_state_law(0.4, 0.3, 0.0)
        diag = HeteroscedasticGaussianLaw.quadratic(0.4, 0.3)
        y, eta = [1.0, -0.5, 2.0], [0.4, 0.3, 1.0]
        self.assertAlmostEqual(
            full.log_likelihood(y, eta), diag.log_likelihood(y, eta), delta=1e-12
        )
        close(self, full.score(y, eta), diag.score(y, eta), 1e-12)
        close(self, full.fisher_information(eta), diag.fisher_information(eta), 1e-12)

    def test_information_matrix_equals_score_covariance(self) -> None:
        """F = E[score score'] checked by simulation (information identity)."""
        law = correlated_state_law(0.5, 0.4, 0.6)
        eta = [0.6, -0.4, 1.0]
        fisher = law.fisher_matrix(eta)
        stream = resolve_stream(41)
        n = 1500
        acc = [[0.0] * 3 for _ in range(3)]
        for i in range(n):
            s = law.score(law.simulate(eta, seed=stream.child(i)), eta)
            for p in range(3):
                for q in range(3):
                    acc[p][q] += s[p] * s[q] / n
        for p in range(3):
            for q in range(3):
                self.assertLess(
                    abs(acc[p][q] - fisher[p][q]),
                    0.12 * math.sqrt(fisher[p][p] * fisher[q][q]),
                )

    def test_scoring_step_and_fit(self) -> None:
        """Full-information scoring reaches a stationary point on correlated data."""
        law = correlated_state_law(0.4, 0.3, 0.5)
        design = [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
        data = law.simulate([1.0 + 2.0 * 0.0, 1.0 + 2.0 * 0.5, 1.0 + 2.0 * 1.0], seed=5)
        result = maximum_likelihood_linear(
            design, data, law, initial=[0.5, 1.0], max_iterations=200
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(result.gradient_norm, 1e-4)

    def test_masks_are_refused_not_silently_approximated(self) -> None:
        """Restrict raises NotImplementedError, so MissingMask cannot wrap the law."""
        law = correlated_state_law(0.4, 0.3, 0.5)
        with self.assertRaises(NotImplementedError):
            law.restrict([0, 1])
        with self.assertRaises(NotImplementedError):
            MissingMask(law, [True, True, False])

    def test_failures(self) -> None:
        """Non-SPD, non-symmetric, wrong shapes and wrong derivative counts."""
        eye = [[1.0, 0.0], [0.0, 1.0]]
        zero = [[0.0, 0.0], [0.0, 0.0]]
        with self.assertRaises(ValueError):
            StateDependentCovarianceLaw(0, lambda e: (eye, [zero]))
        with self.assertRaises(TypeError):
            StateDependentCovarianceLaw(2, 3)  # type: ignore[arg-type]
        cases = {
            "indefinite": lambda e: ([[1.0, 2.0], [2.0, 1.0]], [zero, zero]),
            "asymmetric": lambda e: ([[1.0, 0.5], [0.0, 1.0]], [zero, zero]),
            "wrong derivative count": lambda e: (eye, [zero]),
            "wrong shape": lambda e: ([[1.0]], [zero, zero]),
            "nan": lambda e: ([[math.nan, 0.0], [0.0, 1.0]], [zero, zero]),
        }
        for name, fn in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                StateDependentCovarianceLaw(2, fn).log_likelihood(
                    [0.0, 0.0], [0.0, 0.0]
                )


# ---------------------------------------------------------------------------
# 7. Nuisance scale
# ---------------------------------------------------------------------------


class ProfileScaleTests(unittest.TestCase):
    """Closed-form sigma-hat and the profile objective."""

    def test_profile_equals_maximum_over_sigma(self) -> None:
        """Profile log-likelihood = max over sigma of the Gaussian likelihood."""
        gamma0 = [[2.0, 0.4], [0.4, 1.0]]
        law = ProfiledScaleGaussianLaw(DenseCovariance(gamma0))
        y, eta = [1.5, -0.7], [0.2, 0.1]

        def gaussian(sigma: float) -> float:
            scaled = [[sigma * sigma * v for v in row] for row in gamma0]
            return GaussianLaw(DenseCovariance(scaled)).log_likelihood(y, eta)

        lo, hi = 1e-3, 50.0
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(200):
            c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
            if gaussian(c) > gaussian(d):
                hi = d
            else:
                lo = c
        self.assertAlmostEqual(law.scale_estimate(y, eta), 0.5 * (lo + hi), delta=1e-6)
        self.assertAlmostEqual(
            law.log_likelihood(y, eta),
            gaussian(law.scale_estimate(y, eta)),
            delta=1e-12,
        )
        self.assertGreaterEqual(
            law.log_likelihood(y, eta) + 1e-12, gaussian(0.5 * (lo + hi) * 1.1)
        )

    def test_scale_estimate_closed_form(self) -> None:
        """sigma_hat^2 = r' Gamma0^-1 r / m by hand for a diagonal covariance."""
        law = ProfiledScaleGaussianLaw(DiagonalCovariance([4.0, 1.0, 0.25]))
        y, eta = [2.0, 1.0, 0.5], [0.0, 0.0, 0.0]
        q = 4.0 / 4.0 + 1.0 + 0.25 / 0.25
        self.assertAlmostEqual(law.scale_estimate(y, eta) ** 2, q / 3.0, delta=1e-15)

    def test_score_finite_difference_and_scale_invariance(self) -> None:
        """Score = FD gradient; rescaling the data shifts the value by -m log c only."""
        law = ProfiledScaleGaussianLaw(DenseCovariance([[2.0, 0.4], [0.4, 1.0]]))
        y, eta = [1.5, -0.7], [0.2, 0.1]
        close(self, law.score(y, eta), fd_gradient(law, y, eta), 1e-6)
        shifted = law.log_likelihood([3 * v for v in y], [3 * v for v in eta])
        self.assertAlmostEqual(
            shifted, law.log_likelihood(y, eta) - 2 * math.log(3.0), delta=1e-12
        )

    def test_linear_fit_is_generalized_least_squares(self) -> None:
        """One scoring step reaches the GLS solution computed with linalg."""
        design = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        gamma0 = [[1.0 if i == j else 0.3 for j in range(4)] for i in range(4)]
        y = [0.9, 2.2, 2.7, 4.4]
        law = ProfiledScaleGaussianLaw(DenseCovariance(gamma0))
        result = maximum_likelihood_linear(design, y, law, max_iterations=50)
        self.assertTrue(result.converged, result.stop_reason)
        precision = inverse_spd(gamma0)
        pa = [
            [sum(precision[i][k] * design[k][j] for k in range(4)) for j in range(2)]
            for i in range(4)
        ]
        normal = [
            [sum(design[k][i] * pa[k][j] for k in range(4)) for j in range(2)]
            for i in range(2)
        ]
        rhs = [sum(pa[k][i] * y[k] for k in range(4)) for i in range(2)]
        close(self, result.estimate, solve_spd(normal, rhs), 1e-9)

    def test_undefined_quantities_are_refused(self) -> None:
        """No expected information or unscaled simulation; zero residual refused."""
        law = ProfiledScaleGaussianLaw(ScaledIdentityCovariance(2, 1.0))
        with self.assertRaises(NotImplementedError):
            law.fisher_information([0.0, 0.0])
        with self.assertRaises(NotImplementedError):
            law.simulate([0.0, 0.0], seed=1)
        with self.assertRaises(ValueError):
            law.log_likelihood([1.0, 2.0], [1.0, 2.0])
        with self.assertRaises(TypeError):
            ProfiledScaleGaussianLaw("identity")  # type: ignore[arg-type]
        draw = law.simulate_with_scale([0.0, 0.0], 2.0, seed=3)
        self.assertEqual(len(draw), 2)
        with self.assertRaises(ValueError):
            law.simulate_with_scale([0.0, 0.0], -1.0, seed=3)


def marginal_by_quadrature(
    gamma0: Sequence[Sequence[float]],
    y: Sequence[float],
    eta: Sequence[float],
    alpha: float,
    beta: float,
) -> float:
    """Return ``log int N(y; eta, s2 Gamma0) IG(s2) ds2`` via ``t = log s2``."""
    m = len(y)
    cov = DenseCovariance(gamma0)
    q = cov.quadratic_form([a - b for a, b in zip(y, eta, strict=True)])
    logdet = cov.log_det()

    def log_integrand(t: float) -> float:
        log_normal = -0.5 * (q * math.exp(-t) + logdet + m * t + m * LOG_2PI)
        log_prior = (
            alpha * math.log(beta)
            - math.lgamma(alpha)
            - (alpha + 1.0) * t
            - beta * math.exp(-t)
        )
        return log_normal + log_prior + t

    lo, hi, n = -30.0, 30.0, 6000
    h = (hi - lo) / n
    values = [log_integrand(lo + i * h) for i in range(n + 1)]
    peak = max(values)
    total = sum(
        (1 if i in (0, n) else 4 if i % 2 else 2) * math.exp(v - peak)
        for i, v in enumerate(values)
    )
    return peak + math.log(total * h / 3.0)


class MarginalScaleTests(unittest.TestCase):
    """Inverse-gamma marginalisation gives a multivariate Student-t."""

    gamma0 = [[2.0, 0.4, 0.0], [0.4, 1.0, 0.3], [0.0, 0.3, 1.5]]

    def test_against_one_dimensional_quadrature_over_sigma(self) -> None:
        """The closed form equals the numerical integral over sigma^2."""
        for alpha, beta in ((3.0, 2.0), (0.8, 0.5), (12.0, 20.0)):
            law = MarginalScaleGaussianLaw(DenseCovariance(self.gamma0), alpha, beta)
            for y, eta in (
                ([1.0, -0.5, 2.0], [0.2, 0.0, 0.5]),
                ([0.1, 0.0, -0.1], [0.0, 0.0, 0.0]),
                ([9.0, -7.0, 3.0], [0.0, 0.0, 0.0]),
            ):
                got = law.log_likelihood(y, eta)
                oracle = marginal_by_quadrature(self.gamma0, y, eta, alpha, beta)
                self.assertAlmostEqual(got, oracle, delta=1e-8)

    def test_is_multivariate_t_with_two_alpha_dof(self) -> None:
        """Matches MultivariateStudentT(2 alpha, beta/alpha Gamma0) and our law."""
        alpha, beta = 3.0, 2.0
        law = MarginalScaleGaussianLaw(DenseCovariance(self.gamma0), alpha, beta)
        self.assertEqual(law.degrees_of_freedom, 6.0)
        self.assertAlmostEqual(law.scatter_factor, beta / alpha)
        scatter = [[beta / alpha * v for v in row] for row in self.gamma0]
        y, eta = [1.0, -0.5, 2.0], [0.2, 0.0, 0.5]
        expected = MultivariateStudentT(eta, scatter, 6.0).log_pdf(y)
        self.assertAlmostEqual(law.log_likelihood(y, eta), expected, delta=1e-11)
        twin = MultivariateStudentTLaw(scatter, 6.0)
        close(self, law.score(y, eta), twin.score(y, eta), 1e-12)
        close(self, law.fisher_information(eta), twin.fisher_information(eta), 1e-12)

    def test_score_and_scale_posterior(self) -> None:
        """Score = FD gradient; posterior of sigma^2 matches quadrature."""
        alpha, beta = 2.5, 1.5
        law = MarginalScaleGaussianLaw(DenseCovariance(self.gamma0), alpha, beta)
        y, eta = [1.0, -0.5, 2.0], [0.2, 0.0, 0.5]
        close(self, law.score(y, eta), fd_gradient(law, y, eta), 1e-6)
        shape, scale = law.scale_posterior(y, eta)
        cov = DenseCovariance(self.gamma0)
        q = cov.quadratic_form([a - b for a, b in zip(y, eta, strict=True)])
        self.assertAlmostEqual(shape, alpha + 1.5)
        self.assertAlmostEqual(scale, beta + 0.5 * q, delta=1e-13)

        # Posterior mean of sigma^2 by quadrature of the joint density in t = log s2.
        def log_joint(t: float) -> float:
            return (
                -0.5 * (q * math.exp(-t) + cov.log_det() + 3 * t + 3 * LOG_2PI)
                + alpha * math.log(beta)
                - math.lgamma(alpha)
                - (alpha + 1.0) * t
                - beta * math.exp(-t)
                + t
            )

        norm = simpson(lambda t: math.exp(log_joint(t)), -30.0, 30.0, 6000)
        mean = simpson(lambda t: math.exp(log_joint(t) + t), -30.0, 30.0, 6000) / norm
        self.assertAlmostEqual(mean, scale / (shape - 1.0), delta=1e-7)

    def test_heavier_tails_than_known_scale_gaussian(self) -> None:
        """A gross outlier gets far more mass than under the matching Gaussian."""
        alpha, beta = 3.0, 3.0
        law = MarginalScaleGaussianLaw(ScaledIdentityCovariance(2, 1.0), alpha, beta)
        gauss = GaussianLaw(ScaledIdentityCovariance(2, beta / alpha))
        far = [12.0, 12.0]
        self.assertGreater(
            law.log_likelihood(far, [0.0, 0.0]),
            gauss.log_likelihood(far, [0.0, 0.0]) + 20.0,
        )

    def test_simulation_variance_and_irls(self) -> None:
        """Marginal variance beta/(alpha - 1); IRLS monotone and stationary."""
        alpha, beta, n = 4.0, 3.0, 5000
        law = MarginalScaleGaussianLaw(ScaledIdentityCovariance(n, 1.0), alpha, beta)
        # Entries of one draw share sigma, so use one law per draw here.
        one = MarginalScaleGaussianLaw(ScaledIdentityCovariance(1, 1.0), alpha, beta)
        stream = resolve_stream(51)
        draws = [one.simulate([0.0], seed=stream.child(i))[0] for i in range(n)]
        var = math.fsum(d * d for d in draws) / n
        self.assertLess(abs(var - beta / (alpha - 1.0)) / (beta / (alpha - 1.0)), 0.15)
        self.assertEqual(len(law.simulate([0.0] * n, seed=1)), n)
        design = [[1.0, t / 4.0] for t in range(8)]
        y = [0.0, 0.3, 0.4, 7.0, 0.9, 1.0, 1.4, 1.6]
        fit_law = MarginalScaleGaussianLaw(ScaledIdentityCovariance(8, 1.0), 2.0, 1.0)
        result = maximum_likelihood_linear(design, y, fit_law, max_iterations=300)
        self.assertTrue(result.converged, result.stop_reason)
        objectives = [h.objective for h in result.history]
        self.assertTrue(
            all(
                b <= a + 1e-12 for a, b in zip(objectives, objectives[1:], strict=False)
            )
        )
        self.assertLess(result.gradient_norm, 1e-4)

    def test_missing_mask_uses_submatrix_and_shared_prior(self) -> None:
        """Masking an entry marginalizes with the sub-covariance and the same prior."""
        law = MarginalScaleGaussianLaw(DenseCovariance(self.gamma0), 3.0, 2.0)
        masked = MissingMask(law, [True, False, True])
        value = masked.log_likelihood([1.0, math.nan, 2.0], [0.0, 0.0, 0.5])
        sub = [[2.0, 0.0], [0.0, 1.5]]
        expected = MarginalScaleGaussianLaw(
            DenseCovariance(sub), 3.0, 2.0
        ).log_likelihood([1.0, 2.0], [0.0, 0.5])
        self.assertAlmostEqual(value, expected, delta=1e-12)

    def test_normalizing_constants_matter_for_evidence(self) -> None:
        """Dropping log det and the gamma terms changes comparisons across Gamma0."""
        y, eta = [1.0, -0.5, 2.0], [0.0, 0.0, 0.0]
        small = MarginalScaleGaussianLaw(ScaledIdentityCovariance(3, 0.5), 3.0, 2.0)
        large = MarginalScaleGaussianLaw(ScaledIdentityCovariance(3, 50.0), 3.0, 2.0)
        full = small.log_likelihood(y, eta) - large.log_likelihood(y, eta)
        q = sum(v * v for v in y)
        kernel = lambda g: -(3.0 + 1.5) * math.log(2.0 + 0.5 * q / g)  # noqa: E731
        kernel_only = kernel(0.5) - kernel(50.0)
        self.assertGreater(abs(full - kernel_only), 3.0)

    def test_failures(self) -> None:
        """Invalid hyperparameters, covariance type and data."""
        cov = ScaledIdentityCovariance(2, 1.0)
        for alpha, beta in (
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (math.nan, 1.0),
            (1.0, math.inf),
        ):
            with self.assertRaises(ValueError):
                MarginalScaleGaussianLaw(cov, alpha, beta)
        with self.assertRaises(TypeError):
            MarginalScaleGaussianLaw([[1.0]], 1.0, 1.0)  # type: ignore[arg-type]
        law = MarginalScaleGaussianLaw(cov, 2.0, 1.0)
        with self.assertRaises(ValueError):
            law.log_likelihood([math.nan, 0.0], [0.0, 0.0])
        with self.assertRaises(ValueError):
            law.log_likelihood([0.0], [0.0])


if __name__ == "__main__":
    unittest.main()
