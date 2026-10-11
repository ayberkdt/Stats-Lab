"""Tests for ``probstat.information``.

Discrete identities are checked exactly; closed-form differential entropies
and Gaussian divergences are compared with independent Simpson integration of
``-f log f`` and ``p log(p/q)``; nearest-neighbour estimators are compared
with Gaussian closed forms on seeded samples, with tolerances of about four
Monte Carlo standard deviations (measured over eight seeds at ``n = 400``:
about 0.045 nats for both estimators).
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

from probstat.distributions import (
    Beta,
    Exponential,
    Gamma,
    GeneralizedExtremeValue,
    GeneralizedPareto,
    Gumbel,
    LogNormal,
    Normal,
    Uniform,
    Weibull,
)
from probstat.foundations import RandomStream
from probstat.information import (
    beta_entropy,
    conditional_entropy,
    cross_entropy,
    differential_entropy,
    entropy,
    exponential_entropy,
    gamma_entropy,
    jensen_shannon_divergence,
    joint_entropy,
    kl_divergence,
    kl_multivariate_normal,
    kl_normal,
    kozachenko_leonenko_entropy,
    ksg_mutual_information,
    lognormal_entropy,
    marginals,
    multivariate_normal_entropy,
    mutual_information,
    normal_entropy,
    uniform_entropy,
)
from probstat.numerics import composite_simpson

JOINT = ((0.30, 0.10, 0.05), (0.05, 0.20, 0.30))


def _entropy_integral(
    log_pdf: Callable[[float], float], lower: float, upper: float, *, log_x: bool
) -> float:
    """Return ``-int f log f`` by Simpson's rule, optionally in ``t = log x``."""

    def integrand(t: float) -> float:
        """Return ``-f log f`` (times the Jacobian ``x`` in log coordinates)."""
        x = math.exp(t) if log_x else t
        value = log_pdf(x)
        if value == -math.inf:
            return 0.0
        density = math.exp(value)
        return -density * value * (x if log_x else 1.0)

    if log_x:
        lower, upper = math.log(lower), math.log(upper)
    return composite_simpson(integrand, lower, upper, intervals=20_000).value


def _gaussian_log_pdf(
    x: tuple[float, float],
    mean: tuple[float, float],
    cov: tuple[tuple[float, ...], ...],
) -> float:
    """Return the bivariate normal log-density by explicit 2x2 algebra."""
    det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
    dx, dy = x[0] - mean[0], x[1] - mean[1]
    quad = (cov[1][1] * dx * dx - 2.0 * cov[0][1] * dx * dy + cov[0][0] * dy * dy) / det
    return -math.log(2.0 * math.pi) - 0.5 * math.log(det) - 0.5 * quad


class DiscreteInformationTests(unittest.TestCase):
    """Shannon quantities for finite PMFs."""

    def test_entropy_values_and_units(self) -> None:
        """Dyadic PMF has 1.5 bits; uniform has ``log n``; a point mass has 0."""
        self.assertAlmostEqual(entropy((0.5, 0.25, 0.25), unit="bits"), 1.5, places=14)
        self.assertAlmostEqual(entropy((0.25,) * 4), math.log(4.0), places=14)
        self.assertEqual(entropy((1.0, 0.0, 0.0)), 0.0)
        nats = entropy((0.2, 0.3, 0.5))
        bits = entropy((0.2, 0.3, 0.5), unit="bits")
        self.assertAlmostEqual(bits, nats / math.log(2))
        with self.assertRaises(ValueError):
            entropy((0.5, 0.6))
        with self.assertRaises(ValueError):
            entropy((0.5, 0.5), unit="hartleys")  # type: ignore[arg-type]

    def test_chain_rule_and_mutual_information(self) -> None:
        """``H(X,Y) = H(X) + H(Y|X)`` and ``I = H(X) + H(Y) - H(X,Y) >= 0``."""
        p_x, p_y = marginals(JOINT)
        self.assertAlmostEqual(
            joint_entropy(JOINT), entropy(p_x) + conditional_entropy(JOINT), places=14
        )
        transposed = tuple(zip(*JOINT, strict=True))
        self.assertAlmostEqual(
            joint_entropy(JOINT), entropy(p_y) + conditional_entropy(transposed)
        )
        information = mutual_information(JOINT)
        self.assertGreater(information, 0.0)
        self.assertAlmostEqual(
            information, entropy(p_x) + entropy(p_y) - joint_entropy(JOINT), places=14
        )
        self.assertAlmostEqual(mutual_information(transposed), information, places=14)
        self.assertLessEqual(information, min(entropy(p_x), entropy(p_y)))

    def test_mutual_information_vanishes_under_independence(self) -> None:
        """A product table has ``I = 0`` and ``H(Y|X) = H(Y)``."""
        p_x, p_y = (0.2, 0.8), (0.1, 0.6, 0.3)
        product = tuple(tuple(a * b for b in p_y) for a in p_x)
        self.assertAlmostEqual(mutual_information(product), 0.0, places=15)
        self.assertGreaterEqual(mutual_information(product), 0.0)
        self.assertAlmostEqual(conditional_entropy(product), entropy(p_y), places=14)

    def test_kl_and_jensen_shannon(self) -> None:
        """KL is non-negative, asymmetric, infinite off support; JS is bounded."""
        p, q = (0.5, 0.5), (0.9, 0.1)
        expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
        self.assertAlmostEqual(kl_divergence(p, q), expected, places=14)
        self.assertNotAlmostEqual(kl_divergence(q, p), kl_divergence(p, q))
        self.assertEqual(kl_divergence(p, p), 0.0)
        self.assertEqual(kl_divergence((0.5, 0.5), (1.0, 0.0)), math.inf)
        self.assertAlmostEqual(
            cross_entropy(p, q), entropy(p) + kl_divergence(p, q), places=14
        )
        self.assertAlmostEqual(
            jensen_shannon_divergence((1.0, 0.0), (0.0, 1.0), unit="bits"), 1.0
        )
        js = jensen_shannon_divergence(p, q)
        self.assertAlmostEqual(js, jensen_shannon_divergence(q, p), places=15)
        self.assertGreater(js, 0.0)
        self.assertLess(js, math.log(2.0))
        stream = RandomStream(4).generator()
        for _ in range(50):
            a = [stream.random() for _ in range(4)]
            b = [stream.random() for _ in range(4)]
            first = [x / sum(a) for x in a]
            second = [x / sum(b) for x in b]
            self.assertGreaterEqual(kl_divergence(first, second), 0.0)


class DifferentialEntropyTests(unittest.TestCase):
    """Closed forms against numerical integration."""

    def test_closed_forms_match_integration(self) -> None:
        """Each closed form agrees with Simpson integration of ``-f log f``."""
        cases: tuple[tuple[str, object, float, float, float, bool], ...] = (
            ("normal", Normal(1.0, 0.7), normal_entropy(0.7), -12.0, 14.0, False),
            ("uniform", Uniform(-1.0, 0.5), uniform_entropy(-1.0, 0.5), -1, 0.5, False),
            ("exp", Exponential(2.0), exponential_entropy(2.0), 1e-12, 40.0, True),
            ("gamma", Gamma(2.5, 1.5), gamma_entropy(2.5, 1.5), 1e-10, 60.0, True),
            ("gamma<1", Gamma(0.6, 1.0), gamma_entropy(0.6, 1.0), 1e-30, 80.0, True),
            ("beta", Beta(2.0, 3.0), beta_entropy(2.0, 3.0), 1e-12, 1.0, False),
            (
                "lognormal",
                LogNormal(0.3, 0.7),
                lognormal_entropy(0.3, 0.7),
                1e-6,
                200.0,
                True,
            ),
        )
        for name, law, closed, lower, upper, log_x in cases:
            with self.subTest(name=name):
                numeric = _entropy_integral(
                    law.log_pdf, lower, upper, log_x=log_x  # type: ignore[attr-defined]
                )
                self.assertAlmostEqual(closed, numeric, delta=2e-6)
                self.assertAlmostEqual(differential_entropy(law), closed, places=14)

    def test_extreme_value_entropies(self) -> None:
        """GEV, Gumbel and GPD closed forms agree with integration."""
        gev = GeneralizedExtremeValue(0.0, 2.0, -0.2)
        numeric = _entropy_integral(gev.log_pdf, -15.0, 10.0, log_x=False)
        self.assertAlmostEqual(differential_entropy(gev), numeric, delta=2e-6)
        gumbel = Gumbel(1.0, 1.5)
        numeric = _entropy_integral(gumbel.log_pdf, -10.0, 60.0, log_x=False)
        self.assertAlmostEqual(differential_entropy(gumbel), numeric, delta=2e-6)
        gpd = GeneralizedPareto(0.0, 1.3, 0.3)
        numeric = _entropy_integral(gpd.log_pdf, 1e-12, 1e12, log_x=True)
        self.assertAlmostEqual(gpd.entropy, math.log(1.3) + 1.3, places=14)
        self.assertAlmostEqual(differential_entropy(gpd), numeric, delta=1e-5)
        bounded = GeneralizedPareto(0.0, 1.0, -0.5)
        numeric = _entropy_integral(bounded.log_pdf, 0.0, 2.0, log_x=False)
        self.assertAlmostEqual(bounded.entropy, numeric, delta=1e-6)

    def test_units_negativity_and_dispatch_errors(self) -> None:
        """Bits are nats over ``log 2``; short uniforms have negative entropy."""
        self.assertAlmostEqual(
            normal_entropy(1.0, unit="bits"), normal_entropy(1.0) / math.log(2.0)
        )
        self.assertLess(uniform_entropy(0.0, 0.5), 0.0)
        self.assertAlmostEqual(beta_entropy(1.0, 1.0), 0.0, places=14)
        with self.assertRaises(TypeError):
            differential_entropy(Weibull(2.0, 1.0))
        with self.assertRaises(ValueError):
            normal_entropy(0.0)

    def test_multivariate_normal_entropy_matches_grid(self) -> None:
        """2-D closed form equals a fine trapezoid grid of ``-f log f``."""
        cov = ((1.0, 0.5), (0.5, 2.0))
        step, limit = 0.04, 9.0
        count = int(2 * limit / step)
        total = 0.0
        for i in range(count + 1):
            x = -limit + i * step
            for j in range(count + 1):
                y = -limit + j * step
                value = _gaussian_log_pdf((x, y), (0.0, 0.0), cov)
                total -= math.exp(value) * value
        numeric = total * step * step
        closed = multivariate_normal_entropy(cov)
        self.assertAlmostEqual(closed, numeric, delta=1e-6)
        self.assertAlmostEqual(
            multivariate_normal_entropy(((4.0,),)), normal_entropy(2.0), places=14
        )
        with self.assertRaises(ValueError):
            multivariate_normal_entropy(((1.0, 0.2), (0.3, 1.0)))


class GaussianDivergenceTests(unittest.TestCase):
    """Closed-form Gaussian KL divergences."""

    def test_univariate_kl_matches_integration(self) -> None:
        """``D(N0 || N1)`` agrees with Simpson integration of ``p log(p/q)``."""
        for mu0, s0, mu1, s1 in ((0.0, 1.0, 1.0, 2.0), (2.0, 0.5, -1.0, 1.5)):
            p, q = Normal(mu0, s0), Normal(mu1, s1)

            def integrand(x: float, p: Normal = p, q: Normal = q) -> float:
                """Return ``p(x) log(p(x)/q(x))``."""
                a = p.log_pdf(x)
                return math.exp(a) * (a - q.log_pdf(x))

            numeric = composite_simpson(integrand, -20.0, 20.0, intervals=20_000).value
            with self.subTest(mu0=mu0, s0=s0):
                self.assertAlmostEqual(kl_normal(mu0, s0, mu1, s1), numeric, places=9)
        self.assertEqual(kl_normal(1.0, 2.0, 1.0, 2.0), 0.0)
        self.assertAlmostEqual(
            kl_normal(0.0, 1.0, 1.0, 2.0), math.log(2.0) + 2.0 / 8.0 - 0.5, places=15
        )

    def test_multivariate_kl_reductions_and_grid(self) -> None:
        """Diagonal case sums univariate terms; full case matches a grid."""
        diagonal = kl_multivariate_normal(
            (0.0, 1.0), ((1.0, 0.0), (0.0, 4.0)), (1.0, -1.0), ((2.0, 0.0), (0.0, 0.5))
        )
        expected = kl_normal(0.0, 1.0, 1.0, math.sqrt(2.0)) + kl_normal(
            1.0, 2.0, -1.0, math.sqrt(0.5)
        )
        self.assertAlmostEqual(diagonal, expected, places=13)
        m0, c0 = (0.0, 0.0), ((1.0, 0.3), (0.3, 1.0))
        m1, c1 = (1.0, -1.0), ((2.0, -0.4), (-0.4, 0.8))
        closed = kl_multivariate_normal(m0, c0, m1, c1)
        step, limit = 0.04, 8.0
        count = int(2 * limit / step)
        total = 0.0
        for i in range(count + 1):
            x = -limit + i * step
            for j in range(count + 1):
                point = (x, -limit + j * step)
                a = _gaussian_log_pdf(point, m0, c0)
                total += math.exp(a) * (a - _gaussian_log_pdf(point, m1, c1))
        self.assertAlmostEqual(closed, total * step * step, delta=1e-6)
        self.assertEqual(kl_multivariate_normal(m0, c0, m0, c0), 0.0)
        self.assertGreater(kl_multivariate_normal(m1, c1, m0, c0), 0.0)
        with self.assertRaises(ValueError):
            kl_multivariate_normal((0.0,), ((1.0,),), m1, c1)


class NearestNeighbourEstimatorTests(unittest.TestCase):
    """Kozachenko-Leonenko and KSG estimators on seeded Gaussian samples."""

    @staticmethod
    def _correlated(
        size: int, rho: float, seed: int
    ) -> tuple[list[float], list[float]]:
        """Draw a standard bivariate normal sample with correlation ``rho``."""
        generator = RandomStream(seed).generator()
        xs: list[float] = []
        ys: list[float] = []
        for _ in range(size):
            a = generator.gauss(0.0, 1.0)
            xs.append(a)
            ys.append(rho * a + math.sqrt(1.0 - rho * rho) * generator.gauss(0.0, 1.0))
        return xs, ys

    def test_kozachenko_leonenko_against_normal_entropy(self) -> None:
        """1-D and 2-D estimates are within 0.12 nats of the closed forms."""
        generator = RandomStream(11).generator()
        draws = [generator.gauss(0.0, 2.0) for _ in range(1000)]
        estimate = kozachenko_leonenko_entropy(draws)
        self.assertAlmostEqual(estimate, normal_entropy(2.0), delta=0.12)
        xs, ys = self._correlated(800, 0.6, 12)
        points = list(zip(xs, ys, strict=True))
        closed = multivariate_normal_entropy(((1.0, 0.6), (0.6, 1.0)))
        for metric in ("euclidean", "chebyshev"):
            with self.subTest(metric=metric):
                value = kozachenko_leonenko_entropy(
                    points, k=4, metric=metric  # type: ignore[arg-type]
                )
                self.assertAlmostEqual(value, closed, delta=0.12)
        self.assertAlmostEqual(
            kozachenko_leonenko_entropy(draws, unit="bits"),
            estimate / math.log(2.0),
            places=12,
        )
        with self.assertRaises(ValueError):
            kozachenko_leonenko_entropy([1.0, 1.0, 2.0, 3.0], k=1)
        with self.assertRaises(ValueError):
            kozachenko_leonenko_entropy([1.0, 2.0], k=2)

    def test_ksg_against_gaussian_mutual_information(self) -> None:
        """KSG matches ``-log(1 - rho^2)/2`` and is near zero when independent."""
        xs, ys = self._correlated(800, 0.6, 21)
        closed = -0.5 * math.log(1.0 - 0.36)
        self.assertAlmostEqual(ksg_mutual_information(xs, ys), closed, delta=0.1)
        xs, ys = self._correlated(800, 0.0, 22)
        self.assertAlmostEqual(ksg_mutual_information(xs, ys), 0.0, delta=0.06)
        vectors = [(x, 0.5 * x + y) for x, y in zip(xs, ys, strict=True)]
        self.assertGreater(ksg_mutual_information(xs, vectors, k=4), 0.3)
        with self.assertRaises(ValueError):
            ksg_mutual_information([1.0, 2.0, 3.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            ksg_mutual_information([1.0, 1.0, 2.0], [3.0, 3.0, 4.0], k=1)


if __name__ == "__main__":
    unittest.main()
