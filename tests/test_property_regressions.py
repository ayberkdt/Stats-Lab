"""Regression tests for the defects found by the property harness (ADR-0087).

Each test pins one case reported by ``tests/test_distribution_properties.py``
before the corresponding library fix, so the defect cannot return unnoticed
when the random cases change.
"""

from __future__ import annotations

import math
import time
import unittest
from fractions import Fraction

from probstat.distributions import (
    Affine,
    Bernoulli,
    Beta,
    Binomial,
    ContinuousDistribution,
    Empirical,
    FisherF,
    Gamma,
    GeneralizedExtremeValue,
    GeneralizedPareto,
    Geometric,
    Hypergeometric,
    Logistic,
    Mixture,
    NegativeBinomial,
    Normal,
    Poisson,
    Rayleigh,
    StudentT,
    Truncated,
    Uniform,
)
from probstat.multivariate import Dirichlet
from probstat.numerics import integrate


def exact_binomial_cdf(n: int, p: Fraction, k: int) -> Fraction:
    """Return ``P(X <= k)`` for ``Binomial(n, p)`` in exact rationals."""
    return sum(
        (math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1)),
        Fraction(0),
    )


class ExtremeValueEndpointTests(unittest.TestCase):
    """The closed end of ``support`` has ``F = 0`` or ``F = 1`` exactly."""

    def test_gev_upper_endpoint(self) -> None:
        """``1 + xi z`` rounded to ``1e-16`` once gave ``S(end) = 1.3e-90``."""
        law = GeneralizedExtremeValue(
            35.75949426178988, 157.81486900722527, -0.17750755714362115
        )
        end = law.support.upper
        self.assertEqual(law.sf(end), 0.0)
        self.assertEqual(law.cdf(end), 1.0)
        self.assertEqual(law.quantile(1.0), end)
        self.assertGreater(law.sf(math.nextafter(end, -math.inf)), 0.0)

    def test_gev_lower_endpoint(self) -> None:
        """For ``xi > 0`` the lower end has ``F = 0`` exactly."""
        law = GeneralizedExtremeValue(
            35.75949426178988, 157.81486900722527, 0.17750755714362115
        )
        self.assertEqual(law.cdf(law.support.lower), 0.0)

    def test_gpd_upper_endpoint(self) -> None:
        """About a third of random ``xi < 0`` laws had ``S(end) > 0`` before."""
        for mu, sigma, xi in (
            (-3.7, 0.0123, -0.37),
            (12.0, 451.0, -0.8),
            (0.0, 1.0, -0.01),
        ):
            law = GeneralizedPareto(mu, sigma, xi)
            end = law.support.upper
            self.assertEqual(law.sf(end), 0.0)
            self.assertEqual(law.cdf(end), 1.0)


class DensitySaturationTests(unittest.TestCase):
    """A density above the float range is ``inf``, never ``OverflowError``."""

    def test_unbounded_densities_saturate(self) -> None:
        """Gamma and F densities near zero for shapes below one."""
        gamma = Gamma(0.00850284737202794, 0.029997954305956295)
        self.assertEqual(gamma.pdf(1e-320), math.inf)
        self.assertTrue(math.isfinite(gamma.log_pdf(1e-320)))
        fisher = FisherF(0.01960614935003535, 0.01016251171590655)
        self.assertGreaterEqual(fisher.pdf(1e-320), 0.0)

    def test_tiny_scales_saturate(self) -> None:
        """``log f = -log sigma > 709.78`` gives ``inf`` for every family."""
        for law, x in (
            (Normal(0.0, 1e-310), 0.0),
            (StudentT(3.0, 0.0, 1e-310), 0.0),
            (Logistic(0.0, 1e-310), 0.0),
            (Uniform(0.0, 1e-310), 5e-311),
            (Rayleigh(1e-310), 1e-310),
        ):
            with self.subTest(law=law):
                self.assertEqual(law.pdf(x), math.inf)
                self.assertGreater(law.log_pdf(x), 709.0)


class BinomialTests(unittest.TestCase):
    """Distribution function cost, ``Q(1)``, and ``S <= 1``."""

    def test_quantile_one_is_the_upper_end(self) -> None:
        """``Q(1) = n`` even where the floating CDF reaches one at ``k = 7``."""
        law = Binomial(185, 9.54424281556918e-05)
        self.assertEqual(law.quantile(1.0), 185.0)
        self.assertEqual(law.isf(0.0), 185.0)
        self.assertEqual(Binomial(185, 0.0).quantile(1.0), 0.0)

    def test_survival_never_exceeds_one(self) -> None:
        """``S(0)`` summed to ``1 + 2.4e-14`` and made ``isf(1) = 5``."""
        law = Binomial(155, 0.24735096814942786)
        self.assertLessEqual(law.sf(0.0), 1.0)
        self.assertEqual(law.isf(1.0), 0.0)

    def test_large_n_is_fast_and_accurate(self) -> None:
        """``n = 5e6`` took minutes per quantile by direct summation."""
        law = Binomial(5_000_000, 0.5)
        start = time.perf_counter()
        median = law.quantile(0.5)
        self.assertLess(time.perf_counter() - start, 2.0)
        self.assertEqual(median, 2_500_000.0)
        self.assertGreaterEqual(law.cdf(median), 0.5)
        self.assertLess(law.cdf(median - 1.0), 0.5)

    def test_incomplete_beta_branch_matches_exact_sum(self) -> None:
        """Above ``n = 256`` the beta form agrees with exact rational sums."""
        for n, p, k in (
            (300, Fraction(3, 10), 60),
            (300, Fraction(3, 10), 90),
            (1000, Fraction(1, 100), 25),
        ):
            law = Binomial(n, float(p))
            exact = exact_binomial_cdf(n, p, k)
            self.assertAlmostEqual(law.cdf(k) / float(exact), 1.0, delta=1e-12)
            self.assertAlmostEqual(law.sf(k) / float(1 - exact), 1.0, delta=1e-12)


class DegenerateDiscreteTests(unittest.TestCase):
    """``isf(0) = quantile(1)``, the atom of a degenerate law."""

    def test_isf_zero_is_the_largest_mass_point(self) -> None:
        """``Poisson(0)`` and others returned ``inf`` or ``n`` instead of ``0``."""
        for law in (
            Poisson(0.0),
            Geometric(1.0),
            NegativeBinomial(3, 1.0),
            Bernoulli(0.0),
            Binomial(5, 0.0),
        ):
            with self.subTest(law=law):
                self.assertEqual(law.isf(0.0), 0.0)
                self.assertEqual(law.quantile(1.0), 0.0)
        self.assertEqual(Poisson(2.0).isf(0.0), math.inf)
        self.assertEqual(Binomial(5, 1.0).isf(0.0), 5.0)


class HypergeometricTests(unittest.TestCase):
    """Tail sums stop at negligible mass instead of running over the support."""

    def test_large_population_is_fast(self) -> None:
        """``N = 1e6`` took over ten seconds per property case before."""
        law = Hypergeometric(1_000_000, 400_000, 300_000)
        start = time.perf_counter()
        q = law.quantile(0.3)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertGreaterEqual(law.cdf(q), 0.3)
        self.assertLess(law.cdf(q - 1.0), 0.3)
        self.assertAlmostEqual(law.cdf(119_999.0) + law.sf(119_999.0), 1.0, delta=1e-12)

    def test_truncated_tails_are_exact(self) -> None:
        """Against exact rationals for every cut-off of a moderate law."""
        population, successes, draws = 400, 150, 120
        law = Hypergeometric(population, successes, draws)
        total = math.comb(population, draws)
        running = Fraction(0)
        for k in range(0, draws + 1):
            running += Fraction(
                math.comb(successes, k) * math.comb(population - successes, draws - k),
                total,
            )
            if running > 0 and running < 1:
                self.assertAlmostEqual(law.cdf(k) / float(running), 1.0, delta=1e-13)
                self.assertAlmostEqual(law.sf(k) / float(1 - running), 1.0, delta=1e-13)


class EmpiricalEntropyTests(unittest.TestCase):
    """Closed-form entropy of the interpolated empirical law."""

    def test_unequal_gaps_give_a_finite_entropy(self) -> None:
        """The probability-space quadrature returned ``nan`` for this sample."""
        law = Empirical(
            (
                -91.05552528633622,
                -90.94992654624315,
                -90.92182074231448,
                -90.9217153060074,
                -90.84756234555526,
                -90.845461369036,
                -90.76153547962916,
                -90.74771652390518,
            )
        )
        values = law.values
        expected = (
            math.fsum(
                math.log(7 * (b - a)) for a, b in zip(values, values[1:], strict=False)
            )
            / 7
        )
        self.assertAlmostEqual(law.entropy, expected, delta=1e-14)

    def test_special_cases(self) -> None:
        """Uniform on ``[0, 2]``; ties are atoms; the step law has none."""
        self.assertAlmostEqual(
            Empirical([0.0, 1.0, 2.0]).entropy, math.log(2.0), delta=1e-15
        )
        self.assertEqual(Empirical([0.0, 1.0, 1.0, 2.0]).entropy, -math.inf)
        self.assertTrue(math.isnan(Empirical([0.0, 1.0, 3.0], "step").entropy))


class MixtureMomentTests(unittest.TestCase):
    """Closed-form higher moments and an ``x``-space entropy for mixtures."""

    def test_moments_match_the_generic_quadrature(self) -> None:
        """Skewness and kurtosis agree with the mixin's probability-space rule."""
        law = Mixture(
            (Normal(0.0, 1.0), Normal(3.0, 2.0), Gamma(2.0, 1.0)), (0.3, 0.5, 0.2)
        )
        self.assertAlmostEqual(
            law.skewness, ContinuousDistribution.skewness.fget(law), delta=1e-8
        )
        self.assertAlmostEqual(
            law.kurtosis, ContinuousDistribution.kurtosis.fget(law), delta=1e-8
        )

    def test_entropy_with_a_singular_component(self) -> None:
        """A beta component with ``alpha < 1`` inside the support gave ``nan``."""
        law = Mixture(
            (
                Beta(0.8639025137536595, 3.2424165044576956),
                Beta(5.318881321933692, 1.8415092520288223),
                Logistic(3.989258441126017, 0.73529694),
            ),
            (0.6424728360425114, 0.07828846406352073, 0.2792386998939679),
        )
        value = law.entropy
        self.assertTrue(math.isfinite(value))

        def integrand(x: float) -> float:
            density = law.pdf(x)
            return -density * math.log(density) if 0.0 < density < math.inf else 0.0

        # Independent reference: Gauss-Kronrod on a finer partition.
        edges = [-math.inf, -2.0, 0.0, 0.25, 0.5, 0.75, 1.0, 3.0, 5.0, math.inf]
        reference = math.fsum(
            integrate(
                integrand, a, b, method="gauss-kronrod", rel_tol=1e-11, limit=500
            ).value
            for a, b in zip(edges, edges[1:], strict=False)
        )
        self.assertAlmostEqual(value, reference, delta=1e-7)


class EndpointRoundingTests(unittest.TestCase):
    """Support ends computed in one form are not recomputed in another."""

    def test_hypergeometric_quantile_one(self) -> None:
        """``Q(1)`` was 24 for support ``{4, ..., 29}`` (CDF rounds to one)."""
        law = Hypergeometric(160, 29, 33)
        self.assertEqual(law.support.upper, 29)
        self.assertEqual(law.quantile(1.0), 29.0)
        self.assertEqual(law.isf(0.0), 29.0)

    def test_uniform_ends(self) -> None:
        """``upper - (upper - lower)`` returned ``-9.07692593996319``."""
        law = Uniform(-9.076925939963203, 218.95437935413497)
        self.assertEqual(law.isf(1.0), law.lower)
        self.assertEqual(law.quantile(1.0), law.upper)
        self.assertEqual(law.quantile(0.0), law.lower)
        self.assertEqual(law.isf(0.0), law.upper)

    def test_reflected_affine_ends(self) -> None:
        """A negative scale gave ``F(lower) = 3.3e-26`` instead of zero."""
        law = Affine(
            Beta(2.488704171360501, 1.7426237261923136),
            56.57590254502176,
            -1.8694259597812197,
        )
        lower, upper = law.quantile(0.0), law.quantile(1.0)
        self.assertEqual(law.cdf(lower), 0.0)
        self.assertEqual(law.sf(lower), 1.0)
        self.assertEqual(law.cdf(upper), 1.0)
        self.assertEqual(law.sf(upper), 0.0)

    def test_truncated_isf_one(self) -> None:
        """``isf(1)`` returned ``49.66686163951595`` for ``lower = ...593``."""
        for law in (
            Truncated(
                Gamma(10.652048186945171, 0.22101085239310908), 49.66686163951593
            ),
            Truncated(
                Beta(1.3976746954814334, 1.688221777138217),
                0.4871320855632474,
                0.6450705492493728,
            ),
        ):
            with self.subTest(law=law):
                self.assertEqual(law.isf(1.0), law.lower)
                self.assertEqual(law.isf(0.0), law.quantile(1.0))


class BetaEntropyTests(unittest.TestCase):
    """The beta law has a closed-form entropy."""

    def test_small_shape_with_large_partner(self) -> None:
        """``Beta(11.44, 0.128)`` returned ``nan`` from the generic quadrature."""
        law = Beta(11.442234598686214, 0.12810580634856084)
        value = law.entropy
        # Reference from the closed form in 30-digit arithmetic; quadrature
        # cannot serve here because the mass within one ulp of x = 1 is
        # about eps^0.128 ~ 0.01 [REF-HIGHAM-2002, Section 2.1].
        self.assertAlmostEqual(value, -7.429751452966584596, delta=1e-12)
        dirichlet = Dirichlet((11.442234598686214, 0.12810580634856084))
        self.assertAlmostEqual(value, dirichlet.entropy, delta=1e-10)
        self.assertEqual(Beta(1.0, 1.0).entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
