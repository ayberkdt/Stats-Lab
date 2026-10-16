"""Tests for the UQ/reliability continuous distribution catalog and its MLEs."""

from __future__ import annotations

import math
import statistics
import unittest
from collections.abc import Callable
from typing import Any

from probstat.distributions import (
    Beta,
    ChiSquare,
    Exponential,
    FisherF,
    Gamma,
    GeneralizedExtremeValue,
    Gumbel,
    Logistic,
    Normal,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    Weibull,
)
from probstat.distributions._continuous import (
    log_normal_interval_mass,
    log_standard_normal_sf,
)
from probstat.foundations import (
    ContinuousProbabilityLaw,
    LogProbabilityDensity,
    QuantileFunction,
    SurvivalFunction,
)
from probstat.inference import fit_gamma_mle, fit_weibull_mle, student_t_cdf
from probstat.modeling import JointInput, RandomInput
from probstat.numerics import (
    composite_simpson,
    digamma,
    log_minus_digamma,
    trigamma,
)

EULER_GAMMA = 0.5772156649015329


def catalog() -> dict[str, Any]:
    """Return one representative, finite-density member of each new family.

    Returns:
        Mapping from a label to a distribution object.

    References:
        Test fixture; no equation is evaluated.

    """
    return {
        "gamma": Gamma(2.5, 1.5),
        "gamma_small_shape": Gamma(0.4, 2.0),
        "beta": Beta(2.0, 5.0),
        "beta_u_shaped": Beta(0.5, 0.7),
        "student_t": StudentT(3.0, 1.0, 2.0),
        "student_t_fractional": StudentT(1.5),
        "student_t_near_normal": StudentT(1e6),
        "chi_square": ChiSquare(7.0),
        "fisher_f": FisherF(5.0, 12.0),
        "logistic": Logistic(-1.0, 0.7),
        "gumbel_max": Gumbel(2.0, 1.5),
        "gumbel_min": Gumbel(2.0, 1.5, kind="min"),
        "triangular": Triangular(0.0, 1.0, 4.0),
        "truncated_normal": TruncatedNormal(1.0, 2.0, lower=-1.0, upper=3.0),
        "half_normal_like": TruncatedNormal(1.0, 2.0, lower=0.0),
        "upper_truncated": TruncatedNormal(0.0, 1.0, upper=0.5),
        "gev_frechet": GeneralizedExtremeValue(0.0, 1.0, 0.2),
        "gev_gumbel": GeneralizedExtremeValue(1.0, 2.0, 0.0),
        "gev_weibull": GeneralizedExtremeValue(0.0, 1.0, -0.3),
    }


def assert_relative(
    case: unittest.TestCase, actual: float, expected: float, tolerance: float
) -> None:
    """Assert ``|actual - expected| <= tolerance * |expected|``.

    Args:
        case: Running test case.
        actual: Computed value.
        expected: Reference value.
        tolerance: Relative tolerance.

    References:
        Test helper; no equation is evaluated.

    """
    case.assertLessEqual(
        abs(actual - expected),
        tolerance * abs(expected),
        f"{actual!r} differs from {expected!r}",
    )


def poisson_upper_sum(shape: int, x: float) -> float:
    """Return ``Q(k, x) = exp(-x) sum_(j<k) x^j / j!`` for integer ``k``.

    Args:
        shape: Positive integer gamma shape.
        x: Non-negative point.

    Returns:
        Upper regularized incomplete gamma value.

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.4]; test reference only.

    """
    terms = [math.exp(-x + j * math.log(x) - math.lgamma(j + 1)) for j in range(shape)]
    return math.fsum(terms)


def transformed_integral(law: Any, lower: float, upper: float) -> float:
    """Integrate a density over ``[lower, upper]`` in a tail-flattening coordinate.

    Finite support ends use ``x = l + (u - l) expit(t)`` or ``x = l + exp(t)``;
    unbounded laws use ``x = m + s sinh(t)``, so heavy tails, singular end
    densities, and narrow peaks are all resolved by composite Simpson.

    Args:
        law: Continuous law with ``pdf`` and ``support``.
        lower: Finite lower integration limit inside the support.
        upper: Finite upper integration limit inside the support.

    Returns:
        Quadrature estimate of ``integral f(x) dx``.

    References:
        Test helper built on [EQ-NUM-SIMP-001].

    """
    support = law.support
    left, right = support.lower, support.upper
    if left is not None and right is not None:
        width = right - left

        def to_x(t: float) -> tuple[float, float]:
            """Return ``x`` and ``dx/dt`` for the logit map.

            References:
                Test helper; no equation is evaluated.

            """
            share = 1.0 / (1.0 + math.exp(-t))
            return left + width * share, width * share * (1.0 - share)

        def to_t(x: float) -> float:
            """Return the logit coordinate of ``x``.

            References:
                Test helper; no equation is evaluated.

            """
            share = (x - left) / width
            return math.log(share) - math.log1p(-share)

    elif left is not None:

        def to_x(t: float) -> tuple[float, float]:
            """Return ``x`` and ``dx/dt`` for the log map.

            References:
                Test helper; no equation is evaluated.

            """
            step = math.exp(t)
            return left + step, step

        def to_t(x: float) -> float:
            """Return the log coordinate of ``x``.

            References:
                Test helper; no equation is evaluated.

            """
            return math.log(x - left)

    elif right is not None:

        def to_x(t: float) -> tuple[float, float]:
            """Return ``x`` and ``dx/dt`` for the reflected log map.

            References:
                Test helper; no equation is evaluated.

            """
            step = math.exp(-t)
            return right - step, step

        def to_t(x: float) -> float:
            """Return the reflected log coordinate of ``x``.

            References:
                Test helper; no equation is evaluated.

            """
            return -math.log(right - x)

    else:
        center = law.quantile(0.5)
        scale = law.quantile(0.75) - law.quantile(0.25)

        def to_x(t: float) -> tuple[float, float]:
            """Return ``x`` and ``dx/dt`` for the sinh map.

            References:
                Test helper; no equation is evaluated.

            """
            return center + scale * math.sinh(t), scale * math.cosh(t)

        def to_t(x: float) -> float:
            """Return the asinh coordinate of ``x``.

            References:
                Test helper; no equation is evaluated.

            """
            return math.asinh((x - center) / scale)

    def integrand(t: float) -> float:
        """Return ``f(x(t)) dx/dt``.

        References:
            Test helper; no equation is evaluated.

        """
        point, jacobian = to_x(t)
        return law.pdf(point) * jacobian

    return composite_simpson(integrand, to_t(lower), to_t(upper), intervals=6000).value


class TotalMassAndRoundTripTests(unittest.TestCase):
    """Check normalization, quantile/CDF round trips, and tail inverses."""

    def test_density_integrates_to_one(self) -> None:
        """Quadrature of every density matches the CDF mass, which is ~1."""
        for label, law in catalog().items():
            with self.subTest(label):
                lower = law.quantile(1e-10)
                upper = law.quantile(1.0 - 1e-10)
                expected = law.cdf(upper) - law.cdf(lower)
                self.assertAlmostEqual(expected, 1.0, delta=1e-9)
                integral = transformed_integral(law, lower, upper)
                self.assertAlmostEqual(integral, expected, delta=1e-6)

    def test_quantile_and_isf_round_trips_reach_far_tails(self) -> None:
        """``F(F^-1(p)) = p`` and ``S(S^-1(q)) = q`` down to ``1e-12``.

        The tolerance is ``1e-9`` relative, widened only by the probability
        carried by one float spacing at the returned point (relevant next to
        a finite support end, where ``x`` itself cannot resolve ``p``).
        """
        probabilities = (1e-12, 1e-8, 1e-3, 0.2, 0.5, 0.8, 0.999)
        for label, law in catalog().items():
            for probability in probabilities:
                with self.subTest(label=label, p=probability):
                    point = law.quantile(probability)
                    survival_point = law.isf(probability)
                    for value, tail in (
                        (point, law.cdf(point)),
                        (survival_point, law.sf(survival_point)),
                    ):
                        spacing = 50.0 * law.pdf(value) * math.ulp(value) / probability
                        tolerance = max(1e-9, spacing)
                        if math.isfinite(tolerance):
                            assert_relative(self, tail, probability, tolerance)

    def test_points_round_trip_through_cdf(self) -> None:
        """``F^-1(F(x)) = x`` across the bulk of each law."""
        for label, law in catalog().items():
            for probability in (0.05, 0.3, 0.6, 0.95):
                with self.subTest(label=label, p=probability):
                    point = law.quantile(probability)
                    self.assertAlmostEqual(
                        law.quantile(law.cdf(point)),
                        point,
                        delta=1e-9 * (1 + abs(point)),
                    )
                    self.assertAlmostEqual(
                        law.isf(law.sf(point)), point, delta=1e-9 * (1 + abs(point))
                    )

    def test_cdf_and_sf_are_complementary(self) -> None:
        """Direct survival functions agree with ``1 - F`` in the bulk."""
        for label, law in catalog().items():
            for probability in (0.1, 0.5, 0.9):
                with self.subTest(label=label, p=probability):
                    point = law.quantile(probability)
                    self.assertAlmostEqual(
                        law.cdf(point) + law.sf(point), 1.0, places=13
                    )

    def test_quantile_endpoints_follow_support(self) -> None:
        """Probability zero and one map to the support ends."""
        self.assertEqual(Gamma(2.0).quantile(0.0), 0.0)
        self.assertEqual(Gamma(2.0).quantile(1.0), math.inf)
        self.assertEqual(Beta(2.0, 3.0).isf(0.0), 1.0)
        self.assertEqual(StudentT(4.0).quantile(0.0), -math.inf)
        self.assertEqual(StudentT(4.0, 3.0).quantile(0.5), 3.0)
        self.assertEqual(Triangular(1.0, 2.0, 5.0).quantile(1.0), 5.0)
        self.assertEqual(TruncatedNormal(0.0, 1.0, lower=-2.0).quantile(0.0), -2.0)
        self.assertEqual(TruncatedNormal(0.0, 1.0, lower=-2.0).isf(0.0), math.inf)
        frechet = GeneralizedExtremeValue(0.0, 2.0, 0.5)
        self.assertEqual(frechet.quantile(0.0), -4.0)
        self.assertEqual(GeneralizedExtremeValue(0.0, 2.0, -0.5).isf(0.0), 4.0)

    def test_protocol_conformance(self) -> None:
        """Every family satisfies the structural continuous-law protocols."""
        for label, law in catalog().items():
            with self.subTest(label):
                self.assertIsInstance(law, ContinuousProbabilityLaw)
                self.assertIsInstance(law, LogProbabilityDensity)
                self.assertIsInstance(law, SurvivalFunction)
                self.assertIsInstance(law, QuantileFunction)


class KnownValueTests(unittest.TestCase):
    """Compare against closed forms, special cases, and table values."""

    def test_gamma_special_cases(self) -> None:
        """Gamma(1, rate) is exponential; integer shapes match Poisson sums."""
        exponential = Exponential(0.7)
        gamma = Gamma(1.0, 0.7)
        for x in (0.0, 0.1, 1.0, 5.0, 40.0):
            self.assertAlmostEqual(gamma.pdf(x), exponential.pdf(x), places=14)
            assert_relative(self, gamma.sf(x), exponential.sf(x), 1e-12)
        for q in (1e-12, 0.3, 0.9):
            assert_relative(self, gamma.isf(q), exponential.isf(q), 1e-11)
        law = Gamma(4.0, 2.0)
        for x in (0.5, 2.0, 10.0, 30.0):
            assert_relative(self, law.sf(x), poisson_upper_sum(4, 2.0 * x), 1e-12)
        self.assertEqual(Gamma.from_scale(3.0, 2.0), Gamma(3.0, 0.5))
        self.assertEqual(Gamma(3.0, 0.5).scale, 2.0)

    def test_chi_square_matches_gamma_and_tables(self) -> None:
        """Chi-square equals Gamma(nu/2, 1/2) and reproduces table values."""
        chi = ChiSquare(10.0)
        gamma = Gamma(5.0, 0.5)
        for x in (0.5, 3.0, 18.0, 120.0):
            self.assertEqual(chi.sf(x), gamma.sf(x))
            assert_relative(self, chi.sf(x), poisson_upper_sum(5, 0.5 * x), 1e-12)
        self.assertAlmostEqual(chi.isf(0.05), 18.307038053275146, places=10)
        self.assertAlmostEqual(ChiSquare(1.0).isf(0.05), 3.841458820694124, places=10)
        self.assertAlmostEqual(ChiSquare(2.0).sf(4.0), math.exp(-2.0), places=15)
        far = ChiSquare(1.0).sf(400.0)
        assert_relative(self, far, math.erfc(math.sqrt(200.0)), 1e-12)

    def test_student_t_closed_forms_and_tables(self) -> None:
        """Cauchy and nu=2 closed forms, table quantiles, and the old helper."""
        cauchy = StudentT(1.0)
        for t in (-1e6, -3.0, 0.2, 1.0, 50.0):
            expected = 0.5 + math.atan(t) / math.pi
            self.assertAlmostEqual(cauchy.cdf(t), expected, places=14)
        assert_relative(self, cauchy.sf(1e6), math.atan(1e-6) / math.pi, 1e-10)
        self.assertAlmostEqual(cauchy.quantile(0.975), 12.706204736174707, places=9)
        two = StudentT(2.0)
        for t in (0.0, 0.7, 4.0, 1e4):
            # 1/2 - t/(2 sqrt(2+t^2)) rewritten without cancellation.
            root = math.sqrt(2.0 + t * t)
            assert_relative(self, two.sf(t), 1.0 / (root * (root + t)), 1e-11)
        self.assertAlmostEqual(StudentT(3.0).isf(0.025), 3.182446305284263, places=9)
        self.assertAlmostEqual(StudentT(10.0).isf(0.025), 2.228138851986274, places=9)
        for t in (-4.0, 0.3, 2.5):
            self.assertAlmostEqual(
                StudentT(6.5).cdf(t), student_t_cdf(t, 6.5), places=13
            )
        # Large nu: both incomplete-beta arguments are formed directly, so the
        # far tail approaches the normal one (difference ~ (z^3+z)/(4 nu)).
        near_normal = StudentT(1e8)
        assert_relative(
            self, near_normal.quantile(1e-12), Normal().quantile(1e-12), 1e-6
        )
        assert_relative(self, near_normal.sf(7.0), Normal().sf(7.0), 1e-5)
        scaled = StudentT(4.0, 10.0, 3.0)
        self.assertAlmostEqual(scaled.cdf(16.0), StudentT(4.0).cdf(2.0), places=15)

    def test_beta_special_cases(self) -> None:
        """Beta(1,1) is uniform and power-law shapes have closed tails."""
        uniform = Uniform(0.0, 1.0)
        flat = Beta(1.0, 1.0)
        for x in (0.0, 0.2, 0.75, 1.0):
            self.assertAlmostEqual(flat.cdf(x), uniform.cdf(x), places=14)
            self.assertAlmostEqual(flat.pdf(x), 1.0, places=14)
        power = Beta(2.5, 1.0)
        self.assertAlmostEqual(power.cdf(0.3), 0.3**2.5, places=14)
        assert_relative(self, power.quantile(1e-12), 1e-12 ** (1.0 / 2.5), 1e-10)
        upper = Beta(1.0, 3.0)
        assert_relative(self, upper.sf(1.0 - 1e-4), 1e-12, 1e-9)
        self.assertAlmostEqual(upper.isf(1e-12), 1.0 - 1e-4, places=12)
        self.assertEqual(Beta(0.5, 2.0).pdf(0.0), math.inf)
        self.assertEqual(Beta(2.0, 2.0).pdf(1.0), 0.0)

    def test_fisher_f_identities(self) -> None:
        """F(1, nu) is a squared t law and F(2, nu2) has a closed tail."""
        for t in (0.5, 2.0, 30.0):
            self.assertAlmostEqual(
                FisherF(1.0, 7.0).sf(t * t), 2.0 * StudentT(7.0).sf(t), places=13
            )
        for x in (0.1, 3.0, 500.0):
            expected = (1.0 + 2.0 * x / 9.0) ** -4.5
            assert_relative(self, FisherF(2.0, 9.0).sf(x), expected, 1e-11)
        self.assertAlmostEqual(FisherF(2.0, 2.0).cdf(3.0), 0.75, places=14)
        self.assertEqual(FisherF(2.0, 6.0).pdf(0.0), 1.0)
        # Integer half-shapes give a finite binomial sum for I_y(5, 2).
        law = FisherF(4.0, 10.0)
        for x in (0.2, 1.5, 40.0):
            y = 10.0 / (10.0 + 4.0 * x)
            expected = math.fsum(
                math.comb(6, j) * y**j * (1.0 - y) ** (6 - j) for j in range(5, 7)
            )
            assert_relative(self, law.sf(x), expected, 1e-12)
        # Printed F tables give F_0.05(5, 10) = 3.33 to two decimals.
        self.assertAlmostEqual(FisherF(5.0, 10.0).isf(0.05), 3.33, delta=5e-3)

    def test_logistic_closed_forms(self) -> None:
        """Logistic CDF, tails, density, and quantiles match closed forms."""
        law = Logistic(1.0, 2.0)
        self.assertAlmostEqual(law.cdf(1.0), 0.5)
        self.assertAlmostEqual(law.pdf(1.0), 1.0 / 8.0)
        assert_relative(self, law.sf(1.0 + 2.0 * 600.0), math.exp(-600.0), 1e-12)
        assert_relative(self, law.cdf(1.0 - 2.0 * 600.0), math.exp(-600.0), 1e-12)
        self.assertAlmostEqual(law.quantile(0.75), 1.0 + 2.0 * math.log(3.0))
        self.assertAlmostEqual(
            law.isf(1e-12), 1.0 + 2.0 * math.log((1 - 1e-12) / 1e-12)
        )

    def test_gumbel_kinds_are_reflections(self) -> None:
        """The minimum law is the reflection of a maximum law."""
        maximum = Gumbel(-2.0, 1.5)
        minimum = Gumbel(2.0, 1.5, kind="min")
        for x in (-5.0, 0.0, 2.0, 9.0):
            self.assertAlmostEqual(minimum.cdf(x), maximum.sf(-x), places=15)
            self.assertAlmostEqual(minimum.pdf(x), maximum.pdf(-x), places=15)
        self.assertAlmostEqual(Gumbel().cdf(0.0), math.exp(-1.0))
        self.assertAlmostEqual(Gumbel().mean, EULER_GAMMA)
        self.assertAlmostEqual(minimum.mean, 2.0 - 1.5 * EULER_GAMMA)
        assert_relative(self, Gumbel().sf(40.0), -math.expm1(-math.exp(-40.0)), 1e-14)

    def test_gev_tends_continuously_to_gumbel(self) -> None:
        """GEV with xi -> 0 reproduces the Gumbel law and its moments."""
        gumbel = Gumbel(1.0, 2.0)
        for xi in (0.0, 1e-300, 1e-14, -1e-12):
            gev = GeneralizedExtremeValue(1.0, 2.0, xi)
            for x in (-3.0, 0.5, 4.0, 30.0):
                self.assertAlmostEqual(gev.cdf(x), gumbel.cdf(x), places=10)
                assert_relative(self, gev.sf(x), gumbel.sf(x), 1e-9)
                assert_relative(self, gev.pdf(x), gumbel.pdf(x), 1e-9)
            for p in (1e-12, 0.5, 1.0 - 1e-9):
                assert_relative(self, gev.quantile(p), gumbel.quantile(p), 1e-9)
            self.assertAlmostEqual(gev.mean, gumbel.mean, places=10)
            self.assertAlmostEqual(gev.variance, gumbel.variance, places=10)

    def test_gev_moment_branches_join_smoothly(self) -> None:
        """Series and closed-form moment branches agree at the switch point."""
        for side in (1.0, -1.0):
            inside = GeneralizedExtremeValue(0.0, 1.0, side * (0.01 - 1e-9))
            outside = GeneralizedExtremeValue(0.0, 1.0, side * (0.01 + 1e-9))
            self.assertAlmostEqual(inside.mean, outside.mean, delta=1e-8)
            self.assertAlmostEqual(inside.variance, outside.variance, delta=1e-8)
        heavy = GeneralizedExtremeValue(0.0, 1.0, 0.25)
        self.assertAlmostEqual(heavy.mean, (math.gamma(0.75) - 1.0) / 0.25)
        self.assertEqual(GeneralizedExtremeValue(0.0, 1.0, 1.0).mean, math.inf)
        self.assertEqual(GeneralizedExtremeValue(0.0, 1.0, 0.5).variance, math.inf)

    def test_gev_support_and_types(self) -> None:
        """Frechet and reversed-Weibull types have one-sided endpoints."""
        frechet = GeneralizedExtremeValue(0.0, 1.0, 0.5)
        self.assertEqual(frechet.endpoint, -2.0)
        self.assertEqual(frechet.cdf(-2.5), 0.0)
        self.assertEqual(frechet.pdf(-2.5), 0.0)
        self.assertTrue(frechet.support.contains(-2.0))
        bounded = GeneralizedExtremeValue(0.0, 1.0, -0.5)
        self.assertEqual(bounded.cdf(2.5), 1.0)
        self.assertEqual(bounded.sf(2.0), 0.0)
        self.assertEqual(bounded.pdf(2.0), 0.0)
        self.assertEqual(GeneralizedExtremeValue(0.0, 1.0, -1.0).pdf(1.0), 1.0)
        # xi = 1/2: F(x) = exp(-(1 + x/2)^-2) in closed form.
        self.assertAlmostEqual(frechet.cdf(2.0), math.exp(-0.25), places=15)
        self.assertAlmostEqual(
            frechet.isf(1e-10), 2.0 * ((1e-10) ** -0.5 - 1.0), delta=1e-4
        )

    def test_triangular_closed_forms(self) -> None:
        """Symmetric and right-angled triangles match elementary results."""
        law = Triangular(0.0, 0.5, 1.0)
        self.assertAlmostEqual(law.cdf(0.25), 0.125)
        self.assertAlmostEqual(law.sf(0.75), 0.125)
        self.assertAlmostEqual(law.pdf(0.5), 2.0)
        self.assertAlmostEqual(law.variance, 1.0 / 24.0)
        right = Triangular(2.0, 2.0, 5.0)
        self.assertAlmostEqual(right.pdf(2.0), 2.0 / 3.0)
        self.assertAlmostEqual(right.quantile(0.0), 2.0)
        self.assertAlmostEqual(right.mean, 3.0)
        far = Triangular(1e9, 1e9 + 1.0, 1e9 + 3.0)
        self.assertAlmostEqual(far.variance, (9.0 - 2.0) / 18.0, places=12)

    def test_truncated_normal_half_normal_and_deep_tail(self) -> None:
        """Half-normal identities and a truncation 40 sigma into the tail."""
        half = TruncatedNormal(0.0, 2.0, lower=0.0)
        self.assertAlmostEqual(half.mean, 2.0 * math.sqrt(2.0 / math.pi), places=14)
        self.assertAlmostEqual(half.variance, 4.0 * (1.0 - 2.0 / math.pi), places=13)
        for x in (0.1, 1.0, 5.0):
            self.assertAlmostEqual(half.cdf(x), math.erf(x / (2.0 * math.sqrt(2.0))))
        deep = TruncatedNormal(0.0, 1.0, lower=40.0)
        alpha = 40.0
        mills = (1.0 / alpha) * (
            1.0 - alpha**-2 + 3.0 * alpha**-4 - 15.0 * alpha**-6 + 105.0 * alpha**-8
        )
        assert_relative(self, deep.mean, 1.0 / mills, 1e-12)
        mills_next = (1.0 / 41.0) * (
            1.0 - 41.0**-2 + 3.0 * 41.0**-4 - 15.0 * 41.0**-6 + 105.0 * 41.0**-8
        )
        expected = math.exp(-(41.0**2 - 40.0**2) / 2.0) * mills_next / mills
        assert_relative(self, deep.sf(41.0), expected, 1e-10)
        self.assertGreater(deep.variance, 0.0)
        self.assertEqual(deep.cdf(39.0), 0.0)
        symmetric = TruncatedNormal(0.0, 1.0, -1.0, 1.0)
        self.assertAlmostEqual(symmetric.mean, 0.0, places=15)
        self.assertAlmostEqual(symmetric.quantile(0.5), 0.0, places=12)

    def test_log_normal_tail_helpers(self) -> None:
        """Log tails join erfc and Mills branches and interval masses agree."""
        below = log_standard_normal_sf(35.0 - 1e-9)
        above = log_standard_normal_sf(35.0)
        self.assertAlmostEqual(below, above, delta=1e-7)
        self.assertAlmostEqual(
            log_standard_normal_sf(2.0),
            math.log(0.5 * math.erfc(2.0 / math.sqrt(2))),
            places=14,
        )
        self.assertAlmostEqual(
            math.exp(log_normal_interval_mass(-1.0, 2.0)),
            Normal().cdf(2.0) - Normal().cdf(-1.0),
            places=15,
        )
        self.assertEqual(log_normal_interval_mass(1.0, 1.0), -math.inf)


class MomentAndSamplingTests(unittest.TestCase):
    """Compare analytic moments with seeded sample moments."""

    def test_sample_moments_match_analytic_moments(self) -> None:
        """Sample means lie within five standard errors of analytic means."""
        laws = {
            "gamma": (Gamma(2.5, 1.5), 20_000),
            "gamma_small": (Gamma(0.3, 4.0), 20_000),
            "beta": (Beta(2.0, 5.0), 20_000),
            "beta_small": (Beta(0.2, 0.3), 20_000),
            "student_t": (StudentT(8.0, 1.0, 2.0), 20_000),
            "chi_square": (ChiSquare(3.0), 20_000),
            "fisher_f": (FisherF(6.0, 20.0), 20_000),
            "logistic": (Logistic(-1.0, 0.7), 20_000),
            "gumbel_min": (Gumbel(2.0, 1.5, kind="min"), 20_000),
            "triangular": (Triangular(0.0, 1.0, 4.0), 20_000),
            "truncated": (TruncatedNormal(1.0, 2.0, lower=0.0, upper=6.0), 4_000),
            "gev": (GeneralizedExtremeValue(0.0, 1.0, 0.1), 20_000),
        }
        for label, (law, size) in laws.items():
            with self.subTest(label):
                sample = law.sample(size, seed=2026)
                self.assertEqual(sample, law.sample(size, seed=2026))
                self.assertEqual(len(sample), size)
                standard_error = math.sqrt(law.variance / size)
                self.assertLess(
                    abs(statistics.fmean(sample) - law.mean), 5.0 * standard_error
                )
                assert_relative(self, statistics.pvariance(sample), law.variance, 0.1)
                support = law.support
                self.assertTrue(all(support.contains(value) for value in sample))

    def test_sampler_matches_cdf_at_quartiles(self) -> None:
        """Empirical CDFs of the gamma-based samplers match analytic quartiles."""
        for law in (Gamma(0.3, 1.0), Beta(0.4, 3.0), StudentT(2.5), FisherF(3.0, 4.0)):
            sample = law.sample(10_000, seed=7)
            for probability in (0.25, 0.5, 0.75):
                with self.subTest(law=law, p=probability):
                    point = law.quantile(probability)
                    fraction = sum(value <= point for value in sample) / len(sample)
                    bound = 4.0 * math.sqrt(probability * (1 - probability) / 10_000)
                    self.assertLess(abs(fraction - probability), bound)

    def test_empty_samples_and_invalid_size(self) -> None:
        """Size zero returns an empty tuple and negative sizes raise."""
        for law in catalog().values():
            self.assertEqual(law.sample(0, seed=1), ())
            with self.assertRaises(ValueError):
                law.sample(-1)

    def test_heavy_tail_moment_conventions(self) -> None:
        """Undefined moments are ``nan`` and divergent moments return ``inf``."""
        self.assertTrue(math.isnan(StudentT(1.0).mean))
        self.assertTrue(math.isnan(StudentT(0.5).variance))
        self.assertEqual(StudentT(1.5).variance, math.inf)
        self.assertEqual(StudentT(3.0, 2.0, 1.0).variance, 3.0)
        self.assertEqual(FisherF(3.0, 2.0).mean, math.inf)
        self.assertEqual(FisherF(3.0, 4.0).variance, math.inf)


class InvalidParameterTests(unittest.TestCase):
    """Reject invalid parameters, arguments, and probabilities explicitly."""

    def test_constructors_reject_invalid_parameters(self) -> None:
        """Non-positive scales, shapes, and degrees of freedom raise."""
        builders: tuple[Callable[[], object], ...] = (
            lambda: Gamma(0.0),
            lambda: Gamma(1.0, -1.0),
            lambda: Gamma.from_scale(1.0, 0.0),
            lambda: Beta(1.0, 0.0),
            lambda: Beta(math.inf, 1.0),
            lambda: StudentT(0.0),
            lambda: StudentT(2.0, 0.0, -1.0),
            lambda: StudentT(2.0, math.nan),
            lambda: ChiSquare(-3.0),
            lambda: FisherF(0.0, 1.0),
            lambda: FisherF(1.0, math.inf),
            lambda: Logistic(0.0, 0.0),
            lambda: Gumbel(0.0, -1.0),
            lambda: Gumbel(0.0, 1.0, kind="median"),  # type: ignore[arg-type]
            lambda: Triangular(1.0, 0.0, 2.0),
            lambda: Triangular(1.0, 1.0, 1.0),
            lambda: TruncatedNormal(0.0, 1.0, 2.0, 1.0),
            lambda: TruncatedNormal(0.0, 0.0),
            lambda: GeneralizedExtremeValue(0.0, 0.0, 0.1),
            lambda: GeneralizedExtremeValue(0.0, 1.0, math.inf),
        )
        for index, build in enumerate(builders):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build()
        with self.assertRaises(TypeError):
            Gamma("2")  # type: ignore[arg-type]

    def test_arguments_and_probabilities_are_validated(self) -> None:
        """NaN points and out-of-range probabilities raise; infinities are limits."""
        for label, law in catalog().items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    law.cdf(math.nan)
                self.assertEqual(law.cdf(math.inf), 1.0)
                self.assertEqual(law.cdf(-math.inf), 0.0)
                self.assertEqual(law.sf(-math.inf), 1.0)
                self.assertEqual(law.pdf(math.inf), 0.0)
                self.assertEqual(law.log_pdf(-math.inf), -math.inf)
                with self.assertRaises(ValueError):
                    law.quantile(1.5)
                with self.assertRaises(ValueError):
                    law.isf(-0.1)
                with self.assertRaises(TypeError):
                    law.pdf("x")


class JointInputIntegrationTests(unittest.TestCase):
    """Use every new law as a marginal inside a Gaussian-copula joint input."""

    def test_nataf_round_trips_for_every_family(self) -> None:
        """``from_standard_normal`` and ``to_standard_normal`` are inverses."""
        partner = RandomInput("partner", Normal(0.0, 1.0))
        far = {
            "gamma",
            "gamma_small_shape",
            "student_t",
            "chi_square",
            "fisher_f",
            "logistic",
            "gumbel_max",
            "gumbel_min",
            "half_normal_like",
            "gev_gumbel",
        }
        for label, law in catalog().items():
            with self.subTest(label):
                joint = JointInput(
                    (RandomInput("x", law), partner),
                    ((1.0, 0.4), (0.4, 1.0)),
                )
                extremes = (-7.5, 7.5) if label in far else (-4.0, 4.0)
                for u in ((0.3, -1.2), (extremes[0], 0.5), (extremes[1], -0.2)):
                    x = joint.from_standard_normal(u)
                    back = joint.to_standard_normal(x)
                    self.assertAlmostEqual(back[0], u[0], delta=1e-7)
                    self.assertAlmostEqual(back[1], u[1], delta=1e-7)
                self.assertTrue(
                    math.isfinite(joint.log_pdf(joint.from_unit((0.3, 0.6))))
                )

    def test_correlated_sampling_uses_new_marginals(self) -> None:
        """Joint samples stay inside each marginal support."""
        joint = JointInput.independent(
            RandomInput("load", Gumbel(10.0, 2.0)),
            RandomInput("strength", Gamma(20.0, 1.0)),
            RandomInput("fraction", Beta(2.0, 3.0)),
        )
        for row in joint.sample(200, seed=11):
            self.assertGreater(row[1], 0.0)
            self.assertTrue(0.0 < row[2] < 1.0)


class MaximumLikelihoodTests(unittest.TestCase):
    """Recover gamma and Weibull parameters from seeded synthetic data."""

    def test_gamma_mle_recovers_parameters(self) -> None:
        """Gamma MLE is close to the truth and solves the score equations."""
        for truth in (Gamma(2.5, 1.7), Gamma(0.35, 12.0), Gamma(80.0, 0.5)):
            with self.subTest(truth=truth):
                data = truth.sample(6_000, seed=42)
                fitted = fit_gamma_mle(data)
                assert_relative(self, fitted.shape, truth.shape, 0.06)
                assert_relative(self, fitted.rate, truth.rate, 0.06)
                count = len(data)
                score_rate = count * fitted.shape / fitted.rate - math.fsum(data)
                score_shape = count * (
                    math.log(fitted.rate) - digamma(fitted.shape)
                ) + math.fsum(math.log(value) for value in data)
                self.assertAlmostEqual(score_rate / math.fsum(data), 0.0, places=10)
                self.assertAlmostEqual(score_shape / count, 0.0, places=9)

    def test_weibull_mle_recovers_parameters(self) -> None:
        """Weibull MLE is close to the truth, scale invariant, and a maximum."""
        for truth in (Weibull(1.8, 120.0), Weibull(0.45, 1e-4), Weibull(9.0, 3.0)):
            with self.subTest(truth=truth):
                data = truth.sample(6_000, seed=5)
                fitted = fit_weibull_mle(data)
                assert_relative(self, fitted.shape, truth.shape, 0.05)
                assert_relative(self, fitted.scale, truth.scale, 0.05)
                rescaled = fit_weibull_mle(tuple(1e6 * value for value in data))
                assert_relative(self, rescaled.shape, fitted.shape, 1e-9)
                assert_relative(self, rescaled.scale, 1e6 * fitted.scale, 1e-9)

                def log_likelihood(
                    shape: float, scale: float, data: tuple[float, ...] = data
                ) -> float:
                    """Return the Weibull log-likelihood of the data.

                    References:
                        [EQ-DIST-WEIB-001]; test helper.

                    """
                    law = Weibull(shape, scale)
                    return math.fsum(law.log_pdf(value) for value in data)

                best = log_likelihood(fitted.shape, fitted.scale)
                for factor in (0.999, 1.001):
                    self.assertGreater(
                        best, log_likelihood(fitted.shape * factor, fitted.scale)
                    )
                    self.assertGreater(
                        best, log_likelihood(fitted.shape, fitted.scale * factor)
                    )

    def test_mle_rejects_invalid_samples(self) -> None:
        """Non-positive, constant, and empty samples raise."""
        for fit in (fit_gamma_mle, fit_weibull_mle):
            with self.subTest(fit=fit.__name__):
                with self.assertRaises(ValueError):
                    fit((1.0, 2.0, 0.0))
                with self.assertRaises(ValueError):
                    fit((3.0, 3.0, 3.0))
                with self.assertRaises(ValueError):
                    fit(())


class DigammaTests(unittest.TestCase):
    """Verify the digamma family against exact values and recurrences."""

    def test_exact_values_and_recurrence(self) -> None:
        """Known values at 1 and 1/2 and the unit-shift recurrences hold."""
        self.assertAlmostEqual(digamma(1.0), -EULER_GAMMA, places=14)
        self.assertAlmostEqual(
            digamma(0.5), -EULER_GAMMA - 2.0 * math.log(2.0), places=14
        )
        self.assertAlmostEqual(trigamma(1.0), math.pi**2 / 6.0, places=14)
        self.assertAlmostEqual(trigamma(0.5), math.pi**2 / 2.0, places=14)
        for x in (0.03, 1.7, 9.99, 10.01, 250.0):
            self.assertAlmostEqual(digamma(x + 1.0), digamma(x) + 1.0 / x, places=11)
            self.assertAlmostEqual(
                trigamma(x) - trigamma(x + 1.0), 1.0 / (x * x), delta=1e-11 / (x * x)
            )
        assert_relative(self, log_minus_digamma(1e8), 0.5e-8 + 1.0 / 12e16, 1e-12)
        with self.assertRaises(ValueError):
            digamma(0.0)
        with self.assertRaises(ValueError):
            trigamma(-1.0)


if __name__ == "__main__":
    unittest.main()
