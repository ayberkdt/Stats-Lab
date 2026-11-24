"""Probability content of a ball under a Gaussian law.

Oracles: the isotropic closed form against values pinned from base R 4.6.2
(``pchisq`` with ``ncp``; each pin carries its R expression), the
two-dimensional quadrature and the chi-square series against each other and
against the isotropic form, the one-dimensional case against the normal
distribution function, and a seeded Monte Carlo judged on its binomial
standard error. The tests never call R.

References:
    [REF-RUBEN-1962], [REF-SHEIL-OMUIRCHEARTAIGH-1977], [REF-IMHOF-1961],
    [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995], [REF-FOSTER-ESTES-1992],
    [REF-ALFANO-2005], [REF-ALFANO-2005B], [REF-CHAN-2008],
    [REF-NASA-CARA-HANDBOOK]. Checks [EQ-MVN-DISC-001],
    [EQ-MVN-RICE-001], [EQ-MVN-BALL-001], [EQ-MVN-SMALLDISC-001],
    [EQ-MVN-DILUTION-001].

"""

from __future__ import annotations

import math
import unittest

from probstat.foundations import RandomStream
from probstat.multivariate import (
    MultivariateNormal,
    gaussian_ball_probability,
    gaussian_disc_probability,
    isotropic_gaussian_ball_probability,
    maximum_disc_probability,
    noncentral_chi_square_cdf,
    small_disc_probability,
)


def _rotate(
    mean: tuple[float, float], covariance: tuple[tuple[float, float], ...], angle: float
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
    """Return the mean and covariance rotated by ``angle``.

    References:
        [REF-MARDIA-KENT-BIBBY-1979]. No equation is evaluated.

    """
    c, s = math.cos(angle), math.sin(angle)
    rot = ((c, -s), (s, c))
    new_mean = (
        rot[0][0] * mean[0] + rot[0][1] * mean[1],
        rot[1][0] * mean[0] + rot[1][1] * mean[1],
    )
    inner = tuple(
        tuple(sum(rot[i][k] * covariance[k][j] for k in range(2)) for j in range(2))
        for i in range(2)
    )
    new_cov = tuple(
        tuple(sum(inner[i][k] * rot[j][k] for k in range(2)) for j in range(2))
        for i in range(2)
    )
    return new_mean, new_cov


class IsotropicClosedFormTests(unittest.TestCase):
    """The noncentral chi-square form against R and basic identities.

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Checks [EQ-MVN-RICE-001].

    """

    def test_noncentral_cdf_matches_r_pins(self) -> None:
        """Pinned ``pchisq`` values with a noncentrality parameter agree.

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Checks [EQ-MVN-RICE-001].

        """
        pins = (
            # pchisq(1, 2, ncp = 25/4)
            (1.0, 2, 25.0 / 4.0, 0.03318287957798878),
            # pchisq(9, 2, ncp = 4)
            (9.0, 2, 4.0, 0.785637911837351),
            # pchisq(25, 3, ncp = 9)
            (25.0, 3, 9.0, 0.9592528792140927),
            # pchisq(2, 5, ncp = 1)
            (2.0, 5, 1.0, 0.1043434475962566),
            # pchisq(400, 2, ncp = 400)
            (400.0, 2, 400.0, 0.4900233218590353),
            # pchisq(0.01, 2, ncp = 0)
            (0.01, 2, 0.0, 0.004987520807317687),
        )
        for x, dof, ncp, expected in pins:
            with self.subTest(x=x, dof=dof, ncp=ncp):
                self.assertAlmostEqual(
                    noncentral_chi_square_cdf(x, dof, ncp) / expected, 1.0, places=10
                )

    def test_isotropic_ball_matches_r_pin_and_reports_a_bound(self) -> None:
        """Offset 5, sigma 2, radius 2 gives the pinned Rice probability.

        References:
            [REF-CHAN-2008]. Checks [EQ-MVN-RICE-001].

        """
        # pchisq(2^2 / 2^2, 2, ncp = (5/2)^2)
        result = isotropic_gaussian_ball_probability(5.0, 2.0, 2.0)
        self.assertAlmostEqual(result.probability / 0.03318287957798878, 1.0, places=10)
        self.assertTrue(result.rigorous)
        self.assertLess(result.error_estimate, 1.0e-12 * result.probability + 1e-300)

    def test_central_two_dimensional_case_is_rayleigh(self) -> None:
        """With zero offset the probability is ``1 - exp(-R^2 / (2 s^2))``.

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Checks [EQ-MVN-RICE-001].

        """
        result = isotropic_gaussian_ball_probability(0.0, 1.5, 2.0)
        self.assertAlmostEqual(
            result.probability, -math.expm1(-(2.0**2) / (2 * 1.5**2))
        )

    def test_probability_increases_with_radius_and_decreases_with_offset(self) -> None:
        """Monotonicity in the radius and in the offset.

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Checks [EQ-MVN-RICE-001].

        """
        radii = [
            isotropic_gaussian_ball_probability(3.0, 1.0, r).probability
            for r in (0.5, 1.0, 2.0, 4.0, 8.0)
        ]
        offsets = [
            isotropic_gaussian_ball_probability(d, 1.0, 2.0).probability
            for d in (0.0, 1.0, 2.0, 4.0)
        ]
        self.assertEqual(radii, sorted(radii))
        self.assertEqual(offsets, sorted(offsets, reverse=True))

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-MVN-RICE-001].

        """
        with self.assertRaises(ValueError):
            isotropic_gaussian_ball_probability(-1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            isotropic_gaussian_ball_probability(1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            isotropic_gaussian_ball_probability(1.0, 1.0, 0.0)
        with self.assertRaises(TypeError):
            isotropic_gaussian_ball_probability(1.0, 1.0, 1.0, dimension=2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            noncentral_chi_square_cdf(1.0, 0, 1.0)
        with self.assertRaises(ValueError):
            noncentral_chi_square_cdf(1.0, 2, -1.0)
        with self.assertRaises(TypeError):
            noncentral_chi_square_cdf(1.0, True, 1.0)
        self.assertEqual(noncentral_chi_square_cdf(-1.0, 2, 1.0), 0.0)


class DiscQuadratureTests(unittest.TestCase):
    """The two-dimensional quadrature against independent forms.

    References:
        [REF-FOSTER-ESTES-1992], [REF-CHAN-2008]. Checks [EQ-MVN-DISC-001].

    """

    def test_isotropic_covariance_matches_the_closed_form(self) -> None:
        """Quadrature equals the Rice closed form over a grid of cases.

        References:
            [REF-CHAN-2008]. Checks [EQ-MVN-DISC-001], [EQ-MVN-RICE-001].

        """
        cases = (
            (5.0, 2.0, 2.0),
            (0.0, 1.0, 0.7),
            (3.0, 0.5, 1.0),
            (10.0, 1.0, 12.0),
            (1.0, 3.0, 0.2),
            (40.0, 1.0, 38.5),
        )
        for offset, sigma, radius in cases:
            with self.subTest(offset=offset, sigma=sigma, radius=radius):
                exact = isotropic_gaussian_ball_probability(offset, sigma, radius)
                mean = (0.6 * offset, 0.8 * offset)
                cov = ((sigma**2, 0.0), (0.0, sigma**2))
                result = gaussian_disc_probability(mean, cov, radius)
                self.assertAlmostEqual(
                    result.probability / exact.probability, 1.0, places=8
                )
                self.assertTrue(result.converged)

    def test_matches_the_published_one_dimensional_form(self) -> None:
        """Quadrature equals the error-function integral of the NASA handbook.

        Appendix N of the NASA conjunction assessment handbook (Eq. N-2,
        attributed there to Alfano 2005) writes the probability of the disc of
        combined radius ``R`` about the miss position ``(x_miss, y_miss)`` for
        a covariance with principal sigmas ``(sx, sy)`` as
        ``1 / sqrt(8 pi sx^2) int_(-R)^(R) [erf(y_+) - erf(y_-)]
        exp(-(u + x_miss)^2 / (2 sx^2)) du`` with
        ``y_pm = (y_miss pm sqrt(R^2 - u^2)) / (sqrt(2) sy)``. The integral is
        evaluated here by Simpson's rule in ``u = R sin(theta)`` (no edge
        singularity) and compared with the rotated quadrature of the library.

        References:
            [REF-NASA-CARA-HANDBOOK, Appendix N Eq. N-2],
            [REF-ALFANO-2005]. Checks [EQ-MVN-DISC-001].

        """

        def alfano(sx: float, sy: float, xm: float, ym: float, radius: float) -> float:
            """Return the handbook's one-dimensional integral.

            References:
                [REF-NASA-CARA-HANDBOOK]. Checks [EQ-MVN-DISC-001].

            """
            count = 4000
            lo, hi = -math.pi / 2.0, math.pi / 2.0
            step = (hi - lo) / count

            def integrand(theta: float) -> float:
                """Return the integrand with ``du = R cos(theta) dtheta``.

                References:
                    [REF-NASA-CARA-HANDBOOK]. Checks [EQ-MVN-DISC-001].

                """
                root = radius * math.cos(theta)
                u = radius * math.sin(theta)
                yp = (ym + root) / (math.sqrt(2.0) * sy)
                yn = (ym - root) / (math.sqrt(2.0) * sy)
                gap = math.erf(yp) - math.erf(yn)
                return gap * math.exp(-((u + xm) ** 2) / (2.0 * sx**2)) * root

            total = integrand(lo) + integrand(hi)
            for i in range(1, count):
                total += (4 if i % 2 else 2) * integrand(lo + i * step)
            return total * step / 3.0 / math.sqrt(8.0 * math.pi * sx**2)

        cases = (
            (200.0, 50.0, 120.0, 80.0, 20.0),
            (1.0, 2.0, 3.0, 0.5, 1.0),
            (50.0, 5.0, 10.0, 40.0, 30.0),
            (10.0, 10.0, 0.0, 0.0, 4.0),
        )
        for sx, sy, xm, ym, radius in cases:
            with self.subTest(sx=sx, sy=sy, xm=xm, ym=ym, radius=radius):
                want = alfano(sx, sy, xm, ym, radius)
                got = gaussian_disc_probability(
                    (xm, ym), ((sx**2, 0.0), (0.0, sy**2)), radius
                )
                self.assertAlmostEqual(got.probability / want, 1.0, places=7)

    def test_far_tail_keeps_relative_accuracy(self) -> None:
        """A probability near ``1e-20`` agrees with the series to 8 digits.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-DISC-001], [EQ-MVN-BALL-001].

        """
        mean, cov, radius = (10.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 1.0
        quad = gaussian_disc_probability(mean, cov, radius).probability
        series = gaussian_ball_probability(mean, cov, radius, tolerance=1e-12)
        self.assertLess(quad, 1.0e-18)
        self.assertAlmostEqual(quad / series.probability, 1.0, places=7)

    def test_rotation_and_axis_swap_invariance(self) -> None:
        """Rotating mean and covariance together leaves the probability alone.

        References:
            [REF-MARDIA-KENT-BIBBY-1979]. Checks [EQ-MVN-DISC-001].

        """
        mean, cov = (4.0, -1.5), ((9.0, 2.0), (2.0, 0.6))
        base = gaussian_disc_probability(mean, cov, 3.0).probability
        for angle in (0.3, 1.1, 2.9, -0.7):
            with self.subTest(angle=angle):
                rotated = _rotate(mean, cov, angle)
                value = gaussian_disc_probability(*rotated, 3.0).probability
                self.assertAlmostEqual(value / base, 1.0, places=8)

    def test_elongated_covariance_agrees_with_the_series(self) -> None:
        """Anisotropic cases from short and long ellipses agree with Ruben.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-DISC-001], [EQ-MVN-BALL-001].

        """
        cases = (
            ((1.0, 0.5), ((9.0, 2.0), (2.0, 1.0)), 1.0),
            ((20.0, 3.0), ((9.0, 2.0), (2.0, 1.0)), 5.0),
            ((0.0, 0.0), ((9.0, 2.0), (2.0, 1.0)), 0.3),
            ((-2.0, 6.0), ((100.0, 0.0), (0.0, 0.04)), 4.0),
            ((7.0, 0.1), ((0.01, 0.0), (0.0, 25.0)), 6.0),
        )
        for mean, cov, radius in cases:
            with self.subTest(mean=mean, radius=radius):
                quad = gaussian_disc_probability(mean, cov, radius).probability
                series = gaussian_ball_probability(
                    mean, cov, radius, tolerance=1e-12, max_terms=20000
                )
                self.assertTrue(series.converged)
                self.assertAlmostEqual(quad / series.probability, 1.0, places=7)

    def test_seeded_monte_carlo_on_its_standard_error(self) -> None:
        """The probability agrees with a seeded sample to five standard errors.

        References:
            [REF-FOSTER-ESTES-1992]. Checks [EQ-MVN-DISC-001].

        """
        mean, cov, radius = (1.0, 0.5), ((9.0, 2.0), (2.0, 1.0)), 2.5
        exact = gaussian_disc_probability(mean, cov, radius).probability
        size = 40_000
        draws = MultivariateNormal(mean, cov).sample(size, seed=RandomStream(20260))
        hits = sum(1 for x in draws if x[0] * x[0] + x[1] * x[1] <= radius * radius)
        error = math.sqrt(exact * (1.0 - exact) / size)
        self.assertLess(abs(hits / size - exact), 5.0 * error)

    def test_distant_disc_returns_zero_and_large_disc_returns_one(self) -> None:
        """A disc beyond 40 sigma has probability zero; a huge one has one.

        References:
            [REF-FOSTER-ESTES-1992]. Checks [EQ-MVN-DISC-001].

        """
        cov = ((1.0, 0.0), (0.0, 1.0))
        self.assertEqual(
            gaussian_disc_probability((500.0, 0.0), cov, 1.0).probability, 0.0
        )
        big = gaussian_disc_probability((1.0, 1.0), cov, 50.0).probability
        self.assertAlmostEqual(big, 1.0, places=12)

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-MVN-DISC-001].

        """
        eye = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            gaussian_disc_probability((0.0, 0.0, 0.0), eye, 1.0)
        with self.assertRaises(ValueError):
            gaussian_disc_probability((0.0, 0.0), ((1.0, 2.0), (2.0, 1.0)), 1.0)
        with self.assertRaises(ValueError):
            gaussian_disc_probability((0.0, 0.0), ((1.0, 0.1), (0.2, 1.0)), 1.0)
        with self.assertRaises(ValueError):
            gaussian_disc_probability((0.0, 0.0), eye, -1.0)
        with self.assertRaises(ValueError):
            gaussian_disc_probability((0.0, 0.0), eye, 1.0, tolerance=2.0)
        with self.assertRaises(TypeError):
            gaussian_disc_probability((0.0, "a"), eye, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gaussian_disc_probability((0.0, math.nan), eye, 1.0)


class RubenSeriesTests(unittest.TestCase):
    """The chi-square series in ``n`` dimensions.

    References:
        [REF-RUBEN-1962], [REF-SHEIL-OMUIRCHEARTAIGH-1977]. Checks
        [EQ-MVN-BALL-001].

    """

    def test_one_dimension_is_the_normal_interval(self) -> None:
        """For ``n = 1`` the probability is ``Phi((R-m)/s) - Phi((-R-m)/s)``.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-BALL-001].

        """
        mean, variance, radius = 0.7, 2.3, 1.9
        s = math.sqrt(variance)
        exact = 0.5 * (
            math.erf((radius - mean) / (s * math.sqrt(2.0)))
            - math.erf((-radius - mean) / (s * math.sqrt(2.0)))
        )
        result = gaussian_ball_probability(
            (mean,), ((variance,),), radius, tolerance=1e-13
        )
        self.assertAlmostEqual(result.probability, exact, places=12)

    def test_isotropic_cases_match_r_pins_in_three_and_five_dimensions(self) -> None:
        """Isotropic covariance recovers the pinned noncentral chi-square.

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Checks [EQ-MVN-BALL-001],
            [EQ-MVN-RICE-001].

        """
        # pchisq(25, 3, ncp = 9): offset 3, sigma 1, radius 5.
        three = gaussian_ball_probability(
            (3.0, 0.0, 0.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 5.0, tolerance=1e-13
        )
        self.assertAlmostEqual(three.probability / 0.9592528792140927, 1.0, places=10)
        # pchisq(2, 5, ncp = 1): offset 1, sigma 1, radius sqrt(2).
        eye5 = tuple(tuple(1.0 if i == j else 0.0 for j in range(5)) for i in range(5))
        five = gaussian_ball_probability(
            (0.0, 1.0, 0.0, 0.0, 0.0), eye5, math.sqrt(2.0), tolerance=1e-13
        )
        self.assertAlmostEqual(five.probability / 0.1043434475962566, 1.0, places=10)

    def test_truncation_bound_covers_the_true_error(self) -> None:
        """With a loose tolerance the reported bound exceeds the actual error.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-BALL-001].

        """
        mean, cov, radius = (1.0, 0.5), ((9.0, 2.0), (2.0, 1.0)), 2.0
        exact = gaussian_disc_probability(
            mean, cov, radius, tolerance=1e-13
        ).probability
        for tolerance in (1e-1, 1e-3, 1e-6):
            with self.subTest(tolerance=tolerance):
                result = gaussian_ball_probability(
                    mean, cov, radius, tolerance=tolerance
                )
                self.assertTrue(result.rigorous)
                self.assertLessEqual(
                    abs(result.probability - exact), result.error_estimate + 1e-13
                )
                self.assertGreaterEqual(exact + 1e-13, result.probability)

    def test_series_is_a_lower_partial_sum(self) -> None:
        """Truncation can only lose mass, so fewer terms never exceed more.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-BALL-001].

        """
        args = ((1.0, 0.5), ((9.0, 2.0), (2.0, 1.0)), 2.0)
        values = [
            gaussian_ball_probability(*args, tolerance=1e-14, max_terms=k).probability
            for k in (1, 2, 4, 8, 16)
        ]
        self.assertEqual(values, sorted(values))

    def test_exhausted_terms_report_non_convergence(self) -> None:
        """A single term cannot certify a skewed case and says so.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-BALL-001].

        """
        result = gaussian_ball_probability(
            (0.0, 0.0), ((100.0, 0.0), (0.0, 1.0)), 5.0, max_terms=1
        )
        self.assertFalse(result.converged)
        self.assertGreater(result.error_estimate, 1e-3)

    def test_four_dimensional_anisotropic_case_matches_seeded_monte_carlo(self) -> None:
        """A 4-D correlated case agrees with a seeded sample on its error.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-BALL-001].

        """
        mean = (1.0, -0.5, 0.3, 0.8)
        cov = (
            (4.0, 1.0, 0.0, 0.5),
            (1.0, 2.0, 0.3, 0.0),
            (0.0, 0.3, 1.0, 0.2),
            (0.5, 0.0, 0.2, 0.7),
        )
        radius = 3.0
        exact = gaussian_ball_probability(
            mean, cov, radius, tolerance=1e-12
        ).probability
        size = 40_000
        draws = MultivariateNormal(mean, cov).sample(size, seed=RandomStream(7))
        hits = sum(1 for x in draws if sum(v * v for v in x) <= radius * radius)
        error = math.sqrt(exact * (1.0 - exact) / size)
        self.assertLess(abs(hits / size - exact), 5.0 * error)

    def test_large_offset_survives_underflow_of_the_leading_weight(self) -> None:
        """An offset of 50 sigma (leading weight below ``1e-500``) still works.

        References:
            [REF-RUBEN-1962]. Checks [EQ-MVN-BALL-001].

        """
        cov = ((1.0, 0.0), (0.0, 1.0))
        series = gaussian_ball_probability((50.0, 0.0), cov, 50.0, tolerance=1e-10)
        exact = isotropic_gaussian_ball_probability(50.0, 1.0, 50.0)
        self.assertTrue(series.converged)
        self.assertAlmostEqual(series.probability / exact.probability, 1.0, places=8)

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-MVN-BALL-001].

        """
        eye = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            gaussian_ball_probability((0.0, 0.0), ((1.0, 0.0), (0.0, 0.0)), 1.0)
        with self.assertRaises(ValueError):
            gaussian_ball_probability((0.0, 0.0), eye, 0.0)
        with self.assertRaises(ValueError):
            gaussian_ball_probability((0.0, 0.0), eye, 1.0, tolerance=0.0)
        with self.assertRaises(TypeError):
            gaussian_ball_probability((0.0, 0.0), eye, 1.0, max_terms=2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gaussian_ball_probability((0.0, 0.0), eye, 1.0, max_terms=0)
        with self.assertRaises(ValueError):
            gaussian_ball_probability((0.0,), eye, 1.0)


class SmallDiscAndDilutionTests(unittest.TestCase):
    """The small-disc approximation and the maximum over the scale.

    References:
        [REF-CHAN-2008], [REF-ALFANO-2005]. Checks [EQ-MVN-SMALLDISC-001],
        [EQ-MVN-DILUTION-001].

    """

    def test_error_estimate_tracks_the_true_error_and_scales_as_r_to_the_fourth(
        self,
    ) -> None:
        """The first correction predicts the observed error of ``pi R^2 f(0)``.

        References:
            [REF-CHAN-2008]. Checks [EQ-MVN-SMALLDISC-001].

        """
        mean, cov = (3.0, 1.0), ((4.0, 1.0), (1.0, 2.0))
        errors = []
        for radius in (0.4, 0.2, 0.1):
            exact = gaussian_disc_probability(mean, cov, radius).probability
            approx = small_disc_probability(mean, cov, radius)
            actual = exact - approx.probability
            self.assertFalse(approx.rigorous)
            self.assertAlmostEqual(abs(actual) / approx.error_estimate, 1.0, delta=0.1)
            errors.append(abs(actual))
        self.assertAlmostEqual(errors[0] / errors[1], 16.0, delta=1.0)
        self.assertAlmostEqual(errors[1] / errors[2], 16.0, delta=0.5)

    def test_centred_isotropic_value(self) -> None:
        """At the origin of ``s^2 I`` the approximation is ``R^2 / (2 s^2)``.

        References:
            [REF-CHAN-2008]. Checks [EQ-MVN-SMALLDISC-001].

        """
        value = small_disc_probability((0.0, 0.0), ((4.0, 0.0), (0.0, 4.0)), 0.2)
        self.assertAlmostEqual(value.probability, 0.2**2 / (2.0 * 4.0))

    def test_maximum_over_scale_matches_the_closed_form_for_a_small_disc(self) -> None:
        """For an isotropic covariance the maximum is ``R^2 / (e d^2)``.

        References:
            [REF-ALFANO-2005B]. Checks [EQ-MVN-DILUTION-001].

        """
        result = maximum_disc_probability(
            (10.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 0.1, tolerance=1e-8
        )
        expected = 0.1**2 / (math.e * 10.0**2)
        self.assertAlmostEqual(result.probability / expected, 1.0, delta=1e-3)
        self.assertAlmostEqual(result.scale / 50.0, 1.0, delta=1e-3)

    def test_maximum_over_scale_beats_a_dense_scan_of_the_closed_form(self) -> None:
        """The maximizer is at least as good as a fine grid of Rice values.

        References:
            [REF-ALFANO-2005B]. Checks [EQ-MVN-DILUTION-001],
            [EQ-MVN-RICE-001].

        """
        offset, radius = 6.0, 2.0
        result = maximum_disc_probability(
            (offset, 0.0), ((1.0, 0.0), (0.0, 1.0)), radius, tolerance=1e-8
        )
        scan = max(
            isotropic_gaussian_ball_probability(
                offset, math.sqrt(math.exp(0.5 + 4.5 * i / 40.0)), radius
            ).probability
            for i in range(41)
        )
        self.assertGreaterEqual(result.probability, scan - 1e-12)
        self.assertLess(result.probability - scan, 5e-3 * scan)

    def test_maximum_requires_the_mean_outside_the_disc(self) -> None:
        """A mean inside the disc has supremum one and raises ``ValueError``.

        References:
            [REF-ALFANO-2005B]. Checks [EQ-MVN-DILUTION-001].

        """
        with self.assertRaises(ValueError):
            maximum_disc_probability((1.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 2.0)

    def test_validation(self) -> None:
        """Invalid arguments raise ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-MVN-SMALLDISC-001].

        """
        eye = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            small_disc_probability((0.0, 0.0, 0.0), eye, 1.0)
        with self.assertRaises(ValueError):
            small_disc_probability((0.0, 0.0), eye, 0.0)
        with self.assertRaises(ValueError):
            maximum_disc_probability((5.0, 0.0), eye, 1.0, tolerance=0.0)


if __name__ == "__main__":
    unittest.main()
