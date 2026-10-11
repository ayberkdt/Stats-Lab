"""Tests for determinants, Gaussian conditioning, change of variables, delta method."""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.distributions import LogNormal, Normal
from probstat.linalg import (
    determinant,
    log_abs_determinant,
    log_det_spd,
    lu_solve,
    matmul,
    matvec,
    sandwich,
    transpose,
)
from probstat.multivariate import (
    MultivariateNormal,
    TransformedDensity,
    delta_method,
    transformed_density,
)

COVARIANCE_3 = ((4.0, 1.2, 0.5), (1.2, 2.0, 0.3), (0.5, 0.3, 1.0))
MEAN_3 = (1.0, 2.0, 3.0)


def inverse_by_lu(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Return a general matrix inverse column by column through ``lu_solve``.

    Args:
        matrix: Square non-singular matrix.

    Returns:
        Inverse matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.4]. Test helper only.

    """
    size = len(matrix)
    columns = tuple(
        lu_solve(matrix, tuple(1.0 if row == j else 0.0 for row in range(size)))
        for j in range(size)
    )
    return transpose(columns)


class DeterminantTests(unittest.TestCase):
    """Partial-pivot LU determinants agree with hand-computed values."""

    def test_known_values_sign_and_singularity(self) -> None:
        """A 3x3 determinant, a swap sign, and a singular matrix."""
        tridiagonal = ((2.0, 1.0, 0.0), (1.0, 3.0, 1.0), (0.0, 1.0, 4.0))
        self.assertAlmostEqual(determinant(tridiagonal), 18.0, places=12)
        self.assertAlmostEqual(log_abs_determinant(tridiagonal), math.log(18.0))
        self.assertEqual(determinant(((0.0, 1.0), (1.0, 0.0))), -1.0)
        self.assertAlmostEqual(determinant(((0.0, 2.0), (3.0, 1.0))), -6.0)
        self.assertEqual(determinant(((1.0, 2.0), (2.0, 4.0))), 0.0)
        with self.assertRaises(ValueError):
            log_abs_determinant(((1.0, 2.0), (2.0, 4.0)))

    def test_tiny_scale_is_not_declared_singular(self) -> None:
        """Only exactly vanishing pivots count as singular."""
        tiny = ((1e-20, 0.0), (0.0, 1e-20))
        self.assertAlmostEqual(determinant(tiny) / 1e-40, 1.0, places=12)
        self.assertAlmostEqual(log_abs_determinant(tiny), -40.0 * math.log(10.0))

    def test_product_rule_and_spd_agreement(self) -> None:
        """``det(AB) = det A det B`` and LU agrees with Cholesky on SPD input."""
        a = ((1.0, 2.0, 0.5), (-1.0, 0.3, 2.0), (0.7, -0.4, 1.5))
        b = ((2.0, 0.0, 1.0), (0.5, -1.0, 0.0), (0.0, 3.0, 1.0))
        self.assertAlmostEqual(
            determinant(matmul(a, b)), determinant(a) * determinant(b), places=10
        )
        self.assertAlmostEqual(
            log_abs_determinant(COVARIANCE_3), log_det_spd(COVARIANCE_3), places=13
        )

    def test_malformed_input_is_rejected(self) -> None:
        """Non-square and non-finite matrices raise."""
        with self.assertRaises(ValueError):
            determinant(((1.0, 2.0),))
        with self.assertRaises(ValueError):
            determinant(((1.0, math.nan), (0.0, 1.0)))


class GaussianConditioningTests(unittest.TestCase):
    """Schur-complement conditioning and subset marginals of a Gaussian."""

    def setUp(self) -> None:
        """Create a correlated three-dimensional normal law.

        References:
            [REF-BISHOP-2006, Section 2.3.1]. Test fixture only.

        """
        self.law = MultivariateNormal(MEAN_3, COVARIANCE_3)

    def test_bivariate_closed_form(self) -> None:
        """``X1 | X2 = x`` matches ``mu1 + rho s1/s2 (x - mu2)``, ``s1^2(1-rho^2)``."""
        s1, s2, rho = 2.0, 0.5, -0.6
        law = MultivariateNormal(
            (1.0, -3.0), ((s1 * s1, rho * s1 * s2), (rho * s1 * s2, s2 * s2))
        )
        conditional = law.conditional([1], [-2.2])
        self.assertEqual(conditional.dimension, 1)
        self.assertAlmostEqual(
            conditional.mean[0], 1.0 + rho * s1 / s2 * (-2.2 + 3.0), places=13
        )
        self.assertAlmostEqual(
            conditional.covariance[0][0], s1 * s1 * (1.0 - rho * rho), places=13
        )

    def test_direct_partitioned_formula(self) -> None:
        """Mean and covariance agree with explicit ``Sigma_ab Sigma_bb^-1``."""
        observed, free = (2, 0), (1,)
        values = (3.5, -0.4)
        s_bb = tuple(tuple(COVARIANCE_3[i][j] for j in observed) for i in observed)
        s_ab = tuple(tuple(COVARIANCE_3[i][j] for j in observed) for i in free)
        gain = matmul(s_ab, inverse_by_lu(s_bb))
        residual = tuple(v - MEAN_3[i] for i, v in zip(observed, values, strict=True))
        expected_mean = MEAN_3[1] + matvec(gain, residual)[0]
        expected_variance = COVARIANCE_3[1][1] - matmul(gain, transpose(s_ab))[0][0]
        conditional = self.law.conditional(observed, values)
        self.assertAlmostEqual(conditional.mean[0], expected_mean, places=13)
        self.assertAlmostEqual(
            conditional.covariance[0][0], expected_variance, places=13
        )

    def test_density_chain_rule(self) -> None:
        """``log p(x) = log p(x_b) + log p(x_a | x_b)`` for several partitions."""
        point = (0.3, 2.5, 3.5)
        for observed in ((2,), (0,), (1, 2), (2, 0)):
            with self.subTest(observed=observed):
                free = tuple(i for i in range(3) if i not in observed)
                values = tuple(point[i] for i in observed)
                marginal = self.law.marginal_subset(observed)
                conditional = self.law.conditional(observed, values)
                total = marginal.log_pdf(values) + conditional.log_pdf(
                    tuple(point[i] for i in free)
                )
                self.assertAlmostEqual(total, self.law.log_pdf(point), places=12)

    def test_marginal_subset_order_and_entries(self) -> None:
        """Subset marginals follow the requested order and agree with ``marginal``."""
        subset = self.law.marginal_subset([2, 0])
        self.assertEqual(subset.mean, (3.0, 1.0))
        self.assertEqual(subset.covariance, ((1.0, 0.5), (0.5, 4.0)))
        single = self.law.marginal_subset([1])
        self.assertAlmostEqual(
            single.log_pdf([2.7]), self.law.marginal(1).log_pdf(2.7), places=13
        )

    def test_seeded_sampling_regression_matches_schur_complement(self) -> None:
        """Sample regression slope and residual variance match conditioning."""
        law = MultivariateNormal((0.0, 1.0), ((2.0, 0.8), (0.8, 1.0)))
        draws = law.sample(20_000, seed=2026)
        first = [d[0] for d in draws]
        second = [d[1] for d in draws]
        slope = statistics.covariance(first, second) / statistics.variance(second)
        intercept = statistics.fmean(first) - slope * statistics.fmean(second)
        residual_variance = statistics.variance(
            [x - intercept - slope * y for x, y in zip(first, second, strict=True)]
        )
        conditional = law.conditional([1], [2.0])
        exact_slope = conditional.mean[0] - law.conditional([1], [1.0]).mean[0]
        self.assertAlmostEqual(exact_slope, 0.8, places=13)
        self.assertLess(abs(slope - exact_slope), 0.03)
        self.assertLess(
            abs(residual_variance / conditional.covariance[0][0] - 1.0), 0.04
        )

    def test_invalid_index_requests(self) -> None:
        """Bad index lists and value lengths are rejected."""
        with self.assertRaises(ValueError):
            self.law.marginal_subset([])
        with self.assertRaises(ValueError):
            self.law.marginal_subset([0, 0])
        with self.assertRaises(IndexError):
            self.law.marginal_subset([3])
        with self.assertRaises(TypeError):
            self.law.marginal_subset([True])
        with self.assertRaises(TypeError):
            self.law.marginal_subset(5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.law.conditional([0, 1, 2], [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            self.law.conditional([0], [0.0, 1.0])


class TransformedDensityTests(unittest.TestCase):
    """Change of variables with analytic and finite-difference Jacobians."""

    def setUp(self) -> None:
        """Create the standard bivariate normal law.

        References:
            [REF-CASELLA-BERGER-2002, Section 4.3]. Test fixture only.

        """
        self.standard = MultivariateNormal((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))

    def test_polar_coordinates_give_rayleigh_times_uniform(self) -> None:
        """``(R, Theta)`` of a standard normal pair has ``r e^(-r^2/2) / 2 pi``."""

        def to_cartesian(y: tuple[float, ...]) -> tuple[float, float]:
            """Map polar ``(r, theta)`` to Cartesian ``(x1, x2)``.

            References:
                [REF-CASELLA-BERGER-2002, Section 4.3]. Test helper only.

            """
            return (y[0] * math.cos(y[1]), y[0] * math.sin(y[1]))

        def jacobian(y: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
            """Return ``d(x1, x2) / d(r, theta)``.

            References:
                [REF-CASELLA-BERGER-2002, Section 4.3]. Test helper only.

            """
            r, theta = y
            return (
                (math.cos(theta), -r * math.sin(theta)),
                (math.sin(theta), r * math.cos(theta)),
            )

        analytic = transformed_density(self.standard.log_pdf, to_cartesian, jacobian)
        numeric = transformed_density(self.standard.log_pdf, to_cartesian)
        for r, theta in ((0.4, 0.3), (1.3, 2.0), (2.5, 5.5)):
            expected = math.log(r) - 0.5 * r * r - math.log(2.0 * math.pi)
            self.assertAlmostEqual(analytic([r, theta]), expected, places=13)
            self.assertAlmostEqual(numeric([r, theta]), expected, places=8)

    def test_box_muller_inverse_recovers_bivariate_normal(self) -> None:
        """Uniform density 1 pushed by Box-Muller equals the standard normal."""

        def uniform_log_pdf(u: tuple[float, ...]) -> float:
            """Return the log-density of the unit-square uniform law.

            References:
                [REF-BOX-MULLER-1958]. Test helper only.

            """
            return 0.0 if all(0.0 < c < 1.0 for c in u) else -math.inf

        def inverse(z: tuple[float, ...]) -> tuple[float, float]:
            """Return ``(u1, u2)`` that Box-Muller maps to ``z``.

            References:
                [REF-BOX-MULLER-1958]. Test helper only.

            """
            radius_squared = z[0] * z[0] + z[1] * z[1]
            angle = math.atan2(z[1], z[0]) / (2.0 * math.pi)
            return (math.exp(-0.5 * radius_squared), angle % 1.0)

        def jacobian(z: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
            """Return the analytic Jacobian of the Box-Muller inverse.

            References:
                [REF-BOX-MULLER-1958]. Test helper only.

            """
            u1 = math.exp(-0.5 * (z[0] * z[0] + z[1] * z[1]))
            scale = 2.0 * math.pi * (z[0] * z[0] + z[1] * z[1])
            return ((-z[0] * u1, -z[1] * u1), (-z[1] / scale, z[0] / scale))

        analytic = transformed_density(uniform_log_pdf, inverse, jacobian)
        numeric = transformed_density(uniform_log_pdf, inverse)
        for point in ((0.3, 0.8), (-1.2, 0.4), (-0.7, -1.9), (1.5, -0.6)):
            expected = self.standard.log_pdf(point)
            self.assertAlmostEqual(analytic(point), expected, places=13)
            self.assertAlmostEqual(numeric(point), expected, places=7)

    def test_lognormal_from_normal(self) -> None:
        """``exp`` of ``Normal(mu, sigma)`` has the ``LogNormal`` density."""
        base = Normal(0.5, 0.8)
        target = LogNormal(0.5, 0.8)
        analytic = transformed_density(
            lambda x: base.log_pdf(x[0]),
            lambda y: (math.log(y[0]),),
            lambda y: ((1.0 / y[0],),),
        )
        numeric = transformed_density(
            lambda x: base.log_pdf(x[0]), lambda y: (math.log(y[0]),)
        )
        for y in (0.2, 1.0, 2.0, 7.5):
            self.assertAlmostEqual(analytic([y]), target.log_pdf(y), places=13)
            self.assertAlmostEqual(numeric([y]), target.log_pdf(y), places=8)
            self.assertAlmostEqual(analytic.pdf([y]), target.pdf(y), places=13)

    def test_linear_map_of_gaussian_is_gaussian(self) -> None:
        """``Y = A X + b`` has density ``Normal(A mu + b, A Sigma A')``."""
        a = ((1.0, 2.0, 0.5), (-1.0, 0.3, 2.0), (0.7, -0.4, 1.5))
        shift = (0.5, -1.0, 2.0)
        law = MultivariateNormal(MEAN_3, COVARIANCE_3)
        image = MultivariateNormal(
            tuple(m + s for m, s in zip(matvec(a, MEAN_3), shift, strict=True)),
            sandwich(a, COVARIANCE_3),
        )
        a_inverse = inverse_by_lu(a)

        def inverse(y: tuple[float, ...]) -> tuple[float, ...]:
            """Return ``A^-1 (y - b)``.

            References:
                [REF-GOLUB-VANLOAN-2013, Section 3.4]. Test helper only.

            """
            return lu_solve(a, tuple(v - s for v, s in zip(y, shift, strict=True)))

        analytic = transformed_density(law.log_pdf, inverse, lambda y: a_inverse)
        numeric = transformed_density(law.log_pdf, inverse)
        for point in ((0.0, 0.0, 0.0), (3.0, -2.0, 5.0), (8.5, 4.0, 7.0)):
            self.assertAlmostEqual(analytic(point), image.log_pdf(point), places=11)
            self.assertAlmostEqual(numeric(point), image.log_pdf(point), places=8)

    def test_outside_support_and_error_paths(self) -> None:
        """``-inf`` passes through; malformed maps and steps raise."""
        uniform = transformed_density(
            lambda u: 0.0 if 0.0 < u[0] < 1.0 else -math.inf, lambda y: (y[0] / 2.0,)
        )
        self.assertEqual(uniform.log_pdf([3.0]), -math.inf)
        self.assertEqual(uniform.pdf([3.0]), 0.0)
        self.assertAlmostEqual(uniform([1.0]), -math.log(2.0), places=9)
        with self.assertRaises(ValueError):
            transformed_density(self.standard.log_pdf, lambda y: (y[0],))([1.0, 2.0])
        with self.assertRaises(ValueError):
            transformed_density(lambda x: 0.0, lambda y: (1.0, 2.0))([0.1, 0.2])
        with self.assertRaises(ValueError):
            transformed_density(lambda x: math.nan, lambda y: y)([0.1])
        with self.assertRaises(TypeError):
            TransformedDensity(1, lambda y: y)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            transformed_density(lambda x: 0.0, lambda y: y, relative_step=0.0)
        with self.assertRaises(ValueError):
            transformed_density(
                lambda x: 0.0, lambda y: y, lambda y: ((1.0, 0.0),)
            ).log_pdf([0.5])


class DeltaMethodTests(unittest.TestCase):
    """First-order moment propagation against exact and Monte Carlo values."""

    def test_affine_map_is_exact(self) -> None:
        """For ``g(x) = A x + b`` the delta method returns exact moments."""
        a = ((1.0, -2.0, 0.5), (0.0, 1.0, 3.0))
        shift = (4.0, -1.0)

        def affine(x: tuple[float, ...]) -> tuple[float, ...]:
            """Return ``A x + b``.

            References:
                [REF-GOLUB-VANLOAN-2013, Section 1.1]. Test helper only.

            """
            return tuple(v + s for v, s in zip(matvec(a, x), shift, strict=True))

        result = delta_method(affine, MEAN_3, COVARIANCE_3)
        expected = sandwich(a, COVARIANCE_3)
        self.assertEqual(result.evaluations, 7)
        for i in range(2):
            self.assertAlmostEqual(result.mean[i], affine(MEAN_3)[i], places=13)
            for j in range(2):
                self.assertAlmostEqual(
                    result.covariance[i][j], expected[i][j], places=8
                )

    def test_product_and_ratio_against_monte_carlo(self) -> None:
        """Delta variances agree with seeded Monte Carlo within sampling error."""

        def product_ratio(x: tuple[float, ...]) -> tuple[float, float]:
            """Return ``(x1 x2, x1 / x2)``.

            References:
                [REF-CASELLA-BERGER-2002, Section 5.5.4]. Test helper only.

            """
            return (x[0] * x[1], x[0] / x[1])

        def jacobian(x: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
            """Return the analytic Jacobian of ``product_ratio``.

            References:
                [REF-CASELLA-BERGER-2002, Section 5.5.4]. Test helper only.

            """
            return ((x[1], x[0]), (1.0 / x[1], -x[0] / (x[1] * x[1])))

        law = MultivariateNormal((10.0, 5.0), ((0.04, 0.01), (0.01, 0.09)))
        analytic = delta_method(
            product_ratio, law.mean, law.covariance, jacobian=jacobian
        )
        numeric = delta_method(product_ratio, law.mean, law.covariance)
        self.assertEqual(analytic.evaluations, 1)
        self.assertAlmostEqual(analytic.covariance[0][0], 11.0, places=12)
        self.assertAlmostEqual(analytic.covariance[1][1], 0.0144, places=14)
        self.assertAlmostEqual(analytic.covariance[0][1], -0.32, places=13)
        self.assertAlmostEqual(numeric.covariance[0][0], 11.0, places=7)
        outputs = [product_ratio(x) for x in law.sample(20_000, seed=11)]
        # Second-order bias tr(H Sigma) / 2 that the first-order mean ignores:
        # Sigma_12 for the product, mu1 s2^2 / mu2^3 - Sigma_12 / mu2^2 for the
        # ratio.
        curvature_bias = (0.01, 10.0 * 0.09 / 125.0 - 0.01 / 25.0)
        for index in range(2):
            values = [y[index] for y in outputs]
            variance = analytic.covariance[index][index]
            self.assertLess(abs(statistics.variance(values) / variance - 1.0), 0.06)
            self.assertLess(
                abs(
                    statistics.fmean(values)
                    - analytic.mean[index]
                    - curvature_bias[index]
                ),
                4.0 * analytic.standard_errors[index] / math.sqrt(20_000),
            )

    def test_scalar_logarithm_against_monte_carlo(self) -> None:
        """``log X`` with ``X ~ Normal(10, 0.5)`` has sd near ``0.5 / 10``."""
        result = delta_method(lambda x: (math.log(x[0]),), [10.0], [[0.25]])
        self.assertAlmostEqual(result.standard_errors[0], 0.05, places=9)
        draws = Normal(10.0, 0.5).sample(20_000, seed=5)
        sd = statistics.stdev(math.log(x) for x in draws)
        self.assertLess(abs(sd / 0.05 - 1.0), 0.03)

    def test_invalid_inputs(self) -> None:
        """Non-PSD covariance, shape errors, and non-callables raise."""
        identity_map = lambda x: x  # noqa: E731
        with self.assertRaises(ValueError):
            delta_method(identity_map, [0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(ValueError):
            delta_method(identity_map, [0.0, 0.0], [[1.0]])
        with self.assertRaises(ValueError):
            delta_method(identity_map, [0.0], [[1.0]], jacobian=lambda x: [[1.0, 0.0]])
        with self.assertRaises(TypeError):
            delta_method(3, [0.0], [[1.0]])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            delta_method(identity_map, [], [[1.0]])


if __name__ == "__main__":
    unittest.main()
