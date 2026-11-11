"""Dirichlet, multinomial, multivariate t, Wishart laws and conjugate updates.

Oracles are independent of the code under test: univariate laws from
``probstat.distributions`` for one- and two-category reductions, exact
rational arithmetic (:mod:`fractions`) for mass functions and their moments,
adaptive quadrature of the defining compound integrals, alternative algebraic
routes for the conjugate updates, the chain rule for marginal likelihoods,
and sampling moments with tolerances of four standard errors.

Literal reference values marked ``SCIPY`` were computed outside the test
suite with scipy 1.14.1 (``scipy.stats.dirichlet``, ``multivariate_t``,
``wishart``, ``invwishart``, ``dirichlet_multinomial`` ``logpdf``/``logpmf``
and ``entropy``; ``scipy.special.multigammaln``) on the parameters written
next to them and pasted here; no numerical library is imported.
"""

from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction

from probstat.distributions import Beta, Binomial, Gamma, StudentT
from probstat.foundations import RandomStream
from probstat.linalg import inverse_spd, log_det_spd
from probstat.multivariate import (
    Dirichlet,
    DirichletMultinomial,
    DirichletPosterior,
    InverseWishart,
    Multinomial,
    MultivariateNormal,
    MultivariateStudentT,
    NormalInverseWishart,
    NormalInverseWishartPosterior,
    Wishart,
    dirichlet_posterior,
    multivariate_log_gamma,
    normal_inverse_wishart_posterior,
)
from probstat.numerics import integrate

SCALE_3 = ((2.0, 0.3, -0.4), (0.3, 1.0, 0.2), (-0.4, 0.2, 1.5))


def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """Return every vector of ``parts`` non-negative integers summing to ``total``.

    References:
        Stars-and-bars enumeration; no library code is used.

    """
    result: list[tuple[int, ...]] = []
    for cuts in itertools.combinations(range(total + parts - 1), parts - 1):
        bounds = (-1, *cuts, total + parts - 1)
        result.append(tuple(bounds[i + 1] - bounds[i] - 1 for i in range(parts)))
    return result


def rising(value: Fraction, steps: int) -> Fraction:
    """Return the rising factorial ``value (value + 1) ... (value + steps - 1)``.

    References:
        Elementary definition; exact rational arithmetic.

    """
    product = Fraction(1)
    for step in range(steps):
        product *= value + step
    return product


def exact_multinomial(
    counts: tuple[int, ...], probabilities: tuple[float, ...]
) -> Fraction:
    """Return the multinomial mass in exact rationals of the float probabilities.

    References:
        Definition of the multinomial law; exact rational arithmetic.

    """
    coefficient = Fraction(math.factorial(sum(counts)))
    for count in counts:
        coefficient /= math.factorial(count)
    value = coefficient
    for count, probability in zip(counts, probabilities, strict=True):
        value *= Fraction(probability) ** count
    return value


def close(test: unittest.TestCase, actual: float, expected: float, rel: float) -> None:
    """Assert relative closeness with a readable message.

    References:
        Test helper; no equation.

    """
    test.assertTrue(
        math.isclose(actual, expected, rel_tol=rel, abs_tol=rel),
        f"{actual!r} != {expected!r} (rel {rel})",
    )


class DirichletTests(unittest.TestCase):
    """Density, moments, marginals, aggregation and sampling of Dirichlet."""

    def test_two_categories_reduce_to_beta(self) -> None:
        """``Dir(a, b)`` at ``(x, 1-x)`` equals ``Beta(a, b)`` at ``x``."""
        law = Dirichlet((2.5, 1.7))
        beta = Beta(2.5, 1.7)
        for x in (0.05, 0.3, 0.5, 0.91):
            close(self, law.log_pdf((x, 1.0 - x)), beta.log_pdf(x), 1e-13)
        close(self, law.entropy, beta.entropy, 1e-9)

    def test_log_density_and_entropy_published_values(self) -> None:
        """SCIPY: ``dirichlet([2.5, 0.7, 4.0])`` logpdf at (0.2, 0.3, 0.5), entropy."""
        law = Dirichlet((2.5, 0.7, 4.0))
        close(self, law.log_pdf((0.2, 0.3, 0.5)), 0.48713192462249477, 1e-13)
        close(self, law.entropy, -1.8158985200936097, 1e-13)

    def test_density_integrates_to_one_on_the_simplex(self) -> None:
        """Nested adaptive quadrature over the 2-simplex gives total mass 1."""
        law = Dirichlet((2.5, 1.5, 3.0))

        def inner(x: float) -> float:
            """Integrate over ``y`` in ``[0, 1 - x]`` at fixed ``x``."""
            if x >= 1.0:
                return 0.0
            return integrate(
                lambda y: law.pdf((x, y, max(0.0, 1.0 - x - y)))
                if y < 1.0 - x
                else 0.0,
                0.0,
                1.0 - x,
                rel_tol=1e-11,
            ).value

        total = integrate(inner, 0.0, 1.0, rel_tol=1e-10).value
        close(self, total, 1.0, 1e-8)

    def test_moments_match_beta_marginals_and_aggregates(self) -> None:
        """Means and variances from Beta marginals; covariances from sums."""
        law = Dirichlet((2.5, 0.7, 4.0, 1.3))
        for i in range(4):
            beta = law.marginal(i)
            close(self, law.mean[i], beta.mean, 1e-14)
            close(self, law.covariance[i][i], beta.variance, 1e-14)
        for i, j in itertools.combinations(range(4), 2):
            rest = [k for k in range(4) if k not in (i, j)]
            merged = law.aggregate(((i, j), tuple(rest))).marginal(0)
            covariance = 0.5 * (
                merged.variance - law.covariance[i][i] - law.covariance[j][j]
            )
            close(self, law.covariance[i][j], covariance, 1e-12)
            self.assertEqual(law.covariance[i][j], law.covariance[j][i])

    def test_marginal_and_aggregation_parameters(self) -> None:
        """Marginal ``Beta(alpha_i, alpha_0 - alpha_i)``; group sums of alpha."""
        law = Dirichlet((1.0, 2.0, 3.0, 4.0))
        beta = law.marginal(2)
        self.assertIsInstance(beta, Beta)
        self.assertEqual((beta.alpha, beta.beta), (3.0, 7.0))
        merged = law.aggregate(((3,), (0, 2), (1,)))
        self.assertEqual(merged.alpha, (4.0, 4.0, 2.0))
        self.assertEqual(law.dimension, 4)
        self.assertEqual(law.concentration, 10.0)

    def test_mode_maximizes_density_and_is_nan_when_undefined(self) -> None:
        """The interior mode beats nearby simplex points; else ``nan``."""
        law = Dirichlet((2.5, 3.0, 4.5))
        mode = law.mode
        close(self, math.fsum(mode), 1.0, 1e-15)
        best = law.log_pdf(mode)
        for shift in ((1e-3, -1e-3, 0.0), (0.0, 2e-3, -2e-3), (-1e-3, 0.0, 1e-3)):
            moved = tuple(m + s for m, s in zip(mode, shift, strict=True))
            self.assertLess(law.log_pdf(moved), best)
        self.assertTrue(all(math.isnan(v) for v in Dirichlet((0.5, 2.0)).mode))
        self.assertTrue(all(math.isnan(v) for v in Dirichlet((1.0, 2.0)).mode))

    def test_boundary_and_off_support_points(self) -> None:
        """Faces follow the limit convention; negatives are outside the support."""
        self.assertEqual(Dirichlet((0.5, 2.0, 2.0)).log_pdf((0.0, 0.5, 0.5)), math.inf)
        self.assertEqual(Dirichlet((3.0, 2.0, 2.0)).log_pdf((0.0, 0.5, 0.5)), -math.inf)
        flat = Dirichlet((1.0, 1.0, 1.0))
        close(self, flat.pdf((0.0, 0.25, 0.75)), 2.0, 1e-14)
        self.assertTrue(
            math.isnan(Dirichlet((0.5, 3.0, 1.0)).log_pdf((0.0, 0.0, 1.0)))
        )
        self.assertEqual(flat.log_pdf((-0.1, 0.5, 0.6)), -math.inf)
        with self.assertRaises(ValueError):
            flat.log_pdf((0.2, 0.2, 0.2))
        with self.assertRaises(ValueError):
            flat.log_pdf((0.5, 0.5))
        with self.assertRaises(ValueError):
            flat.log_pdf((math.nan, 0.5, 0.5))

    def test_invalid_parameters_and_arguments(self) -> None:
        """Short, non-positive, non-finite or non-real concentrations raise."""
        for bad in ((1.0,), (1.0, 0.0), (1.0, -2.0), (1.0, math.inf), (1.0, math.nan)):
            with self.assertRaises(ValueError):
                Dirichlet(bad)
        with self.assertRaises(TypeError):
            Dirichlet((1.0, "2"))  # type: ignore[arg-type]
        law = Dirichlet((1.0, 2.0, 3.0))
        with self.assertRaises(IndexError):
            law.marginal(3)
        with self.assertRaises(TypeError):
            law.marginal(True)
        invalid = (((0, 1, 2),), ((0,), (1,)), ((0, 1), (1, 2)), ((0,), (), (1, 2)))
        for groups in invalid:
            with self.assertRaises(ValueError):
                law.aggregate(groups)
        with self.assertRaises(IndexError):
            law.aggregate(((0,), (1, 5)))
        with self.assertRaises(ValueError):
            law.sample(-1, seed=1)
        with self.assertRaises(TypeError):
            law.sample(1, seed=1.5)  # type: ignore[arg-type]

    def test_sampling_moments_within_four_standard_errors(self) -> None:
        """Sample means and variances agree with the closed forms.

        With ``X_i`` in ``[0, 1]``, ``Var((X_i - m_i)^2) <= sigma_i^2``, so the
        standard error of the sample variance is at most ``sigma_i/sqrt(N)``.
        """
        law = Dirichlet((2.5, 0.7, 4.0))
        size = 6000
        draws = law.sample(size, seed=20261111)
        for draw in draws:
            close(self, math.fsum(draw), 1.0, 1e-14)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in draw))
        for i in range(3):
            column = [draw[i] for draw in draws]
            mean = math.fsum(column) / size
            variance = math.fsum((v - law.mean[i]) ** 2 for v in column) / size
            sigma = math.sqrt(law.covariance[i][i])
            self.assertLess(abs(mean - law.mean[i]), 4.0 * sigma / math.sqrt(size))
            self.assertLess(
                abs(variance - law.covariance[i][i]), 4.0 * sigma / math.sqrt(size)
            )

    def test_sampling_is_reproducible_and_stable_for_tiny_concentrations(self) -> None:
        """Equal seeds repeat; ``alpha = 1e-3`` still gives simplex points."""
        law = Dirichlet((1e-3, 1e-3, 2e-3))
        first = law.sample(50, seed=7)
        self.assertEqual(first, law.sample(50, seed=RandomStream(7)))
        for draw in first:
            self.assertTrue(all(math.isfinite(v) and v >= 0.0 for v in draw))
            close(self, math.fsum(draw), 1.0, 1e-14)
        self.assertEqual(law.sample(0, seed=1), ())


class MultinomialTests(unittest.TestCase):
    """Mass function, moments, marginals and sampling of the multinomial."""

    probabilities = (0.2, 0.3, 0.5)

    def test_mass_matches_exact_rationals_and_sums_to_one(self) -> None:
        """Every composition of ``n = 9`` matches the exact rational mass."""
        law = Multinomial(9, self.probabilities)
        exact_total = Fraction(0)
        masses: list[float] = []
        for counts in compositions(9, 3):
            exact = exact_multinomial(counts, law.probabilities)
            exact_total += exact
            masses.append(law.pmf(counts))
            close(self, masses[-1], float(exact), 4e-15)
        self.assertEqual(exact_total, sum(
            (Fraction(p) for p in law.probabilities), Fraction(0)
        ) ** 9)
        close(self, math.fsum(masses), 1.0, 1e-14)

    def test_moments_by_exact_enumeration(self) -> None:
        """``E[X] = n p`` and ``Cov = n (diag p - p p')`` by enumeration."""
        law = Multinomial(6, (0.1, 0.25, 0.4, 0.25))
        support = compositions(6, 4)
        weights = {c: exact_multinomial(c, law.probabilities) for c in support}
        for i in range(4):
            mean = sum(w * c[i] for c, w in weights.items())
            close(self, law.mean[i], float(mean), 1e-14)
            for j in range(4):
                second = sum(w * c[i] * c[j] for c, w in weights.items())
                mean_j = sum(w * c[j] for c, w in weights.items())
                close(self, law.covariance[i][j], float(second - mean * mean_j), 1e-13)

    def test_two_categories_and_marginal_are_binomial(self) -> None:
        """``Mult(n, (p, 1-p))`` is ``Bin(n, p)``; every marginal is binomial."""
        law = Multinomial(12, (0.35, 0.65))
        binomial = Binomial(12, 0.35)
        for k in range(13):
            close(self, law.log_pmf((k, 12 - k)), binomial.log_pmf(k), 1e-13)
        marginal = Multinomial(12, self.probabilities).marginal(1)
        self.assertIsInstance(marginal, Binomial)
        self.assertEqual((marginal.n, marginal.p), (12, 0.3))

    def test_large_n_uses_log_gamma_form_consistently(self) -> None:
        """Above the exact-coefficient limit the log-gamma form is accurate."""
        counts = (400, 700, 1100)
        law = Multinomial(2200, self.probabilities)
        coefficient = math.factorial(2200) // math.prod(
            math.factorial(c) for c in counts
        )
        expected = math.log(coefficient) + math.fsum(
            c * math.log(p) for c, p in zip(counts, law.probabilities, strict=True)
        )
        close(self, law.log_pmf(counts), expected, 1e-12)

    def test_support_conventions_and_count_types(self) -> None:
        """Outside the support is ``-inf``; integral floats are accepted."""
        law = Multinomial(4, (0.5, 0.5, 0.0))
        self.assertEqual(law.log_pmf((1, 1, 1)), -math.inf)
        self.assertEqual(law.log_pmf((2, 2, 1)), -math.inf)
        self.assertEqual(law.log_pmf((5, -1, 0)), -math.inf)
        close(self, law.pmf((2.0, 2.0, 0.0)), 6.0 / 16.0, 1e-15)
        self.assertEqual(Multinomial(0, (0.5, 0.5)).pmf((0, 0)), 1.0)
        with self.assertRaises(TypeError):
            law.pmf((1.5, 2.5, 0))
        with self.assertRaises(TypeError):
            law.pmf((True, 3, 0))
        with self.assertRaises(ValueError):
            law.pmf((4, 0))

    def test_invalid_parameters(self) -> None:
        """Bad trial counts and probability vectors raise."""
        for bad in ((1.0,), (0.5, 0.6), (-0.1, 1.1), (0.5, math.nan)):
            with self.assertRaises(ValueError):
                Multinomial(3, bad)
        with self.assertRaises(ValueError):
            Multinomial(-1, (0.5, 0.5))
        with self.assertRaises(TypeError):
            Multinomial(2.5, (0.5, 0.5))  # type: ignore[arg-type]
        with self.assertRaises(IndexError):
            Multinomial(3, (0.5, 0.5)).marginal(2)

    def test_sampling_moments_and_reproducibility(self) -> None:
        """Sequential binomials give the right means; seeds repeat exactly."""
        law = Multinomial(30, (0.1, 0.0, 0.6, 0.3))
        size = 4000
        draws = law.sample(size, seed=11)
        self.assertEqual(draws, law.sample(size, seed=11))
        self.assertTrue(all(sum(d) == 30 and d[1] == 0 for d in draws))
        for i in (0, 2, 3):
            mean = sum(d[i] for d in draws) / size
            error = math.sqrt(law.covariance[i][i] / size)
            self.assertLess(abs(mean - law.mean[i]), 4.0 * error)
        covariance = math.fsum(
            (d[0] - law.mean[0]) * (d[2] - law.mean[2]) for d in draws
        ) / size
        bound = math.sqrt(
            (law.covariance[0][0] * law.covariance[2][2] + law.covariance[0][2] ** 2)
            / size
        )
        self.assertLess(abs(covariance - law.covariance[0][2]), 5.0 * bound)


class DirichletMultinomialTests(unittest.TestCase):
    """The compound law against its defining integral and the Polya urn."""

    def test_two_categories_match_beta_binomial_integral(self) -> None:
        """``int Bin(k | n, p) Beta(p | a, b) dp`` by adaptive quadrature."""
        law = DirichletMultinomial(7, (2.5, 0.7))
        beta = Beta(2.5, 0.7)
        for k in range(8):
            integral = integrate(
                lambda p, k=k: Binomial(7, p).pmf(k) * beta.pdf(p)
                if 0.0 < p < 1.0
                else 0.0,
                0.0,
                1.0,
                rel_tol=1e-12,
            ).value
            close(self, law.pmf((k, 7 - k)), integral, 1e-9)

    def test_three_categories_match_compound_integral(self) -> None:
        """Nested quadrature of ``Mult(x | n, p) Dir(p | alpha)`` over the simplex."""
        alpha = (2.5, 1.5, 3.0)
        prior = Dirichlet(alpha)
        counts = (2, 1, 3)
        multinomial_coefficient = 60.0

        def integrand(x: float, y: float) -> float:
            """Joint mass-density at ``p = (x, y, 1 - x - y)``."""
            z = 1.0 - x - y
            if x <= 0.0 or y <= 0.0 or z <= 0.0:
                return 0.0
            return (
                multinomial_coefficient
                * x ** counts[0]
                * y ** counts[1]
                * z ** counts[2]
                * prior.pdf((x, y, z))
            )

        def inner(x: float) -> float:
            """Integrate over ``y`` in ``[0, 1 - x]``."""
            if x >= 1.0:
                return 0.0
            return integrate(
                lambda y: integrand(x, y), 0.0, 1.0 - x, rel_tol=1e-11
            ).value

        total = integrate(inner, 0.0, 1.0, rel_tol=1e-10).value
        close(self, DirichletMultinomial(6, alpha).pmf(counts), total, 1e-8)

    def test_integer_concentrations_match_polya_urn_exactly(self) -> None:
        """Rising factorials in exact rationals; masses sum to one."""
        alpha = (1, 2, 3)
        law = DirichletMultinomial(5, alpha)
        total = Fraction(0)
        masses = []
        for counts in compositions(5, 3):
            coefficient = Fraction(math.factorial(5))
            for count in counts:
                coefficient /= math.factorial(count)
            exact = coefficient
            for count, shape in zip(counts, alpha, strict=True):
                exact *= rising(Fraction(shape), count)
            exact /= rising(Fraction(sum(alpha)), 5)
            total += exact
            masses.append(law.pmf(counts))
            close(self, masses[-1], float(exact), 1e-13)
        self.assertEqual(total, 1)
        close(self, math.fsum(masses), 1.0, 1e-13)

    def test_moments_and_overdispersion_by_enumeration(self) -> None:
        """``Cov = n (n + A)/(1 + A) (diag q - q q')`` by summing the mass."""
        law = DirichletMultinomial(6, (2.5, 0.7, 4.0))
        support = compositions(6, 3)
        weights = {c: law.pmf(c) for c in support}
        close(self, law.overdispersion, (6 + 7.2) / (1 + 7.2), 1e-15)
        for i in range(3):
            mean_i = math.fsum(w * c[i] for c, w in weights.items())
            close(self, law.mean[i], mean_i, 1e-12)
            for j in range(3):
                mean_j = math.fsum(w * c[j] for c, w in weights.items())
                second = math.fsum(w * c[i] * c[j] for c, w in weights.items())
                close(self, law.covariance[i][j], second - mean_i * mean_j, 1e-11)
        close(self, law.log_pmf((1, 2, 3)), -3.5650757706759575, 1e-13)  # SCIPY

    def test_large_concentration_tends_to_multinomial(self) -> None:
        """``alpha = c p`` with ``c -> inf`` recovers ``Mult(n, p)``."""
        p = (0.2, 0.3, 0.5)
        law = DirichletMultinomial(8, tuple(1e9 * v for v in p))
        multinomial = Multinomial(8, p)
        for counts in ((1, 2, 5), (0, 0, 8), (3, 3, 2)):
            close(self, law.log_pmf(counts), multinomial.log_pmf(counts), 1e-6)

    def test_sampling_and_invalid_inputs(self) -> None:
        """Composition sampling has the right mean; bad inputs raise."""
        law = DirichletMultinomial(20, (0.5, 1.5, 3.0))
        size = 4000
        draws = law.sample(size, seed=3)
        self.assertEqual(draws, law.sample(size, seed=3))
        self.assertTrue(all(sum(d) == 20 for d in draws))
        for i in range(3):
            mean = sum(d[i] for d in draws) / size
            error = math.sqrt(law.covariance[i][i] / size)
            self.assertLess(abs(mean - law.mean[i]), 4.0 * error)
        self.assertEqual(law.log_pmf((10, 10, 1)), -math.inf)
        with self.assertRaises(ValueError):
            DirichletMultinomial(-1, (1.0, 1.0))
        with self.assertRaises(ValueError):
            DirichletMultinomial(3, (1.0, 0.0))
        with self.assertRaises(ValueError):
            law.pmf((20, 0))


class MultivariateStudentTTests(unittest.TestCase):
    """Density, moments, marginals and sampling of the multivariate t."""

    def test_one_dimension_matches_univariate_student_t(self) -> None:
        """``p = 1`` is the location-scale Student t."""
        law = MultivariateStudentT((1.5,), ((4.0,),), 3.5)
        reference = StudentT(3.5, 1.5, 2.0)
        for x in (-7.0, 0.0, 1.5, 4.2):
            close(self, law.log_pdf((x,)), reference.log_pdf(x), 1e-13)

    def test_published_values_and_subset_marginal(self) -> None:
        """SCIPY: ``multivariate_t`` logpdf in three and two dimensions."""
        law = MultivariateStudentT((1.0, -2.0, 0.5), SCALE_3, 4.5)
        close(self, law.log_pdf((0.3, -1.0, 2.0)), -4.6629722601496155, 1e-13)
        subset = law.marginal_subset((2, 0))
        self.assertEqual(subset.location, (0.5, 1.0))
        self.assertEqual(subset.scale_matrix, ((1.5, -0.4), (-0.4, 2.0)))
        close(self, subset.log_pdf((2.0, 0.3)), -3.3203926072627317, 1e-13)
        marginal = law.marginal(1)
        self.assertIsInstance(marginal, StudentT)
        self.assertEqual((marginal.nu, marginal.mu, marginal.sigma), (4.5, -2.0, 1.0))

    def test_large_dof_tends_to_normal_and_mahalanobis_agrees(self) -> None:
        """``nu -> inf`` gives the Gaussian density; distances coincide."""
        location = (1.0, -2.0, 0.5)
        law = MultivariateStudentT(location, SCALE_3, 1e10)
        normal = MultivariateNormal(location, SCALE_3)
        point = (0.3, -1.0, 2.0)
        close(self, law.log_pdf(point), normal.log_pdf(point), 1e-8)
        close(
            self,
            law.mahalanobis_squared(point),
            normal.mahalanobis_squared(point),
            1e-14,
        )

    def test_moment_conventions(self) -> None:
        """Covariance ``nu/(nu-2) Sigma``; ``inf``/``nan`` where undefined."""
        law = MultivariateStudentT((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0)), 6.0)
        self.assertEqual(law.mean, (0.0, 1.0))
        self.assertEqual(law.covariance, ((3.0, 0.75), (0.75, 1.5)))
        heavy = MultivariateStudentT((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0)), 2.0)
        self.assertEqual(heavy.mean, (0.0, 1.0))
        self.assertEqual(heavy.covariance[0][0], math.inf)
        self.assertTrue(math.isnan(heavy.covariance[0][1]))
        cauchy = MultivariateStudentT((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0)), 1.0)
        self.assertTrue(all(math.isnan(v) for v in cauchy.mean))
        self.assertTrue(all(math.isnan(v) for row in cauchy.covariance for v in row))

    def test_sampling_moments_within_four_standard_errors(self) -> None:
        """Means and variances of ``mu + L z / sqrt(W/nu)`` draws.

        For ``nu = 9`` the marginal excess kurtosis is ``6/(nu - 4)``, so the
        standard error of a sample variance is
        ``sigma^2 sqrt((2 + 6/(nu-4))/N)``.
        """
        law = MultivariateStudentT((1.0, -2.0, 0.5), SCALE_3, 9.0)
        size = 6000
        draws = law.sample(size, seed=99)
        self.assertEqual(draws[:5], law.sample(5, seed=99))
        for i in range(3):
            column = [d[i] for d in draws]
            variance_i = law.covariance[i][i]
            mean = math.fsum(column) / size
            standard_error = math.sqrt(variance_i / size)
            self.assertLess(abs(mean - law.mean[i]), 4.0 * standard_error)
            variance = math.fsum((v - law.mean[i]) ** 2 for v in column) / size
            error = variance_i * math.sqrt((2.0 + 6.0 / 5.0) / size)
            self.assertLess(abs(variance - variance_i), 4.0 * error)

    def test_invalid_inputs(self) -> None:
        """Bad dof, scales, shapes and indices raise."""
        with self.assertRaises(ValueError):
            MultivariateStudentT((0.0,), ((1.0,),), 0.0)
        with self.assertRaises(ValueError):
            MultivariateStudentT((0.0, 0.0), ((1.0, 2.0), (2.0, 1.0)), 3.0)
        with self.assertRaises(ValueError):
            MultivariateStudentT((0.0, 0.0), ((1.0,),), 3.0)
        with self.assertRaises(ValueError):
            MultivariateStudentT((), ((1.0,),), 3.0)
        law = MultivariateStudentT((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 3.0)
        with self.assertRaises(ValueError):
            law.log_pdf((0.0,))
        with self.assertRaises(ValueError):
            law.marginal_subset((0, 0))
        with self.assertRaises(IndexError):
            law.marginal_subset((2,))
        with self.assertRaises(ValueError):
            law.marginal_subset(())


class WishartTests(unittest.TestCase):
    """Multivariate log-gamma, Wishart and inverse-Wishart laws."""

    def test_multivariate_log_gamma(self) -> None:
        """``p = 1`` is ``lgamma``; ``p = 2`` duplication; SCIPY ``multigammaln``."""
        close(self, multivariate_log_gamma(3.3, 1), math.lgamma(3.3), 1e-15)
        expected = 0.5 * math.log(math.pi) + math.lgamma(2.7) + math.lgamma(2.2)
        close(self, multivariate_log_gamma(2.7, 2), expected, 1e-15)
        close(self, multivariate_log_gamma(3.7, 4), 6.279434831814241, 1e-14)
        with self.assertRaises(ValueError):
            multivariate_log_gamma(1.0, 3)
        with self.assertRaises(ValueError):
            multivariate_log_gamma(1.0, 0)

    def test_one_dimension_is_a_gamma_law(self) -> None:
        """``Wishart_1(s, nu)`` is ``Gamma(nu/2, rate 1/(2s))``."""
        law = Wishart(((1.7,),), 4.4)
        reference = Gamma(2.2, 1.0 / 3.4)
        for x in (0.3, 2.0, 9.5):
            close(self, law.log_pdf(((x,),)), reference.log_pdf(x), 1e-13)

    def test_published_densities(self) -> None:
        """SCIPY: ``wishart`` and ``invwishart`` logpdf in two and three dimensions."""
        scale = ((2.0, 0.5), (0.5, 1.0))
        value = ((3.0, 1.0), (1.0, 4.0))
        close(self, Wishart(scale, 5.3).log_pdf(value), -6.2945904247900595, 1e-13)
        close(
            self, InverseWishart(scale, 5.3).log_pdf(value), -13.634874291966282, 1e-13
        )
        value_3 = ((4.0, 0.5, -1.0), (0.5, 2.0, 0.3), (-1.0, 0.3, 3.0))
        close(self, Wishart(SCALE_3, 3.2).log_pdf(value_3), -11.009539271196289, 1e-13)
        close(
            self,
            InverseWishart(SCALE_3, 6.5).log_pdf(value_3),
            -23.566805550306313,
            1e-13,
        )

    def test_inverse_wishart_is_the_inverted_wishart(self) -> None:
        """``f_IW(X | Psi) = f_W(X^-1 | Psi^-1) |X|^-(p+1)``."""
        psi = ((2.0, 0.5), (0.5, 1.0))
        value = ((3.0, 1.0), (1.0, 4.0))
        inverse_law = InverseWishart(psi, 4.1)
        wishart = Wishart(inverse_spd(psi), 4.1)
        expected = wishart.log_pdf(inverse_spd(value)) - 3.0 * log_det_spd(value)
        close(self, inverse_law.log_pdf(value), expected, 1e-12)

    def test_support_and_moments(self) -> None:
        """Off-cone matrices have zero density; means and modes by formula."""
        scale = ((2.0, 0.5), (0.5, 1.0))
        law = Wishart(scale, 5.0)
        self.assertEqual(law.log_pdf(((1.0, 2.0), (2.0, 1.0))), -math.inf)
        with self.assertRaises(ValueError):
            law.log_pdf(((1.0, 0.2), (0.3, 1.0)))
        with self.assertRaises(ValueError):
            law.log_pdf(((1.0,),))
        self.assertEqual(law.mean, ((10.0, 2.5), (2.5, 5.0)))
        self.assertEqual(law.mode, ((4.0, 1.0), (1.0, 2.0)))
        self.assertTrue(math.isnan(Wishart(scale, 2.5).mode[0][0]))
        inverse = InverseWishart(scale, 6.0)
        self.assertEqual(inverse.mean, ((2.0 / 3.0, 0.5 / 3.0), (0.5 / 3.0, 1.0 / 3.0)))
        self.assertEqual(inverse.mode, ((2.0 / 9.0, 0.5 / 9.0), (0.5 / 9.0, 1.0 / 9.0)))
        divergent = InverseWishart(scale, 3.0).mean
        self.assertEqual(divergent[0][0], math.inf)
        self.assertTrue(math.isnan(divergent[0][1]))
        for law_mode, density in ((law.mode, law), (inverse.mode, inverse)):
            best = density.log_pdf(law_mode)
            for delta in (((1e-2, 0.0), (0.0, 0.0)), ((0.0, 1e-2), (1e-2, 0.0))):
                for sign in (1.0, -1.0):
                    moved = tuple(
                        tuple(m + sign * d for m, d in zip(mr, dr, strict=True))
                        for mr, dr in zip(law_mode, delta, strict=True)
                    )
                    self.assertLess(density.log_pdf(moved), best)

    def test_bartlett_sampling_moments(self) -> None:
        """Wishart draws and inverted inverse-Wishart draws have mean ``nu S``.

        ``Var(W_ij) = nu (S_ij^2 + S_ii S_jj)`` gives the standard errors.
        """
        scale = ((2.0, 0.5), (0.5, 1.0))
        size = 3000
        for draws, target in (
            (Wishart(scale, 4.5).sample(size, seed=5), scale),
            (
                tuple(
                    inverse_spd(d)
                    for d in InverseWishart(inverse_spd(scale), 4.5).sample(
                        size, seed=6
                    )
                ),
                scale,
            ),
        ):
            for i in range(2):
                for j in range(2):
                    mean = math.fsum(d[i][j] for d in draws) / size
                    variance = 4.5 * (target[i][j] ** 2 + target[i][i] * target[j][j])
                    self.assertLess(
                        abs(mean - 4.5 * target[i][j]),
                        4.0 * math.sqrt(variance / size),
                    )
            self.assertTrue(all(d[0][1] == d[1][0] for d in draws))
        self.assertEqual(
            Wishart(scale, 4.5).sample(3, seed=5), Wishart(scale, 4.5).sample(3, seed=5)
        )
        self.assertEqual(InverseWishart(scale, 4.5).sample(0, seed=1), ())

    def test_invalid_parameters(self) -> None:
        """Degrees of freedom ``<= p - 1`` and non-SPD scales raise."""
        with self.assertRaises(ValueError):
            Wishart(((1.0, 0.0), (0.0, 1.0)), 1.0)
        with self.assertRaises(ValueError):
            InverseWishart(((1.0, 0.0), (0.0, 1.0)), 0.5)
        with self.assertRaises(ValueError):
            Wishart(((1.0, 2.0), (2.0, 1.0)), 5.0)
        with self.assertRaises(ValueError):
            InverseWishart(((1.0, 0.1), (0.0, 1.0)), 5.0)
        with self.assertRaises(TypeError):
            Wishart(((1.0,),), "3")  # type: ignore[arg-type]


class DirichletConjugacyTests(unittest.TestCase):
    """Dirichlet-categorical updates and their predictive laws."""

    def test_posterior_evidence_and_predictive(self) -> None:
        """``alpha + x``; evidence by the sequential chain rule in rationals."""
        prior = Dirichlet((1.0, 2.0, 3.0))
        result = dirichlet_posterior(prior, (3, 0, 2))
        self.assertIsInstance(result, DirichletPosterior)
        self.assertEqual(result.posterior.alpha, (4.0, 2.0, 5.0))
        self.assertEqual(result.total, 5)
        sequence = (0, 2, 0, 2, 0)
        counts = [Fraction(1), Fraction(2), Fraction(3)]
        probability = Fraction(1)
        for category in sequence:
            probability *= counts[category] / sum(counts)
            counts[category] += 1
        close(self, result.log_marginal_likelihood, math.log(probability), 1e-14)
        predictive = result.predictive(4)
        self.assertIsInstance(predictive, DirichletMultinomial)
        self.assertEqual(predictive.alpha, (4.0, 2.0, 5.0))
        close(self, result.predictive_probabilities[0], 4.0 / 11.0, 1e-15)
        unordered = DirichletMultinomial(5, prior.alpha).log_pmf((3, 0, 2))
        close(
            self,
            result.log_marginal_likelihood + math.log(10.0),
            unordered,
            1e-13,
        )

    def test_sequential_updates_equal_one_batch(self) -> None:
        """Updating twice equals updating once with the summed counts."""
        prior = Dirichlet((0.5, 0.5, 0.5, 0.5))
        first = dirichlet_posterior(prior, (2, 0, 1, 4))
        second = dirichlet_posterior(first.posterior, (1, 3, 0, 0))
        batch = dirichlet_posterior(prior, (3, 3, 1, 4))
        self.assertEqual(second.posterior.alpha, batch.posterior.alpha)
        close(
            self,
            first.log_marginal_likelihood + second.log_marginal_likelihood,
            batch.log_marginal_likelihood,
            1e-13,
        )
        empty = dirichlet_posterior(prior, (0, 0, 0, 0))
        self.assertEqual(empty.posterior, prior)
        self.assertEqual(empty.log_marginal_likelihood, 0.0)

    def test_invalid_updates(self) -> None:
        """Negative counts, wrong lengths and wrong prior types raise."""
        prior = Dirichlet((1.0, 1.0))
        with self.assertRaises(ValueError):
            dirichlet_posterior(prior, (1, -1))
        with self.assertRaises(ValueError):
            dirichlet_posterior(prior, (1, 1, 1))
        with self.assertRaises(TypeError):
            dirichlet_posterior(prior, (1.5, 1))
        with self.assertRaises(TypeError):
            dirichlet_posterior((1.0, 1.0), (1, 1))  # type: ignore[arg-type]


class NormalInverseWishartTests(unittest.TestCase):
    """Normal-inverse-Wishart prior, posterior, evidence and predictive law."""

    prior = NormalInverseWishart((0.5, -1.0), 2.0, ((1.5, 0.3), (0.3, 0.8)), 5.0)
    data = ((1.0, -0.5), (0.2, -1.4), (1.6, 0.1), (0.9, -0.8))

    def test_posterior_matches_raw_moment_route_in_rationals(self) -> None:
        """``Psi_n = Psi_0 + sum x x' + kappa_0 m_0 m_0' - kappa_n m_n m_n'``."""
        result = normal_inverse_wishart_posterior(self.prior, self.data)
        self.assertIsInstance(result, NormalInverseWishartPosterior)
        n = len(self.data)
        kappa_0 = Fraction(self.prior.kappa)
        m_0 = [Fraction(v) for v in self.prior.location]
        rows = [[Fraction(v) for v in row] for row in self.data]
        kappa_n = kappa_0 + n
        m_n = [
            (kappa_0 * m_0[j] + sum(row[j] for row in rows)) / kappa_n for j in range(2)
        ]
        self.assertEqual(result.posterior.kappa, float(kappa_n))
        self.assertEqual(result.posterior.dof, self.prior.dof + n)
        self.assertEqual(result.sample_size, n)
        for j in range(2):
            close(self, result.posterior.location[j], float(m_n[j]), 1e-15)
            for k in range(2):
                exact = (
                    Fraction(self.prior.scale[j][k])
                    + sum(row[j] * row[k] for row in rows)
                    + kappa_0 * m_0[j] * m_0[k]
                    - kappa_n * m_n[j] * m_n[k]
                )
                close(self, result.posterior.scale[j][k], float(exact), 1e-13)

    def test_evidence_obeys_the_chain_rule(self) -> None:
        """``p(x_1)`` is the prior predictive t; ``p(D) = prod p(x_i | x_<i)``."""
        single = normal_inverse_wishart_posterior(self.prior, self.data[:1])
        close(
            self,
            single.log_marginal_likelihood,
            self.prior.predictive.log_pdf(self.data[0]),
            1e-13,
        )
        total = 0.0
        current = self.prior
        for row in self.data:
            total += current.predictive.log_pdf(row)
            current = normal_inverse_wishart_posterior(current, (row,)).posterior
        batch = normal_inverse_wishart_posterior(self.prior, self.data)
        close(self, batch.log_marginal_likelihood, total, 1e-12)
        for left, right in zip(
            current.scale, batch.posterior.scale, strict=True
        ):
            for a, b in zip(left, right, strict=True):
                close(self, a, b, 1e-13)
        self.assertEqual(batch.predictive, batch.posterior.predictive)

    def test_one_dimension_predictive_is_student_t(self) -> None:
        """``p = 1``: ``t_nu(m, sqrt(psi (kappa + 1) / (kappa nu)))``."""
        prior = NormalInverseWishart((0.3,), 1.5, ((2.4,),), 6.0)
        reference = StudentT(6.0, 0.3, math.sqrt(2.4 * 2.5 / (1.5 * 6.0)))
        for x in (-2.0, 0.3, 1.7):
            close(self, prior.predictive.log_pdf((x,)), reference.log_pdf(x), 1e-13)
        mean_law = prior.mean_marginal.marginal(0)
        close(self, mean_law.sigma, math.sqrt(2.4 / (1.5 * 6.0)), 1e-15)

    def test_joint_density_and_marginals_by_sampling(self) -> None:
        """``log p(mu, Sigma)`` decomposes; sampled moments match the marginals."""
        prior = NormalInverseWishart((0.5, -1.0), 2.0, ((1.5, 0.3), (0.3, 0.8)), 9.0)
        sigma = ((0.4, 0.05), (0.05, 0.2))
        mu = (0.4, -0.9)
        expected = MultivariateNormal(
            prior.location, tuple(tuple(v / 2.0 for v in r) for r in sigma)
        ).log_pdf(mu) + InverseWishart(prior.scale, 9.0).log_pdf(sigma)
        close(self, prior.log_pdf(mu, sigma), expected, 1e-14)
        self.assertEqual(prior.log_pdf(mu, ((1.0, 2.0), (2.0, 1.0))), -math.inf)
        size = 3000
        pairs = prior.sample(size, seed=2026)
        self.assertEqual(pairs[:3], prior.sample(3, seed=2026))
        mean_law = prior.mean_marginal
        for j in range(2):
            mean = math.fsum(p[0][j] for p in pairs) / size
            error = math.sqrt(mean_law.covariance[j][j] / size)
            self.assertLess(abs(mean - prior.location[j]), 4.0 * error)
        covariance_mean = prior.covariance_marginal.mean
        sampled = math.fsum(p[1][0][0] for p in pairs) / size
        # Sigma_11 ~ InvGamma((nu-p+1)/2, psi_11/2): variance 2 psi_11^2 /
        # ((nu-p-1)^2 (nu-p-3)).
        variance = 2.0 * 1.5**2 / (6.0**2 * 4.0)
        self.assertLess(
            abs(sampled - covariance_mean[0][0]), 4.0 * math.sqrt(variance / size)
        )

    def test_invalid_priors_and_data(self) -> None:
        """Bad hyperparameters and observation shapes raise."""
        scale = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            NormalInverseWishart((0.0, 0.0), 0.0, scale, 3.0)
        with self.assertRaises(ValueError):
            NormalInverseWishart((0.0, 0.0), 1.0, scale, 1.0)
        with self.assertRaises(ValueError):
            NormalInverseWishart((0.0,), 1.0, scale, 3.0)
        with self.assertRaises(ValueError):
            normal_inverse_wishart_posterior(self.prior, ())
        with self.assertRaises(ValueError):
            normal_inverse_wishart_posterior(self.prior, ((1.0, 2.0, 3.0),))
        with self.assertRaises(TypeError):
            normal_inverse_wishart_posterior(None, self.data)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
