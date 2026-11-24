"""Gaussian splitting libraries and mixture propagation.

Oracles: exact moments of the standard normal (mean 0, variance 1), a
brute-force Simpson integral of the squared density difference, perturbation
of the optimised weights (the optimum must not be improved), the exact mean
and covariance of a quadratic map from Gaussian moment identities, the single
unscented transform of ``probstat.filtering.unscented_transform``, and the
batch-evaluation contract of ``probstat.foundations.batch`` (three drivers,
one result). Nothing here pins a digit of a stochastic result.

References:
    [REF-DEMARS-BISHOP-JAH-2013], [REF-HUBER-ETAL-2008],
    [REF-VITTALDEV-RUSSELL-2016], [REF-JULIER-UHLMANN-2004],
    [REF-SORENSON-ALSPACH-1971]. Checks [EQ-FILT-GMSPLIT-001],
    [EQ-FILT-GMSPLIT-002], [EQ-FILT-GMPROP-001], [EQ-FILT-GMPROP-002].

"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from probstat.filtering import (
    GaussianMixtureState,
    nonlinearity_index,
    propagate_gaussian_mixture,
    propagate_gaussian_mixture_procedure,
    split_gaussian,
    univariate_splitting,
    unscented_transform,
)
from probstat.filtering.gaussian_mixture import _solve_splitting, _spacing
from probstat.foundations.batch import AskTellSession, run_procedure
from probstat.multivariate import MultivariateNormal

MEAN = (1.0, 0.8)
COVARIANCE = ((2.0, 0.5), (0.5, 1.0))


def _quadratic_map(x: tuple[float, ...]) -> tuple[float, float]:
    """Return ``(x1 + x2^2 / 2, x2)``, a quadratic map with exact moments.

    References:
        [REF-JULIER-UHLMANN-2004]. No equation is evaluated.

    """
    return (x[0] + 0.5 * x[1] ** 2, x[1])


def _exact_quadratic_moments() -> tuple[
    tuple[float, float], tuple[float, float, float]
]:
    """Return exact mean and ``(var1, cov12, var2)`` of the quadratic map.

    From ``Var(x^2) = 2 s^4 + 4 m^2 s^2`` and ``Cov(x1, x2^2) = 2 m2 P12``
    for Gaussian ``x``.

    References:
        [REF-SORENSON-ALSPACH-1971]. Checks [EQ-FILT-GMPROP-001].

    """
    m1, m2 = MEAN
    p11, p12, p22 = COVARIANCE[0][0], COVARIANCE[0][1], COVARIANCE[1][1]
    mean = (m1 + 0.5 * (m2**2 + p22), m2)
    var1 = p11 + 0.25 * (2 * p22**2 + 4 * m2**2 * p22) + 2 * m2 * p12
    cov12 = p12 + m2 * p22
    return mean, (var1, cov12, p22)


def _simpson_l2(split: object) -> float:
    """Return ``int (f_mix - phi)^2`` by Simpson's rule on ``[-12, 12]``.

    References:
        [REF-HUBER-ETAL-2008]. Checks [EQ-FILT-GMSPLIT-001].

    """
    weights = split.weights  # type: ignore[attr-defined]
    offsets = split.offsets  # type: ignore[attr-defined]
    sigma = split.sigma  # type: ignore[attr-defined]

    def gap(x: float) -> float:
        """Return ``f_mix(x) - phi(x)``.

        References:
            [REF-HUBER-ETAL-2008]. Checks [EQ-FILT-GMSPLIT-001].

        """
        mix = sum(
            w
            * math.exp(-0.5 * ((x - o) / sigma) ** 2)
            / (sigma * math.sqrt(2 * math.pi))
            for w, o in zip(weights, offsets, strict=True)
        )
        return mix - math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    count = 24000
    step = 24.0 / count
    total = gap(-12.0) ** 2 + gap(12.0) ** 2
    for i in range(1, count):
        total += (4 if i % 2 else 2) * gap(-12.0 + i * step) ** 2
    return total * step / 3.0


class PublishedLibraryTests(unittest.TestCase):
    """The moment condition and the L2 optimum against published libraries.

    References:
        [REF-ZANETTI-ETAL-PREPRINT], [REF-HUBER-ETAL-2008]. Checks
        [EQ-FILT-GMSPLIT-001].

    """

    def test_published_three_component_library(self) -> None:
        """A published equal-variance library satisfies this spacing relation.

        Table 3 of the directional-splitting preprint lists three equal-variance
        components (weights 0.1616701997, 0.6766596007, 0.1616701997, means
        0, plus or minus 1.0908000117, standard deviation 0.78439476713) found
        by minimising the Kullback-Leibler divergence with the mean and
        variance constraints. The variance condition ``h^2 sum w_i i^2 = 1 -
        sigma^2`` of this library must reproduce its printed offset, and
        because the printed library lies in the family searched here (equal
        sigma, equally spaced, same moment constraints) the L2 optimum at that
        sigma must not be worse in squared L2 distance than the published
        weights.

        References:
            [REF-ZANETTI-ETAL-PREPRINT, Table 3]. Checks
            [EQ-FILT-GMSPLIT-001].

        """
        weights = (0.1616701997, 0.6766596007, 0.1616701997)
        sigma = 0.78439476713
        self.assertAlmostEqual(_spacing(weights, sigma), 1.0908000117, delta=2e-9)
        ours = univariate_splitting(3, sigma=sigma)
        published = SimpleNamespace(
            weights=weights, offsets=(-1.0908000117, 0.0, 1.0908000117), sigma=sigma
        )
        self.assertLess(_simpson_l2(ours), _simpson_l2(published))
        self.assertAlmostEqual(
            sum(
                w * (sigma**2 + o * o)
                for w, o in zip(ours.weights, ours.offsets, strict=True)
            ),
            1.0,
            places=12,
        )

    def test_published_libraries_that_do_not_preserve_the_variance(self) -> None:
        """Two printed libraries have a second moment below one; this one has one.

        The three-component library printed from DeMars, Bishop and Jah (2013)
        in the directional-splitting preprint (Table 2) and the four-component
        library of Huber et al. (2008, Table I) both have mean zero but
        variances of 0.9548 and 0.9273 as printed (computed here from the
        tables), so a single split changes the second moment by 4.5 percent and
        7.3 percent; the library of this module preserves it exactly by
        construction.

        References:
            [REF-ZANETTI-ETAL-PREPRINT, Table 2],
            [REF-HUBER-ETAL-2008, Table I]. Checks [EQ-FILT-GMSPLIT-001].

        """

        def variance(
            weights: tuple[float, ...], means: tuple[float, ...], sigma: float
        ) -> float:
            """Return the variance of an equal-variance Gaussian mixture.

            References:
                [REF-SORENSON-ALSPACH-1971]. Checks [EQ-FILT-GMSPLIT-001].

            """
            mean = sum(w * m for w, m in zip(weights, means, strict=True))
            second = sum(
                w * (sigma**2 + m * m) for w, m in zip(weights, means, strict=True)
            )
            return second - mean * mean

        demars = variance(
            (0.22522462491, 0.5495507501, 0.22522462491),
            (-1.0575154614, 0.0, 1.0575154614),
            0.67156628866,
        )
        huber = variance(
            (0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098),
            (-1.4131205233, -0.44973059608, 0.44973059608, 1.4131205233),
            0.51751260421,
        )
        self.assertAlmostEqual(demars, 0.9548, delta=5e-5)
        self.assertAlmostEqual(huber, 0.9273, delta=5e-5)
        for count, sigma in ((3, 0.67156628866), (5, 0.5)):
            split = univariate_splitting(count, sigma=sigma)
            moment = sum(
                w * (split.sigma**2 + o * o)
                for w, o in zip(split.weights, split.offsets, strict=True)
            )
            self.assertAlmostEqual(moment, 1.0, places=12)


class UnivariateSplittingTests(unittest.TestCase):
    """The splitting library solved by L2 minimisation.

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-HUBER-ETAL-2008]. Checks
        [EQ-FILT-GMSPLIT-001].

    """

    def test_moments_of_the_standard_normal_are_preserved(self) -> None:
        """Weights sum to one and mean 0, variance 1 hold to round-off.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-001].

        """
        for count, sigma in (
            (3, None),
            (5, None),
            (7, None),
            (3, 0.5),
            (9, 0.6),
            (5, 0.8),
        ):
            with self.subTest(count=count, sigma=sigma):
                split = univariate_splitting(count, sigma=sigma)
                self.assertEqual(len(split.weights), count)
                self.assertAlmostEqual(math.fsum(split.weights), 1.0, places=14)
                self.assertTrue(all(w > 0.0 for w in split.weights))
                mean = math.fsum(
                    w * o for w, o in zip(split.weights, split.offsets, strict=True)
                )
                second = math.fsum(
                    w * (o * o + split.sigma**2)
                    for w, o in zip(split.weights, split.offsets, strict=True)
                )
                self.assertAlmostEqual(mean, 0.0, places=14)
                self.assertAlmostEqual(second, 1.0, places=14)
                self.assertEqual(split.weights, split.weights[::-1])
                spacing = split.offsets[count // 2 + 1] - split.offsets[count // 2]
                self.assertAlmostEqual(split.spacing, spacing, places=14)

    def test_equal_spacing_and_symmetry(self) -> None:
        """Component means are ``i h`` for ``i = -K..K``.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-001].

        """
        split = univariate_splitting(7)
        gaps = [b - a for a, b in zip(split.offsets, split.offsets[1:], strict=False)]
        for gap in gaps:
            self.assertAlmostEqual(gap, split.spacing, places=13)
        self.assertAlmostEqual(split.offsets[3], 0.0, places=15)

    def test_squared_l2_error_matches_a_brute_force_integral(self) -> None:
        """The closed-form product-integral error equals a Simpson integral.

        References:
            [REF-HUBER-ETAL-2008]. Checks [EQ-FILT-GMSPLIT-001].

        """
        for count in (3, 5, 7):
            with self.subTest(count=count):
                split = univariate_splitting(count)
                brute = _simpson_l2(split)
                self.assertAlmostEqual(split.squared_l2_error / brute, 1.0, places=6)

    def test_cached_solutions_equal_a_fresh_solve(self) -> None:
        """The stored weights for 3, 5, and 7 components are the optimum.

        References:
            [REF-NOCEDAL-WRIGHT-2006]. Checks [EQ-FILT-GMSPLIT-001].

        """
        for count, sigma in ((3, 0.7), (5, 0.5), (7, 0.4)):
            with self.subTest(count=count):
                cached = univariate_splitting(count)
                fresh = _solve_splitting(count, sigma)
                for a, b in zip(cached.weights, fresh.weights, strict=True):
                    self.assertAlmostEqual(a, b, places=6)
                self.assertAlmostEqual(
                    cached.squared_l2_error, fresh.squared_l2_error, places=9
                )

    def test_optimum_is_not_improved_by_moment_preserving_perturbations(self) -> None:
        """Moving weight between components raises the squared L2 error.

        References:
            [REF-HUBER-ETAL-2008]. Checks [EQ-FILT-GMSPLIT-001].

        """
        from probstat.filtering.gaussian_mixture import _assemble

        for count in (3, 5, 7):
            base = univariate_splitting(count)
            half = count // 2
            for k in range(1, half + 1):
                for delta in (0.01, -0.01):
                    weights = list(base.weights)
                    weights[half] -= 2 * delta
                    weights[half + k] += delta
                    weights[half - k] += delta
                    perturbed = _assemble(tuple(weights), base.sigma)
                    with self.subTest(count=count, k=k, delta=delta):
                        self.assertGreater(
                            perturbed.squared_l2_error, base.squared_l2_error
                        )

    def test_more_components_fit_better_at_a_fixed_width(self) -> None:
        """At sigma 0.5 the error falls from 3 to 5 to 7 components.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-001].

        """
        errors = [
            univariate_splitting(n, sigma=0.5).squared_l2_error for n in (3, 5, 7)
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(univariate_splitting(5).relative_l2_error, 0.05)

    def test_single_component_is_the_normal(self) -> None:
        """One component is ``N(0, 1)`` with no error.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-001].

        """
        split = univariate_splitting(1)
        self.assertEqual(
            (split.weights, split.offsets, split.sigma), ((1.0,), (0.0,), 1.0)
        )
        self.assertEqual(split.spacing, 0.0)

    def test_validation(self) -> None:
        """Invalid sizes and widths raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-GMSPLIT-001].

        """
        for bad in (0, 2, -3):
            with self.assertRaises(ValueError):
                univariate_splitting(bad)
        with self.assertRaises(TypeError):
            univariate_splitting(3.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            univariate_splitting(9)
        with self.assertRaises(ValueError):
            univariate_splitting(3, sigma=1.0)
        with self.assertRaises(ValueError):
            univariate_splitting(3, sigma=0.0)


class SplitGaussianTests(unittest.TestCase):
    """The multivariate split preserves the mean and covariance.

    References:
        [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-002].

    """

    def test_mean_and_covariance_are_preserved_for_every_direction(self) -> None:
        """Moments equal the input to round-off for several directions.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-002].

        """
        mean = (1.0, -2.0, 0.5)
        cov = ((4.0, 1.0, 0.5), (1.0, 3.0, -0.4), (0.5, -0.4, 1.5))
        for direction in (None, (1.0, 0.0, 0.0), (1.0, -2.0, 3.0), (0.0, 0.0, -1.0)):
            for count in (1, 3, 5, 7):
                with self.subTest(direction=direction, count=count):
                    mixture = split_gaussian(
                        mean, cov, direction=direction, components=count
                    )
                    self.assertEqual(mixture.components, count)
                    for a, b in zip(mixture.mean, mean, strict=True):
                        self.assertAlmostEqual(a, b, places=13)
                    for ra, rb in zip(mixture.covariance, cov, strict=True):
                        for a, b in zip(ra, rb, strict=True):
                            self.assertAlmostEqual(a, b, places=13)

    def test_default_direction_is_the_principal_axis(self) -> None:
        """The offsets lie along the eigenvector of the largest eigenvalue.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-002].

        """
        cov = ((5.0, 0.0), (0.0, 1.0))
        mixture = split_gaussian((0.0, 0.0), cov, components=3)
        for mean in mixture.means:
            self.assertAlmostEqual(mean[1], 0.0, places=14)
        self.assertGreater(mixture.means[2][0], 0.0)
        # The component variance along the axis is sigma^2 times lambda.
        self.assertAlmostEqual(
            mixture.covariances[0][0][0],
            univariate_splitting(3).sigma ** 2 * 5.0,
            places=13,
        )
        self.assertAlmostEqual(mixture.covariances[0][1][1], 1.0, places=14)

    def test_whitened_variance_along_a_general_direction_is_sigma_squared(self) -> None:
        """Along ``d`` the component precision is ``q / sigma^2`` with ``q = d'P^-1 d``.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-002].

        """
        direction = (1.0, 2.0)
        split = univariate_splitting(3)
        mixture = split_gaussian(MEAN, COVARIANCE, direction=direction, components=3)
        inv_p = MultivariateNormal((0.0, 0.0), COVARIANCE)
        q = inv_p.mahalanobis_squared(direction)
        child = MultivariateNormal((0.0, 0.0), mixture.covariances[0])
        self.assertAlmostEqual(
            child.mahalanobis_squared(direction) / q, 1.0 / split.sigma**2, places=10
        )

    def test_direction_sign_and_scale_do_not_matter(self) -> None:
        """``d`` and ``-3 d`` give the same mixture up to component order.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMSPLIT-002].

        """
        a = split_gaussian(MEAN, COVARIANCE, direction=(1.0, 1.0))
        b = split_gaussian(MEAN, COVARIANCE, direction=(-3.0, -3.0))
        self.assertEqual(a.covariances[0], b.covariances[0])
        for pa, pb in zip(a.means, reversed(b.means), strict=True):
            for x, y in zip(pa, pb, strict=True):
                self.assertAlmostEqual(x, y, places=12)

    def test_mixture_density_approximates_the_gaussian(self) -> None:
        """The split density is within a few percent of the Gaussian density.

        References:
            [REF-HUBER-ETAL-2008]. Checks [EQ-FILT-GMSPLIT-002].

        """
        law = MultivariateNormal(MEAN, COVARIANCE)
        mixture = split_gaussian(MEAN, COVARIANCE, components=5)
        peak = law.pdf(MEAN)
        worst = 0.0
        for dx in (-2.0, -1.0, 0.0, 1.0, 2.5):
            for dy in (-1.5, 0.0, 1.0, 2.0):
                point = (MEAN[0] + dx, MEAN[1] + dy)
                worst = max(worst, abs(mixture.pdf(point) - law.pdf(point)))
        self.assertLess(worst / peak, 0.05)

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-GMSPLIT-002].

        """
        with self.assertRaises(ValueError):
            split_gaussian((), ())
        with self.assertRaises(ValueError):
            split_gaussian(MEAN, ((1.0, 0.0), (0.0, 0.0)))
        with self.assertRaises(ValueError):
            split_gaussian(MEAN, COVARIANCE, direction=(0.0, 0.0))
        with self.assertRaises(ValueError):
            split_gaussian(MEAN, COVARIANCE, direction=(1.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            split_gaussian(MEAN, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        with self.assertRaises(TypeError):
            split_gaussian((1.0, "a"), COVARIANCE)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            split_gaussian(MEAN, COVARIANCE, components=4)


class GaussianMixtureStateTests(unittest.TestCase):
    """The mixture record: moments, density, sampling, validation.

    References:
        [REF-SORENSON-ALSPACH-1971], [REF-BISHOP-2006, Section 9.2]. Checks
        [EQ-FILT-GMPROP-001].

    """

    def test_moments_of_a_two_component_mixture(self) -> None:
        """Mean and covariance follow the total-variance decomposition.

        References:
            [REF-SORENSON-ALSPACH-1971]. Checks [EQ-FILT-GMPROP-001].

        """
        mixture = GaussianMixtureState(
            (0.25, 0.75), ((0.0,), (4.0,)), (((1.0,),), ((2.0,),))
        )
        self.assertAlmostEqual(mixture.mean[0], 3.0)
        # E[C] = 0.25 + 1.5 = 1.75; Var(m) = 0.25*9 + 0.75*1 = 3.0
        self.assertAlmostEqual(mixture.covariance[0][0], 4.75)
        self.assertEqual((mixture.components, mixture.dimension), (2, 1))

    def test_density_and_sampling(self) -> None:
        """One component equals the Gaussian density; samples are seeded.

        References:
            [REF-BISHOP-2006, Section 9.2]. Checks [EQ-FILT-GMPROP-001].

        """
        one = GaussianMixtureState((1.0,), (MEAN,), (COVARIANCE,))
        law = MultivariateNormal(MEAN, COVARIANCE)
        self.assertAlmostEqual(one.pdf((0.3, 1.2)), law.pdf((0.3, 1.2)), places=14)
        self.assertAlmostEqual(
            one.log_pdf((0.3, 1.2)), law.log_pdf((0.3, 1.2)), places=12
        )
        mixture = split_gaussian(MEAN, COVARIANCE, components=3)
        draws = mixture.sample(20000, seed=11)
        again = mixture.sample(20000, seed=11)
        self.assertEqual(draws, again)
        for j in range(2):
            average = math.fsum(d[j] for d in draws) / len(draws)
            error = math.sqrt(COVARIANCE[j][j] / len(draws))
            self.assertLess(abs(average - MEAN[j]), 5.0 * error)

    def test_validation(self) -> None:
        """Malformed weights, shapes and non-finite entries are rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-GMPROP-001].

        """
        eye = ((1.0,),)
        with self.assertRaises(ValueError):
            GaussianMixtureState((0.5, 0.4), ((0.0,), (1.0,)), (eye, eye))
        with self.assertRaises(ValueError):
            GaussianMixtureState((1.5, -0.5), ((0.0,), (1.0,)), (eye, eye))
        with self.assertRaises(ValueError):
            GaussianMixtureState((1.0,), ((0.0,), (1.0,)), (eye,))
        with self.assertRaises(ValueError):
            GaussianMixtureState((1.0,), ((0.0, 1.0),), (eye,))
        with self.assertRaises(ValueError):
            GaussianMixtureState((math.nan,), ((0.0,),), (eye,))
        singular = GaussianMixtureState((1.0,), ((0.0,),), (((0.0,),),))
        with self.assertRaises(ValueError):
            singular.pdf((0.0,))


class PropagationTests(unittest.TestCase):
    """Mixture propagation through linear and nonlinear maps.

    References:
        [REF-DEMARS-BISHOP-JAH-2013], [REF-JULIER-UHLMANN-2004]. Checks
        [EQ-FILT-GMPROP-001], [EQ-FILT-GMPROP-002].

    """

    def test_linear_map_reproduces_the_exact_gaussian_moments(self) -> None:
        """For ``y = A x + b`` the output mean and covariance are exact.

        References:
            [REF-JULIER-UHLMANN-2004]. Checks [EQ-FILT-GMPROP-001].

        """
        a = ((2.0, -1.0), (0.5, 3.0), (1.0, 1.0))
        b = (1.0, -2.0, 0.0)

        def affine(x: tuple[float, ...]) -> tuple[float, ...]:
            """Return ``A x + b``.

            References:
                [REF-JULIER-UHLMANN-2004]. No equation is evaluated.

            """
            return tuple(
                sum(r * v for r, v in zip(row, x, strict=True)) + c
                for row, c in zip(a, b, strict=True)
            )

        for count in (1, 3, 5):
            with self.subTest(count=count):
                result = propagate_gaussian_mixture(
                    affine, MEAN, COVARIANCE, components=count
                )
                expected_mean = affine(MEAN)
                for got, want in zip(result.mean, expected_mean, strict=True):
                    self.assertAlmostEqual(got, want, places=12)
                for i in range(3):
                    for j in range(3):
                        want = sum(
                            a[i][k] * COVARIANCE[k][m] * a[j][m]
                            for k in range(2)
                            for m in range(2)
                        )
                        self.assertAlmostEqual(result.covariance[i][j], want, places=11)
                self.assertEqual(result.mixture.components, count)
                self.assertLess(max(result.nonlinearity), 1e-12)

    def test_single_component_equals_the_unscented_transform(self) -> None:
        """With one component the output is the unscented transform.

        References:
            [REF-JULIER-UHLMANN-2004]. Checks [EQ-FILT-GMPROP-001].

        """
        result = propagate_gaussian_mixture(
            _quadratic_map, MEAN, COVARIANCE, components=1
        )
        direct = unscented_transform(_quadratic_map, MEAN, COVARIANCE)
        self.assertEqual(result.mean, direct.mean)
        self.assertEqual(result.covariance, direct.covariance)

    def test_quadratic_map_mixture_beats_a_single_unscented_transform(self) -> None:
        """Mixture mean is exact; covariance is closer than the single transform.

        References:
            [REF-DEMARS-BISHOP-JAH-2013], [REF-JULIER-UHLMANN-2004]. Checks
            [EQ-FILT-GMPROP-001].

        """
        exact_mean, (var1, cov12, var2) = _exact_quadratic_moments()
        single = unscented_transform(_quadratic_map, MEAN, COVARIANCE)
        single_error = abs(single.covariance[0][0] - var1)
        self.assertGreater(single_error, 0.1)
        for count in (3, 5, 7):
            with self.subTest(count=count):
                result = propagate_gaussian_mixture(
                    _quadratic_map,
                    MEAN,
                    COVARIANCE,
                    components=count,
                    direction=(0.0, 1.0),
                )
                for got, want in zip(result.mean, exact_mean, strict=True):
                    self.assertAlmostEqual(got, want, places=12)
                self.assertAlmostEqual(result.covariance[0][1], cov12, places=11)
                self.assertAlmostEqual(result.covariance[1][1], var2, places=11)
                self.assertLess(abs(result.covariance[0][0] - var1), 0.5 * single_error)

    def test_recursive_splitting_refines_by_the_nonlinearity_index(self) -> None:
        """Refinement adds components, respects the cap, and improves the variance.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMPROP-001],
            [EQ-FILT-GMPROP-002].

        """
        _, (var1, _, _) = _exact_quadratic_moments()
        plain = propagate_gaussian_mixture(
            _quadratic_map, MEAN, COVARIANCE, components=1
        )
        refined = propagate_gaussian_mixture(
            _quadratic_map,
            MEAN,
            COVARIANCE,
            components=1,
            max_depth=3,
            nonlinearity_threshold=0.05,
        )
        self.assertGreater(refined.mixture.components, 1)
        self.assertEqual(refined.levels, 4)
        self.assertLessEqual(max(refined.depths), 3)
        self.assertAlmostEqual(math.fsum(refined.mixture.weights), 1.0, places=13)
        self.assertLess(
            abs(refined.covariance[0][0] - var1), abs(plain.covariance[0][0] - var1)
        )
        capped = propagate_gaussian_mixture(
            _quadratic_map,
            MEAN,
            COVARIANCE,
            components=1,
            max_depth=5,
            nonlinearity_threshold=0.0,
            max_components=10,
        )
        self.assertLessEqual(capped.mixture.components, 10)
        affine = propagate_gaussian_mixture(
            lambda x: (2.0 * x[0], x[1] + x[0]),
            MEAN,
            COVARIANCE,
            components=1,
            max_depth=3,
            nonlinearity_threshold=1e-6,
        )
        self.assertEqual((affine.mixture.components, affine.levels), (1, 1))

    def test_nonlinearity_index_is_zero_for_affine_maps_and_scale_free(self) -> None:
        """The index vanishes for affine maps and ignores the output scale.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMPROP-002].

        """
        self.assertLess(
            nonlinearity_index(lambda x: (3 * x[0] - x[1], x[1] + 2), MEAN, COVARIANCE),
            1e-12,
        )
        base = nonlinearity_index(_quadratic_map, MEAN, COVARIANCE)
        scaled = nonlinearity_index(
            lambda x: tuple(7.0 * v for v in _quadratic_map(x)), MEAN, COVARIANCE
        )
        self.assertGreater(base, 0.1)
        self.assertAlmostEqual(base, scaled, places=12)
        self.assertEqual(nonlinearity_index(lambda x: (1.0,), MEAN, COVARIANCE), 0.0)

    def test_three_drivers_give_one_result(self) -> None:
        """Pointwise wrapper, batch model, chunked batch and ask-tell agree.

        References:
            [REF-DEMARS-BISHOP-JAH-2013]. Checks [EQ-FILT-GMPROP-001].

        """
        options = {
            "components": 3,
            "max_depth": 2,
            "nonlinearity_threshold": 0.05,
        }
        reference = propagate_gaussian_mixture(
            _quadratic_map, MEAN, COVARIANCE, **options
        )

        def batch(points: object) -> list[tuple[float, float]]:
            """Evaluate the map on a sequence of points at once.

            References:
                [REF-JULIER-UHLMANN-2004]. No equation is evaluated.

            """
            return [_quadratic_map(p) for p in points]  # type: ignore[attr-defined]

        sizes: list[int] = []

        def recording(points: object) -> list[tuple[float, float]]:
            """Record the batch size and evaluate.

            References:
                [REF-JULIER-UHLMANN-2004]. No equation is evaluated.

            """
            sizes.append(len(points))  # type: ignore[arg-type]
            return batch(points)

        via_batch = propagate_gaussian_mixture(
            _quadratic_map, MEAN, COVARIANCE, batch_model=recording, **options
        )
        self.assertEqual(via_batch, reference)
        self.assertEqual(sizes[0], 3 * 5)
        self.assertEqual(len(sizes), reference.levels)
        self.assertEqual(sum(sizes), reference.evaluations)
        chunked = propagate_gaussian_mixture(
            _quadratic_map,
            MEAN,
            COVARIANCE,
            batch_model=batch,
            max_batch_size=4,
            **options,
        )
        self.assertEqual(chunked, reference)
        session = AskTellSession(
            propagate_gaussian_mixture_procedure(MEAN, COVARIANCE, **options)
        )
        while (pending := session.ask()) is not None:
            session.tell(batch(pending.points))
        self.assertEqual(session.result, reference)
        direct = run_procedure(
            propagate_gaussian_mixture_procedure(MEAN, COVARIANCE, **options), batch
        )
        self.assertEqual(direct, reference)

    def test_validation(self) -> None:
        """Invalid options raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-GMPROP-001].

        """
        with self.assertRaises(ValueError):
            propagate_gaussian_mixture(_quadratic_map, MEAN, COVARIANCE, max_depth=-1)
        with self.assertRaises(TypeError):
            propagate_gaussian_mixture(_quadratic_map, MEAN, COVARIANCE, max_depth=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            propagate_gaussian_mixture(
                _quadratic_map, MEAN, COVARIANCE, nonlinearity_threshold=-1.0
            )
        with self.assertRaises(ValueError):
            propagate_gaussian_mixture(
                _quadratic_map, MEAN, COVARIANCE, components=5, max_components=3
            )
        with self.assertRaises(ValueError):
            propagate_gaussian_mixture(
                lambda x: (1.0,), MEAN, ((1.0, 0.0), (0.0, -1.0))
            )
        with self.assertRaises(ValueError):
            propagate_gaussian_mixture(
                _quadratic_map, MEAN, COVARIANCE, batch_model=lambda pts: [(1.0, 1.0)]
            )


if __name__ == "__main__":
    unittest.main()
