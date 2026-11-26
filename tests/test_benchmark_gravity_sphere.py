"""Tests for the synthetic spherical-harmonic gravity benchmark (roadmap 20C).

Everything here is SYNTHETIC: no planet, mission or tracking data. Order:
closed forms of the fully normalized Legendre functions, orthonormality by
quadrature, the addition theorem, the independent closed-form point-mass
attraction against the harmonic series, the radial derivative against finite
differences of the potential, symmetry (longitude rotation, equatorial
reflection), the attenuation in the singular values, the exact Gaussian
posterior against ``linear_gaussian_posterior``, resolved degrees, then the
failure cases (omission aliasing, polar-cap coverage) and invalid inputs. The
slow tier (``PROBSTAT_SLOW=1``) runs a larger design.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse.benchmark_gravity_sphere import (
    GravitySphereBenchmark,
    attenuation,
    coefficient_labels,
    degree_resolution,
    degree_signal_variance,
    gravity_design,
    gravity_posterior,
    gravity_spectrum,
    gravity_sphere_benchmark,
    kaula_prior_sd,
    legendre_orthonormality_error,
    normalized_legendre_table,
    omission_variance,
    point_mass_coefficients,
    point_mass_radial_gravity,
    potential,
    resolution_sweep,
    resolved_degree,
    sphere_design,
)
from probstat.inverse.linear_gaussian import linear_gaussian_posterior
from probstat.linalg import DiagonalCovariance, ScaledIdentityCovariance


class LegendreTests(unittest.TestCase):
    """Closed forms, orthonormality and the addition theorem."""

    def test_low_degree_closed_forms(self) -> None:
        """Pbar_lm for l <= 3 equals the textbook closed forms."""
        theta = 0.7
        t, s = math.cos(theta), math.sin(theta)
        table = normalized_legendre_table(3, theta)
        expected = {
            (1, 0): math.sqrt(3.0) * t,
            (1, 1): math.sqrt(3.0) * s,
            (2, 0): math.sqrt(5.0) * (3.0 * t * t - 1.0) / 2.0,
            (2, 1): math.sqrt(15.0) * s * t,
            (2, 2): math.sqrt(15.0) / 2.0 * s * s,
            (3, 0): math.sqrt(7.0) * (5.0 * t**3 - 3.0 * t) / 2.0,
            (3, 3): math.sqrt(14.0 / 720.0) * 15.0 * s**3,
        }
        for (ell, m), value in expected.items():
            self.assertAlmostEqual(table[ell][m], value, delta=1e-14, msg=(ell, m))

    def test_orthonormality_by_quadrature(self) -> None:
        """(1/2) int Pbar_lm Pbar_l'm dt = (2 - delta_m0) delta_ll' to 1e-12."""
        self.assertLess(legendre_orthonormality_error(10), 1e-12)

    def test_addition_theorem_and_poles(self) -> None:
        """Sum_m Pbar_lm^2 = 2 l + 1 at any colatitude; pole values."""
        rng = random.Random(3)
        for _ in range(5):
            theta = rng.uniform(0.0, math.pi)
            table = normalized_legendre_table(30, theta)
            for ell in range(31):
                total = math.fsum(v * v for v in table[ell])
                self.assertAlmostEqual(total, 2 * ell + 1, delta=1e-10 * (2 * ell + 1))
        north = normalized_legendre_table(20, 0.0)
        south = normalized_legendre_table(20, math.pi)
        for ell in range(21):
            self.assertAlmostEqual(north[ell][0], math.sqrt(2 * ell + 1), delta=1e-12)
            self.assertAlmostEqual(
                south[ell][0], (-1) ** ell * math.sqrt(2 * ell + 1), delta=1e-12
            )
            for m in range(1, ell + 1):
                self.assertEqual(north[ell][m], 0.0)

    def test_equatorial_reflection(self) -> None:
        """Pbar_lm(pi - theta) = (-1)^(l + m) Pbar_lm(theta)."""
        a = normalized_legendre_table(12, 0.9)
        b = normalized_legendre_table(12, math.pi - 0.9)
        for ell in range(13):
            for m in range(ell + 1):
                self.assertAlmostEqual(
                    b[ell][m], (-1) ** (ell + m) * a[ell][m], delta=1e-12
                )


class OperatorTests(unittest.TestCase):
    """The design matrix against independent analytic objects."""

    def test_point_mass_series_matches_closed_form(self) -> None:
        """The harmonic series sums to the closed-form point-mass attraction.

        Truncation: the tail is O((r_s / r)^(L + 1)) = 0.545^41 ~ 1e-11 times
        a modest factor, so 1e-8 is a safe, justified tolerance.
        """
        source = (0.9, 0.4, 0.6, 1.0)
        thetas, phis = sphere_design(12)
        design = gravity_design(thetas, phis, 1.1, 40, min_degree=0)
        coefficients = point_mass_coefficients(*source[:3], 1.0, 40, min_degree=0)
        for row, theta, phi in zip(design.matrix, thetas, phis, strict=True):
            series = math.fsum(a * c for a, c in zip(row, coefficients, strict=True))
            exact = point_mass_radial_gravity(theta, phi, 1.1, source)
            self.assertAlmostEqual(series, exact, delta=1e-8 * abs(exact))

    def test_radial_derivative_against_finite_difference(self) -> None:
        """-dV/dr from the design equals a central difference of the potential."""
        rng = random.Random(11)
        degree = 5
        n = len(coefficient_labels(degree))
        coefficients = tuple(rng.gauss(0.0, 1.0) for _ in range(n))
        h = 1e-5
        for _ in range(4):
            theta = rng.uniform(0.2, 2.9)
            phi = rng.uniform(0.0, 6.0)
            radius = rng.uniform(1.05, 1.5)
            design = gravity_design((theta,), (phi,), radius, degree)
            analytic = math.fsum(
                a * c for a, c in zip(design.matrix[0], coefficients, strict=True)
            )
            numeric = -(
                potential(theta, phi, radius + h, coefficients, degree)
                - potential(theta, phi, radius - h, coefficients, degree)
            ) / (2.0 * h)
            self.assertAlmostEqual(
                analytic, numeric, delta=1e-7 * (1.0 + abs(analytic))
            )

    def test_longitude_rotation_symmetry(self) -> None:
        """Rotating (C, S) by alpha and the longitude by alpha leaves g unchanged."""
        degree, alpha = 4, 0.83
        labels = coefficient_labels(degree)
        rng = random.Random(5)
        c = {lab: rng.gauss(0.0, 1.0) for lab in labels}
        rotated = []
        for ell, m, kind in labels:
            cc = c[(ell, m, "C")]
            ss = c.get((ell, m, "S"), 0.0)
            if kind == "C":
                rotated.append(cc * math.cos(m * alpha) - ss * math.sin(m * alpha))
            else:
                rotated.append(cc * math.sin(m * alpha) + ss * math.cos(m * alpha))
        theta, phi = 1.1, 2.0
        base = gravity_design((theta,), (phi,), 1.2, degree).matrix[0]
        moved = gravity_design((theta,), (phi + alpha,), 1.2, degree).matrix[0]
        before = math.fsum(a * c[lab] for a, lab in zip(base, labels, strict=True))
        after = math.fsum(a * v for a, v in zip(moved, rotated, strict=True))
        self.assertAlmostEqual(before, after, delta=1e-12)

    def test_zonal_field_is_independent_of_longitude(self) -> None:
        """A field with only m = 0 terms does not depend on phi."""
        degree = 5
        labels = coefficient_labels(degree)
        rng = random.Random(8)
        c = [rng.gauss(0.0, 1.0) if lab[1] == 0 else 0.0 for lab in labels]
        values = []
        for phi in (0.0, 1.3, 4.0):
            row = gravity_design((0.9,), (phi,), 1.1, degree).matrix[0]
            values.append(math.fsum(a * v for a, v in zip(row, c, strict=True)))
        self.assertAlmostEqual(values[0], values[1], delta=1e-12)
        self.assertAlmostEqual(values[0], values[2], delta=1e-12)

    def test_attenuation_in_singular_values(self) -> None:
        """Singular values cluster near sqrt(m) (l + 1) r^-(l + 2) by degree.

        The Fibonacci design is quasi-uniform, so the columns are nearly but
        not exactly orthogonal; the measured clusters stay within 8 % of the
        prediction and the altitude ratio of the clusters follows the
        predicted power law.
        """
        thetas, phis = sphere_design(120)
        spectra = {}
        for radius in (1.1, 1.3):
            design = gravity_design(thetas, phis, radius, 4)
            prior = kaula_prior_sd(design.labels)
            spectra[radius] = gravity_spectrum(design, 1e-5, prior)
            for ell, predicted, low, high in spectra[radius].degree_predictions:
                self.assertLess(abs(low - predicted) / predicted, 0.08, (radius, ell))
                self.assertLess(abs(high - predicted) / predicted, 0.08, (radius, ell))
        for (ell, _, low_a, high_a), (_, _, low_b, high_b) in zip(
            spectra[1.1].degree_predictions,
            spectra[1.3].degree_predictions,
            strict=True,
        ):
            ratio = 0.5 * (low_b + high_b) / (0.5 * (low_a + high_a))
            expected = (attenuation(ell, 1.3)) / attenuation(ell, 1.1)
            self.assertLess(abs(ratio - expected) / expected, 0.1)
        self.assertLess(
            spectra[1.3].singular_values[-1], spectra[1.1].singular_values[-1]
        )

    def test_gram_and_svd_spectra_agree(self) -> None:
        """The faster Gram method reproduces the SVD singular values."""
        thetas, phis = sphere_design(60)
        design = gravity_design(thetas, phis, 1.2, 3)
        prior = kaula_prior_sd(design.labels)
        a = gravity_spectrum(design, 1e-5, prior, method="svd")
        b = gravity_spectrum(design, 1e-5, prior, method="gram")
        for x, y in zip(a.singular_values, b.singular_values, strict=True):
            self.assertAlmostEqual(x, y, delta=1e-7 * a.singular_values[0])


class PosteriorTests(unittest.TestCase):
    """Exact Gaussian posterior, resolution and resolved degrees."""

    bench: GravitySphereBenchmark

    @classmethod
    def setUpClass(cls) -> None:
        """Build one small benchmark (degrees 2..4 solved, truth to 10)."""
        cls.bench = gravity_sphere_benchmark(
            solver_degree=4, truth_degree=10, count=100, sigma=1e-5, seed=21
        )

    def test_posterior_matches_data_space_oracle(self) -> None:
        """Parameter-space posterior equals ``linear_gaussian_posterior``."""
        b = self.bench
        post = gravity_posterior(b.design, b.sigma, b.prior_sd, b.data)
        oracle = linear_gaussian_posterior(
            b.design.matrix,
            b.data,
            noise=ScaledIdentityCovariance(len(b.data), b.sigma**2),
            prior_covariance=DiagonalCovariance([s * s for s in b.prior_sd]),
            method="data_space",
        )
        scale = max(b.prior_sd)
        for x, y in zip(post.mean, oracle.mean, strict=True):
            self.assertAlmostEqual(x, y, delta=1e-9 * scale)
        self.assertAlmostEqual(
            post.log_evidence,
            oracle.log_evidence,
            delta=1e-7 * abs(oracle.log_evidence),
        )
        variances = oracle.marginal_variances()
        for sd, v in zip(post.sd, variances, strict=True):
            self.assertAlmostEqual(sd * sd, v, delta=1e-9 * scale**2)

    def test_resolution_matches_orthogonal_prediction(self) -> None:
        """Exact resolved fractions match tau / (1 + tau) within 0.03."""
        b = self.bench
        post = gravity_posterior(b.design, b.sigma, b.prior_sd)
        table = degree_resolution(b.design, post)
        for row in table:
            self.assertLess(abs(row.resolved_fraction - row.expected_fraction), 0.03)
        fractions = [r.resolved_fraction for r in table]
        self.assertEqual(fractions, sorted(fractions, reverse=True))

    def test_resolved_degree_versus_noise_and_altitude(self) -> None:
        """More noise or more altitude resolves fewer degrees (never more)."""
        b = self.bench
        sweep = resolution_sweep(b.design, b.prior_sd, (1e-8, 1e-6, 1e-5, 1e-4, 1e-3))
        degrees = [r.resolved_degree for r in sweep]
        self.assertEqual(degrees, sorted(degrees, reverse=True))
        self.assertEqual(degrees[0], 4)
        self.assertEqual(degrees[-1], 1)
        thetas, phis = sphere_design(100)
        fractions = []
        for radius in (1.05, 1.2, 1.5):
            design = gravity_design(thetas, phis, radius, 4)
            post = gravity_posterior(design, 1e-5, kaula_prior_sd(design.labels), None)
            fractions.append(degree_resolution(design, post)[-1].resolved_fraction)
        self.assertGreater(fractions[0], fractions[1])
        self.assertGreater(fractions[1], fractions[2])

    def test_kaula_signal_variance_matches_sampling(self) -> None:
        """Degree signal variance = Monte Carlo variance of the degree's signal."""
        thetas, phis = sphere_design(60)
        ell, radius = 4, 1.15
        design = gravity_design(thetas, phis, radius, ell, min_degree=ell)
        sd = kaula_prior_sd(design.labels)
        rng = random.Random(1)
        total, draws = 0.0, 400
        for _ in range(draws):
            c = [rng.gauss(0.0, s) for s in sd]
            signal = [
                math.fsum(a * v for a, v in zip(row, c, strict=True))
                for row in design.matrix
            ]
            total += math.fsum(v * v for v in signal) / len(signal)
        expected = degree_signal_variance(ell, radius)
        # variance of the mean of 400 chi-square(9)-like sums ~ 2/(9 * 400)
        self.assertLess(abs(total / draws - expected) / expected, 0.15)


class BenchmarkTests(unittest.TestCase):
    """No inverse crime, omission error, truths and failure cases."""

    def test_truth_degree_exceeds_solver_degree(self) -> None:
        """Higher-degree truth is flagged; equal degree is the flagged crime."""
        high = gravity_sphere_benchmark(solver_degree=4, truth_degree=9, count=60)
        same = gravity_sphere_benchmark(solver_degree=4, truth_degree=4, count=60)
        self.assertFalse(high.same_discretization)
        self.assertTrue(same.same_discretization)
        self.assertGreater(high.omission_rms, 0.0)
        self.assertLess(same.omission_rms, 1e-12)
        self.assertEqual(len(high.truth_coefficients), high.design.unknowns)
        self.assertIn("SYNTHETIC", high.notice)

    def test_omission_rms_matches_prior_expectation(self) -> None:
        """Mean squared omission signal matches the closed-form degree sum.

        Twenty seeds; the comparison band is four standard errors of the
        empirical mean (the omitted degree 5 dominates, so a single draw
        fluctuates by roughly 40 %).
        """
        values = []
        for seed in range(20):
            bench = gravity_sphere_benchmark(
                solver_degree=4, truth_degree=10, count=80, seed=100 + seed
            )
            values.append(bench.omission_rms**2)
        mean = math.fsum(values) / len(values)
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
        expected = omission_variance(4, 10, 1.1)
        self.assertLess(abs(mean - expected), 4.0 * sd / math.sqrt(len(values)))

    def test_calibrated_without_omission(self) -> None:
        """With no omission the standardized posterior errors have rms near 1."""
        bench = gravity_sphere_benchmark(
            solver_degree=4, truth_degree=4, count=100, sigma=1e-5, seed=31
        )
        post = gravity_posterior(bench.design, bench.sigma, bench.prior_sd, bench.data)
        z = [
            (m - t) / s
            for m, t, s in zip(
                post.mean, bench.truth_coefficients, post.sd, strict=True
            )
        ]
        rms = math.sqrt(math.fsum(v * v for v in z) / len(z))
        self.assertTrue(0.6 < rms < 1.5, rms)

    def test_failure_omission_aliasing_overconfident(self) -> None:
        """FAILURE CASE: unmodelled high degrees alias into the low degrees.

        With a tiny noise level the posterior sd is far below the omission
        error, so the standardized error is much larger than 1: the posterior
        is overconfident. This is a property of a truncated solver, not a bug.
        """
        bench = gravity_sphere_benchmark(
            solver_degree=4, truth_degree=12, count=100, sigma=1e-8, seed=31
        )
        post = gravity_posterior(bench.design, bench.sigma, bench.prior_sd, bench.data)
        z = [
            (m - t) / s
            for m, t, s in zip(
                post.mean, bench.truth_coefficients, post.sd, strict=True
            )
        ]
        rms = math.sqrt(math.fsum(v * v for v in z) / len(z))
        self.assertGreater(bench.omission_rms, 100.0 * bench.sigma)
        self.assertGreater(rms, 5.0, rms)

    def test_failure_polar_cap_coverage(self) -> None:
        """FAILURE CASE: data over a polar cap cannot resolve sectorial terms."""
        thetas, phis = sphere_design(100)
        cap_t, cap_p = sphere_design(100, cap_colatitude=0.6)
        global_design = gravity_design(thetas, phis, 1.1, 5)
        cap_design = gravity_design(cap_t, cap_p, 1.1, 5)
        prior = kaula_prior_sd(global_design.labels)
        full = gravity_posterior(global_design, 1e-5, prior)
        capped = gravity_posterior(cap_design, 1e-5, prior)
        sectorial = [
            j
            for j, (ell, m, _) in enumerate(global_design.labels)
            if ell == 5 and m == 5
        ]
        for j in sectorial:
            self.assertGreater(full.resolution_diagonal[j], 0.4)
            self.assertLess(capped.resolution_diagonal[j], 0.1)
        self.assertLess(
            resolved_degree(degree_resolution(cap_design, capped)),
            resolved_degree(degree_resolution(global_design, full)),
        )

    def test_mascon_truth_is_closed_form(self) -> None:
        """The mascon omission equals the independent tail series of the masses."""
        bench = gravity_sphere_benchmark(
            truth="mascon", solver_degree=4, truth_degree=4, count=60, sigma=1e-9
        )
        thetas, phis = bench.design.colatitudes, bench.design.longitudes
        tail = gravity_design(thetas, phis, 1.1, 40, min_degree=5)
        sources = (
            (0.9, 0.4, 0.8, 4e-6),
            (2.1, 3.5, 0.7, -3e-6),
            (1.3, 5.0, 0.85, 2e-6),
        )
        parts = [
            point_mass_coefficients(s[0], s[1], s[2], s[3], 40, min_degree=5)
            for s in sources
        ]
        coefficients = [math.fsum(c) for c in zip(*parts, strict=True)]
        signal = [
            math.fsum(a * c for a, c in zip(row, coefficients, strict=True))
            for row in tail.matrix
        ]
        rms = math.sqrt(math.fsum(v * v for v in signal) / len(signal))
        self.assertAlmostEqual(bench.omission_rms, rms, delta=2e-3 * rms)
        self.assertFalse(bench.same_discretization)


class ValidationTests(unittest.TestCase):
    """Invalid input is rejected, never repaired."""

    def test_invalid_inputs(self) -> None:
        """TypeError / ValueError for bad arguments."""
        with self.assertRaises(ValueError):
            normalized_legendre_table(101, 1.0)
        with self.assertRaises(ValueError):
            normalized_legendre_table(5, 4.0)
        with self.assertRaises(TypeError):
            normalized_legendre_table(2.5, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gravity_design((1.0,), (1.0,), 0.9, 3)
        with self.assertRaises(ValueError):
            gravity_design((1.0, 1.1), (1.0,), 1.1, 3)
        with self.assertRaises(ValueError):
            kaula_prior_sd(((0, 0, "C"),))
        with self.assertRaises(ValueError):
            kaula_prior_sd(((2, 0, "C"),), kappa=0.0)
        with self.assertRaises(ValueError):
            gravity_sphere_benchmark(truth="nope")
        with self.assertRaises(ValueError):
            gravity_sphere_benchmark(solver_degree=6, truth_degree=4)
        with self.assertRaises(ValueError):
            point_mass_coefficients(1.0, 1.0, 1.2, 1.0, 4)
        with self.assertRaises(ValueError):
            point_mass_radial_gravity(1.0, 1.0, 0.5, (1.0, 1.0, 0.6, 1.0))
        with self.assertRaises(ValueError):
            sphere_design(10, cap_colatitude=0.0)
        design = gravity_design(*sphere_design(20), 1.1, 3)
        with self.assertRaises(ValueError):
            gravity_posterior(design, 0.0, (1.0,) * design.unknowns)
        with self.assertRaises(ValueError):
            gravity_posterior(design, 1.0, (1.0,) * (design.unknowns - 1))
        with self.assertRaises(ValueError):
            gravity_spectrum(design, 1.0, (1.0,) * design.unknowns, method="x")
        with self.assertRaises(ValueError):
            degree_resolution(
                design,
                gravity_posterior(design, 1.0, (1.0,) * design.unknowns),
                threshold=1.0,
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowGravityStudy(unittest.TestCase):
    """Larger design: degrees 2..10, 300 observations, spectrum by Gram."""

    def test_degree_ten_study(self) -> None:
        """Print the measured resolved degree, spectrum and omission numbers."""
        bench = gravity_sphere_benchmark(
            solver_degree=10, truth_degree=20, count=300, sigma=1e-5, seed=7
        )
        post = gravity_posterior(bench.design, bench.sigma, bench.prior_sd, bench.data)
        table = degree_resolution(bench.design, post)
        spectrum = gravity_spectrum(
            bench.design, bench.sigma, bench.prior_sd, method="gram"
        )
        sweep = resolution_sweep(
            bench.design, bench.prior_sd, (1e-7, 1e-6, 1e-5, 3e-5, 1e-4)
        )
        print("\nSLOW gravity degree 10, 300 obs, h = 0.1 R")
        for row in table:
            print(
                f"  l={row.degree:2d} fraction={row.resolved_fraction:.3f} "
                f"expected={row.expected_fraction:.3f} resolved={row.resolved}"
            )
        print(
            f"  whitened modes > 1: {spectrum.resolved_modes} of "
            f"{bench.design.unknowns}; cond(A) = {spectrum.condition_number:.2f}"
        )
        print(
            "  sigma -> resolved degree:", [(r.sigma, r.resolved_degree) for r in sweep]
        )
        print(f"  omission rms {bench.omission_rms:.3e} vs sigma {bench.sigma:.1e}")
        degrees = [r.resolved_degree for r in sweep]
        self.assertEqual(degrees, sorted(degrees, reverse=True))
        for row in table:
            self.assertLess(abs(row.resolved_fraction - row.expected_fraction), 0.05)


if __name__ == "__main__":
    unittest.main()
