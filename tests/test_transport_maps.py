"""Tests of triangular transport maps, SVGD posteriors and reference comparison.

Experimental (added in 0.28).

Oracles: the exact Gaussian triangular (Cholesky) map, a finite-difference
triangular determinant, the analytic banana and Gaussian-mixture densities,
the deterministic Gauss-Legendre ``quadrature_reference`` and the closed-form
conditional of a linear Gaussian model. Stochastic checks use fixed seeds and
bands of several Monte Carlo standard errors.

References:
    [EQ-INV-TMAP-001], [EQ-INV-TMAP-002], [EQ-INV-TMAP-003],
    [EQ-INV-TMAP-004], [EQ-INV-TMAP-005], [EQ-INV-TMAP-006],
    [EQ-INV-TMAP-007].

"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse.posterior_sampling import quadrature_reference
from probstat.inverse.transport_maps import (
    TriangularTransportMap,
    compare_samples_to_reference,
    fit_triangular_map_from_density,
    fit_triangular_map_from_samples,
    svgd_posterior,
)
from probstat.linalg import cholesky, solve_lower


def banana_log_density(x):
    """Return the unnormalized banana log density (x2 = x1^2 / 2 + 0.3 e)."""
    return -0.5 * x[0] ** 2 - 0.5 * ((x[1] - 0.5 * x[0] ** 2) / 0.3) ** 2


def banana_gradient(x):
    """Return the gradient of the banana log density."""
    r = (x[1] - 0.5 * x[0] ** 2) / 0.3
    return (-x[0] + r / 0.3 * x[0], -r / 0.3)


def banana_samples(n, seed):
    """Draw exact banana samples."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.gauss(0.0, 1.0)
        out.append((a, 0.5 * a * a + 0.3 * rng.gauss(0.0, 1.0)))
    return out


def bimodal_log_density(x, separation=2.0):
    """Return the log density: mixture in x1, x2 | x1 ~ N(x1 / 2, 0.5^2)."""
    a = math.exp(-0.5 * ((x[0] - separation) / 0.6) ** 2) + math.exp(
        -0.5 * ((x[0] + separation) / 0.6) ** 2
    )
    return math.log(a) - 0.5 * ((x[1] - 0.5 * x[0]) / 0.5) ** 2


_BANANA_REFERENCE = None
_BIMODAL_REFERENCE = None


def banana_reference():
    """Return the cached quadrature reference of the banana density."""
    global _BANANA_REFERENCE
    if _BANANA_REFERENCE is None:
        _BANANA_REFERENCE = quadrature_reference(
            banana_log_density, [(-6.0, 6.0), (-3.0, 20.0)], panels=24, order=8
        )
    return _BANANA_REFERENCE


def bimodal_reference():
    """Return the cached quadrature reference of the bimodal density."""
    global _BIMODAL_REFERENCE
    if _BIMODAL_REFERENCE is None:
        _BIMODAL_REFERENCE = quadrature_reference(
            bimodal_log_density, [(-5.0, 5.0), (-5.0, 5.0)], panels=24, order=8
        )
    return _BIMODAL_REFERENCE


class MonotoneParametrizationTests(unittest.TestCase):
    """Monotonicity by construction, inverse and Jacobian checks."""

    @staticmethod
    def random_map(seed, scale):
        """Return a map with random (possibly adversarial) coefficients."""
        rng = random.Random(seed)
        from probstat.inverse.transport_maps import _exponents

        coefficients = []
        for k in range(2):
            alpha = tuple(scale * rng.gauss(0, 1) for _ in _exponents(k, 2))
            beta = tuple(scale * rng.gauss(0, 1) for _ in _exponents(k + 1, 2))
            coefficients.append((alpha, beta))
        return TriangularTransportMap(
            2,
            (2, 2),
            (2, 2),
            12,
            1e-3,
            tuple(coefficients),
            (0.5, -1.0),
            (2.0, 0.7),
            (0.1, 0.2),
            (1.5, 0.5),
        )

    def test_strictly_increasing_for_adversarial_coefficients(self):
        """The slope floor holds for strongly negative slope coefficients."""
        for seed in range(4):
            tmap = self.random_map(seed, 1.5)
            prefix = (0.3, 0.0)
            previous = None
            for index in range(-20, 21):
                value = tmap.forward((0.3, 0.2 * index))[1]
                if previous is not None:
                    # dS/dp >= min_slope * output_scale / input_scale
                    self.assertGreater(value - previous, 0.9 * 1e-3 * 0.5 / 0.7 * 0.2)
                previous = value
            del prefix

    def test_triangular_structure(self):
        """The first component does not depend on the second input."""
        tmap = self.random_map(1, 0.5)
        a = tmap.forward((0.4, -1.0))[0]
        b = tmap.forward((0.4, 3.0))[0]
        self.assertEqual(a, b)

    def test_inverse_roundtrip(self):
        """inverse(forward(x)) recovers x to root-finder accuracy."""
        tmap = self.random_map(2, 0.4)
        for point in ((0.0, 0.0), (1.2, -2.5), (-3.0, 4.0)):
            recovered = tmap.inverse(tmap.forward(point))
            for r, p in zip(recovered, point, strict=True):
                self.assertAlmostEqual(r, p, delta=1e-9 * (1.0 + abs(p)))

    def test_partial_inverse_keeps_head(self):
        """With a fixed head only the tail is solved (conditional inverse)."""
        tmap = self.random_map(3, 0.4)
        point = (0.8, -0.6)
        out = tmap.forward(point)
        recovered = tmap.inverse(out[1:], head=point[:1])
        self.assertEqual(recovered[0], point[0])
        self.assertAlmostEqual(recovered[1], point[1], delta=1e-9)

    def test_log_det_matches_finite_difference_triangular_determinant(self):
        """log|det| equals the log of the product of the diagonal slopes."""
        tmap = self.random_map(4, 0.4)
        point = (0.7, -0.4)
        total = 0.0
        for k in range(2):
            h = 1e-5
            up = list(point)
            dn = list(point)
            up[k] += h
            dn[k] -= h
            slope = (tmap.forward(up)[k] - tmap.forward(dn)[k]) / (2 * h)
            total += math.log(slope)
        # quadrature of order 12 on a smooth integrand: 1e-8 relative
        self.assertAlmostEqual(tmap.log_det_jacobian(point), total, delta=1e-6)

    def test_invalid_constructions_and_inputs(self):
        """Wrong shapes, signs and lengths are rejected."""
        tmap = self.random_map(5, 0.3)
        with self.assertRaises(ValueError):
            tmap.forward((1.0,))
        with self.assertRaises(ValueError):
            tmap.forward((1.0, float("nan")))
        with self.assertRaises(ValueError):
            tmap.inverse((1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            TriangularTransportMap(
                1,
                (1,),
                (0,),
                8,
                1e-3,
                (((0.0, 0.0), (0.0,)),),
                (0.0,),
                (1.0,),
                (0.0,),
                (1.0,),
            )
        with self.assertRaises(ValueError):
            TriangularTransportMap(
                1,
                (0,),
                (0,),
                8,
                0.0,
                (((0.0,), (0.0,)),),
                (0.0,),
                (1.0,),
                (0.0,),
                (1.0,),
            )


class SampleFitTests(unittest.TestCase):
    """Fit from samples: exact Gaussian map, banana, bimodal, conditionals."""

    def test_gaussian_matches_cholesky_oracle(self):
        """Degree 1 and constant slope reproduce the Cholesky map of the sample."""
        rng = random.Random(1)
        data = []
        for _ in range(500):
            a = rng.gauss(0.0, 1.0)
            data.append((1.0 + 2.0 * a, -1.0 + 0.8 * a + 0.6 * rng.gauss(0.0, 1.0)))
        fit = fit_triangular_map_from_samples(data, degree=1, slope_degree=0)
        self.assertTrue(all(fit.converged), fit.stop_reasons)
        n = len(data)
        mean = [sum(r[i] for r in data) / n for i in range(2)]
        cov = tuple(
            tuple(
                sum((r[i] - mean[i]) * (r[j] - mean[j]) for r in data) / n
                for j in range(2)
            )
            for i in range(2)
        )
        factor = cholesky(cov)
        for r in data[:25]:
            expected = solve_lower(factor, tuple(r[i] - mean[i] for i in range(2)))
            got = fit.transport_map.forward(r)
            for e, g in zip(expected, got, strict=True):
                self.assertAlmostEqual(g, e, delta=2e-6)
        # in-sample log density equals the Gaussian MLE density
        log_det = math.log(factor[0][0]) + math.log(factor[1][1])
        z = solve_lower(factor, tuple(data[0][i] - mean[i] for i in range(2)))
        expected_log = -0.5 * sum(v * v for v in z) - math.log(2 * math.pi) - log_det
        self.assertAlmostEqual(fit.log_density(data[0]), expected_log, delta=2e-6)

    def test_banana_density_and_moments(self):
        """The fit recovers the banana log density up to a constant and moments."""
        fit = fit_triangular_map_from_samples(
            banana_samples(600, 11), degree=(0, 2), slope_degree=0
        )
        self.assertTrue(all(fit.converged), fit.stop_reasons)
        ref = banana_reference()
        errors = []
        for point in ((0.0, 0.5), (1.0, 0.9), (-1.5, 1.0), (0.5, 0.0)):
            exact = banana_log_density(point) - ref.log_normalizer
            errors.append(fit.log_density(point) - exact)
        # sampling error of 600 samples: a few percent of the density
        self.assertLess(max(abs(e) for e in errors), 0.25, errors)
        comparison = compare_samples_to_reference(fit.sample(300, seed=2), ref)
        for value in comparison.mean_error_in_sd:
            self.assertLess(abs(value), 0.3)
        for ratio in comparison.variance_ratio:
            self.assertGreater(ratio, 0.6)
            self.assertLess(ratio, 1.5)

    def test_bimodal_marginal_and_mass_balance(self):
        """A quartic slope represents a two-mode marginal."""
        rng = random.Random(7)
        data = []
        for _ in range(600):
            a = rng.gauss(0.0, 0.6) + (2.0 if rng.random() < 0.5 else -2.0)
            data.append((a, 0.5 * a + 0.5 * rng.gauss(0.0, 1.0)))
        fit = fit_triangular_map_from_samples(
            data, degree=(0, 1), slope_degree=(4, 0), quadrature_order=8
        )
        self.assertTrue(all(fit.converged), fit.stop_reasons)
        draws = fit.sample(300, seed=3)
        positive = sum(1 for d in draws if d[0] > 0.0) / len(draws)
        self.assertAlmostEqual(positive, 0.5, delta=0.12)
        # density dip between the modes is represented
        self.assertGreater(
            fit.log_density((2.0, 1.0)), fit.log_density((0.0, 0.0)) + 1.0
        )
        comparison = compare_samples_to_reference(draws, bimodal_reference())
        for value in comparison.mean_error_in_sd:
            self.assertLess(abs(value), 0.35)
        for ratio in comparison.variance_ratio:
            self.assertGreater(ratio, 0.6)
            self.assertLess(ratio, 1.6)

    def test_conditional_sampling_matches_closed_form(self):
        """Likelihood-free posterior: y = theta + 0.5 e, theta ~ N(0, 1)."""
        rng = random.Random(5)
        joint = []
        for _ in range(800):
            theta = rng.gauss(0.0, 1.0)
            joint.append((theta + 0.5 * rng.gauss(0.0, 1.0), theta))  # (y, theta)
        fit = fit_triangular_map_from_samples(joint, degree=1, slope_degree=0)
        draws = fit.conditional_sample((1.0,), 400, seed=9)
        values = [d[0] for d in draws]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        # closed form: mean y / 1.25 = 0.8, var 0.25 / 1.25 = 0.2
        self.assertAlmostEqual(mean, 0.8, delta=0.08)
        self.assertAlmostEqual(var, 0.2, delta=0.05)

    def test_banana_conditional(self):
        """x2 | x1 has mean x1^2 / 2 and standard deviation 0.3."""
        fit = fit_triangular_map_from_samples(
            banana_samples(600, 12), degree=(0, 2), slope_degree=0
        )
        draws = [d[0] for d in fit.conditional_sample((1.2,), 300, seed=4)]
        mean = sum(draws) / len(draws)
        sd = math.sqrt(sum((v - mean) ** 2 for v in draws) / (len(draws) - 1))
        self.assertAlmostEqual(mean, 0.72, delta=0.07)
        self.assertAlmostEqual(sd, 0.3, delta=0.06)

    def test_nonconvergence_is_reported(self):
        """An iteration limit shows up in the status instead of being hidden."""
        fit = fit_triangular_map_from_samples(
            banana_samples(200, 3), degree=(0, 2), slope_degree=0, max_iterations=1
        )
        self.assertFalse(all(fit.converged))
        self.assertIn("iteration limit", fit.stop_reasons)

    def test_invalid_inputs(self):
        """Bad samples and options raise."""
        good = banana_samples(20, 1)
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples([(0.0, 1.0)])
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples([(0.0, 1.0), (1.0,)])
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples([(0.0, 1.0), (0.0, 2.0)])
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples([(0.0, 1.0), (float("nan"), 2.0)])
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples(good, min_slope=1.5)
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples(good, degree=(1, 1, 1))
        with self.assertRaises(ValueError):
            fit_triangular_map_from_samples(good, degree=9)
        with self.assertRaises(TypeError):
            fit_triangular_map_from_samples(good, degree=1.5)
        fit = fit_triangular_map_from_samples(good, degree=1, slope_degree=0)
        with self.assertRaises(ValueError):
            fit.conditional_sample((0.0, 0.0), 3)
        with self.assertRaises(ValueError):
            fit.sample(0)


class DensityFitTests(unittest.TestCase):
    """Fit from an unnormalized density, with the variance diagnostic."""

    def test_gaussian_exact_map_and_normalizer(self):
        """A linear target gives T = mu + L z, zero diagnostic and exact log Z."""
        mu = (1.0, -2.0)
        s11, s12, s22 = 1.5**2, 1.5 * 0.9, 0.9**2 + 0.5**2
        det = s11 * s22 - s12**2

        def log_target(x):
            a, b = x[0] - mu[0], x[1] - mu[1]
            return -0.5 * (s22 * a * a - 2.0 * s12 * a * b + s11 * b * b) / det

        fit = fit_triangular_map_from_density(
            log_target, 2, center=mu, scale=(1.0, 1.0), degree=1, slope_degree=0
        )
        self.assertTrue(fit.converged, fit.stop_reason)
        t = fit.transport_map
        for got, want in zip(
            t.forward((1.0, 0.0)), (mu[0] + 1.5, mu[1] + 0.9), strict=True
        ):
            self.assertAlmostEqual(got, want, delta=1e-6)
        for got, want in zip(t.forward((0.0, 1.0)), (mu[0], mu[1] + 0.5), strict=True):
            self.assertAlmostEqual(got, want, delta=1e-6)
        self.assertLess(fit.log_weight_variance, 1e-10)
        self.assertAlmostEqual(
            fit.log_normalizer_estimate,
            0.5 * math.log((2.0 * math.pi) ** 2 * det),
            delta=1e-6,
        )
        self.assertGreater(fit.importance_ess_fraction, 0.999999)
        # normalized pushforward density equals the Gaussian density
        point = (0.3, -1.1)
        exact = log_target(point) - 0.5 * math.log((2.0 * math.pi) ** 2 * det)
        self.assertAlmostEqual(fit.log_density(point), exact, delta=1e-6)

    def test_banana_against_quadrature(self):
        """The exact-in-family banana map matches log Z and moments."""
        fit = fit_triangular_map_from_density(
            banana_log_density,
            2,
            center=(0.0, 0.5),
            scale=(1.0, 0.8),
            degree=(0, 2),
            slope_degree=0,
            gradient=banana_gradient,
        )
        self.assertTrue(fit.converged, fit.stop_reason)
        ref = banana_reference()
        self.assertAlmostEqual(
            fit.log_normalizer_estimate, ref.log_normalizer, delta=1e-5
        )
        self.assertLess(fit.log_weight_variance, 1e-8)
        comparison = compare_samples_to_reference(fit.sample(400, seed=1), ref)
        for value in comparison.mean_error_in_sd:
            self.assertLess(abs(value), 0.3)
        for ratio in comparison.variance_ratio:
            self.assertGreater(ratio, 0.7)
            self.assertLess(ratio, 1.4)

    def test_finite_difference_gradient_agrees_with_analytic(self):
        """The default central-difference score gives the same fit."""
        kwargs = dict(
            center=(0.0, 0.5), scale=(1.0, 0.8), degree=(0, 2), slope_degree=0
        )
        exact = fit_triangular_map_from_density(
            banana_log_density, 2, gradient=banana_gradient, **kwargs
        )
        approx = fit_triangular_map_from_density(banana_log_density, 2, **kwargs)
        for a, b in zip(
            exact.transport_map.forward((0.4, 0.7)),
            approx.transport_map.forward((0.4, 0.7)),
            strict=True,
        ):
            self.assertAlmostEqual(a, b, delta=1e-5)

    def test_bimodal_is_approximate_and_flagged(self):
        """Mode-seeking fit of a bimodal target: moments fair, diagnostic honest."""
        ref = bimodal_reference()
        fit = fit_triangular_map_from_density(
            bimodal_log_density,
            2,
            center=(0.0, 0.0),
            scale=(2.2, 0.9),
            degree=(0, 1),
            slope_degree=(2, 0),
            reference_order=14,
        )
        self.assertTrue(fit.converged, fit.stop_reason)
        comparison = compare_samples_to_reference(fit.sample(400, seed=2), ref)
        for value in comparison.mean_error_in_sd:
            self.assertLess(abs(value), 0.3)
        self.assertLess(comparison.covariance_relative_error, 0.5)
        # the family cannot represent the mixture: the diagnostic must say so
        self.assertGreater(fit.log_weight_variance, 0.05)
        self.assertLess(fit.importance_ess_fraction, 0.95)
        # E[w] = log Z - KL <= log Z, up to Monte Carlo error of the mean
        slack = 4.0 * math.sqrt(fit.log_weight_variance / fit.diagnostic_samples)
        self.assertLess(fit.log_normalizer_estimate, ref.log_normalizer + slack)

    def test_monte_carlo_reference_and_invalid_inputs(self):
        """The Monte Carlo reference runs; bad arguments raise."""
        fit = fit_triangular_map_from_density(
            banana_log_density,
            2,
            center=(0.0, 0.5),
            scale=(1.0, 0.8),
            degree=(0, 2),
            slope_degree=0,
            reference="monte-carlo",
            reference_samples=300,
            gradient=banana_gradient,
            seed=4,
        )
        self.assertLess(
            abs(fit.log_normalizer_estimate - banana_reference().log_normalizer), 0.05
        )
        with self.assertRaises(TypeError):
            fit_triangular_map_from_density("not callable", 2)
        with self.assertRaises(ValueError):
            fit_triangular_map_from_density(banana_log_density, 2, reference="grid")
        with self.assertRaises(ValueError):
            fit_triangular_map_from_density(banana_log_density, 2, scale=(1.0, -1.0))
        with self.assertRaises(ValueError):
            fit_triangular_map_from_density(
                lambda x: 0.0, 3, reference="gauss-hermite", reference_order=40
            )
        with self.assertRaises(ValueError):
            fit_triangular_map_from_density(lambda x: float("nan"), 2)


class SVGDAndComparisonTests(unittest.TestCase):
    """SVGD on a whitened posterior and the reference comparison."""

    def test_svgd_banana_mean_and_underdispersion(self):
        """SVGD finds the banana mean; the spread is not trusted to be exact."""
        result = svgd_posterior(
            banana_log_density,
            banana_gradient,
            2,
            particles=40,
            iterations=200,
            step_size=0.05,
            seed=5,
        )
        self.assertEqual(result.bandwidth_rule, "median")
        self.assertGreater(result.bandwidth, 0.0)
        comparison = compare_samples_to_reference(
            result.physical_particles, banana_reference()
        )
        for value in comparison.mean_error_in_sd:
            self.assertLess(abs(value), 0.4)
        # documented property: variance is not overestimated by a few particles
        for ratio in comparison.variance_ratio:
            self.assertGreater(ratio, 0.25)
            self.assertLess(ratio, 1.15)
        self.assertEqual(result.gradient_evaluations > 0, True)

    def test_svgd_fixed_bandwidth_and_physical_map(self):
        """A fixed bandwidth is reported and to_physical is applied."""
        result = svgd_posterior(
            lambda z: -0.5 * (z[0] ** 2),
            lambda z: (-z[0],),
            1,
            particles=10,
            iterations=5,
            step_size=0.1,
            bandwidth=0.7,
            to_physical=lambda z: (2.0 * z[0] + 1.0,),
            seed=1,
        )
        self.assertEqual(result.bandwidth_rule, "fixed")
        self.assertEqual(result.bandwidth, 0.7)
        for z, x in zip(result.particles, result.physical_particles, strict=True):
            self.assertAlmostEqual(x[0], 2.0 * z[0] + 1.0, places=12)

    def test_svgd_invalid_inputs(self):
        """Bad arguments raise."""
        f = lambda z: -0.5 * z[0] ** 2  # noqa: E731
        g = lambda z: (-z[0],)  # noqa: E731
        with self.assertRaises(TypeError):
            svgd_posterior(f, None, 1)
        with self.assertRaises(ValueError):
            svgd_posterior(f, g, 1, particles=1)
        with self.assertRaises(ValueError):
            svgd_posterior(f, g, 1, initial_scale=0.0)
        with self.assertRaises(ValueError):
            svgd_posterior(f, g, 1, step_size=-0.1)

    def test_comparison_exact_samples_and_validation(self):
        """Exact banana samples agree with the reference within noise."""
        comparison = compare_samples_to_reference(
            banana_samples(4000, 21), banana_reference()
        )
        for value in comparison.mean_error_in_sd:
            self.assertLess(abs(value), 0.08)
        for ratio in comparison.variance_ratio:
            self.assertAlmostEqual(ratio, 1.0, delta=0.12)
        with self.assertRaises(TypeError):
            compare_samples_to_reference(banana_samples(5, 1), "reference")
        with self.assertRaises(ValueError):
            compare_samples_to_reference([(0.0,), (1.0,)], banana_reference())


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudies(unittest.TestCase):
    """Larger comparison of the three posterior approximations on the banana."""

    def test_banana_study(self):
        """Print moment errors of SVGD (100 particles) and the density fit."""
        ref = banana_reference()
        svgd = svgd_posterior(
            banana_log_density,
            banana_gradient,
            2,
            particles=100,
            iterations=500,
            step_size=0.05,
            seed=5,
        )
        comparison = compare_samples_to_reference(svgd.physical_particles, ref)
        print("SVGD 100 particles:", comparison)
        density = fit_triangular_map_from_density(
            banana_log_density,
            2,
            center=(0.0, 0.5),
            scale=(1.0, 0.8),
            degree=(0, 2),
            slope_degree=0,
            gradient=banana_gradient,
        )
        print(
            "Density fit:",
            compare_samples_to_reference(density.sample(2000, seed=1), ref),
        )
        self.assertLess(max(abs(v) for v in comparison.mean_error_in_sd), 0.4)


if __name__ == "__main__":
    unittest.main()
