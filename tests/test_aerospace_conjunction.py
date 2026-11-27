"""Tests for short-term-encounter conjunction assessment.

Closed forms first (isotropic covariance: Rayleigh/Rice laws), then agreement
of independent methods, scale (dilution) analysis, sensitivities, the
log-normal scale law, crude Monte Carlo against the planar integral, covariance
realism, and adversarial input. Synthetic cases only.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.aerospace import conjunction as cj
from probstat.multivariate import isotropic_gaussian_ball_probability

DIAG = ((1.0e4, 0.0, 0.0), (0.0, 4.0e4, 0.0), (0.0, 0.0, 9.0e4))  # m^2
SMALL = ((25.0, 0.0, 0.0), (0.0, 25.0, 0.0), (0.0, 0.0, 25.0))


def isotropic(sigma: float, miss: float, radius: float) -> cj.EncounterGeometry:
    """Return an isotropic planar geometry."""
    return cj.plane_encounter((miss, 0.0), ((sigma**2, 0.0), (0.0, sigma**2)), radius)


def rotation_z(angle: float) -> tuple[tuple[float, float, float], ...]:
    """Return a rotation about the third axis."""
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rotation_x(angle: float) -> tuple[tuple[float, float, float], ...]:
    """Return a rotation about the first axis."""
    c, s = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def matmul(a: object, b: object) -> tuple[tuple[float, ...], ...]:
    """Return the matrix product."""
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0])))  # type: ignore[index]
        for i in range(len(a))  # type: ignore[arg-type]
    )


def apply(m: object, v: tuple[float, ...]) -> tuple[float, ...]:
    """Return ``M v``."""
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[index]


def transpose(m: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Return the transpose."""
    return tuple(tuple(m[j][i] for j in range(len(m))) for i in range(len(m[0])))


class GaussLegendreTest(unittest.TestCase):
    """The node generator is exact for polynomials."""

    def test_polynomial_exactness(self) -> None:
        """An n-point rule integrates degree 2n-1 exactly."""
        for order in (1, 2, 5, 16, 33):
            nodes, weights = cj._gauss_legendre(order)
            self.assertAlmostEqual(sum(weights), 2.0, places=13)
            for degree in range(0, min(2 * order, 12)):
                exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
                value = sum(w * x**degree for x, w in zip(nodes, weights, strict=True))
                self.assertAlmostEqual(value, exact, places=12)


class GeometryTest(unittest.TestCase):
    """Encounter-plane frame, projection and invariances."""

    def test_axis_aligned_case_by_hand(self) -> None:
        """Hand-computed plane covariance and miss for a 3-4-5 offset."""
        g = cj.encounter_plane_geometry(
            (300.0, 400.0, 0.0), (0.0, 0.0, 7000.0), DIAG, SMALL, 2.0, 3.0
        )
        self.assertAlmostEqual(g.miss_distance, 500.0, places=10)
        self.assertEqual(g.miss_vector[1], 0.0)
        self.assertAlmostEqual(g.hard_body_radius, 5.0)
        self.assertAlmostEqual(g.relative_speed or 0.0, 7000.0)
        assert g.along_track_offset is not None
        self.assertAlmostEqual(g.along_track_offset, 0.0, places=12)
        # e1 = (0.6, 0.8, 0), e2 = w x e1 = (-0.8, 0.6, 0); C = diag(1e4, 4e4, 9e4)
        # + 25 I.
        c11 = 0.36 * 10025.0 + 0.64 * 40025.0
        c22 = 0.64 * 10025.0 + 0.36 * 40025.0
        c12 = -0.6 * 0.8 * 10025.0 + 0.8 * 0.6 * 40025.0
        self.assertAlmostEqual(g.covariance[0][0], c11, places=8)
        self.assertAlmostEqual(g.covariance[1][1], c22, places=8)
        self.assertAlmostEqual(g.covariance[0][1], c12, places=8)
        self.assertAlmostEqual(g.covariance[1][0], c12, places=8)

    def test_frame_is_orthonormal_right_handed(self) -> None:
        """``(e1, e2, w)`` is orthonormal with ``e1 x e2 = w``."""
        g = cj.encounter_plane_geometry(
            (120.0, -80.0, 15.0), (3000.0, -2000.0, 5000.0), DIAG, SMALL, 1.0, 1.0
        )
        assert g.basis is not None
        e1, e2, w = g.basis
        for a in (e1, e2, w):
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in a)), 1.0, places=13)
        self.assertAlmostEqual(
            sum(x * y for x, y in zip(e1, e2, strict=True)), 0.0, places=13
        )
        self.assertAlmostEqual(
            sum(x * y for x, y in zip(e1, w, strict=True)), 0.0, places=13
        )
        cross = cj._cross(e1, e2)
        for got, want in zip(cross, w, strict=True):
            self.assertAlmostEqual(got, want, places=13)

    def test_along_track_component_removed(self) -> None:
        """A state before TCA gives the same plane data plus the offset."""
        at_tca = cj.encounter_plane_geometry(
            (300.0, 400.0, 0.0), (0.0, 0.0, 7000.0), DIAG, SMALL, 2.0, 3.0
        )
        early = cj.encounter_plane_geometry(
            (300.0, 400.0, -700.0), (0.0, 0.0, 7000.0), DIAG, SMALL, 2.0, 3.0
        )
        self.assertAlmostEqual(early.miss_distance, at_tca.miss_distance, places=10)
        self.assertAlmostEqual(early.along_track_offset or 0.0, -700.0, places=10)
        self.assertAlmostEqual(early.time_to_closest_approach or 0.0, 0.1, places=12)
        self.assertEqual(early.covariance, at_tca.covariance)

    def test_rotation_invariance_of_probability(self) -> None:
        """Rotating the whole scene leaves the miss distance and ``Pc`` alone."""
        r = (300.0, 400.0, 20.0)
        v = (0.0, 0.0, 7000.0)
        base = cj.encounter_plane_geometry(r, v, DIAG, SMALL, 2.0, 3.0)
        q = matmul(rotation_z(0.7), rotation_x(1.1))
        qt = transpose(q)
        rotated = cj.encounter_plane_geometry(
            apply(q, r),
            apply(q, v),
            matmul(matmul(q, DIAG), qt),
            matmul(matmul(q, SMALL), qt),
            2.0,
            3.0,
        )
        self.assertAlmostEqual(rotated.miss_distance, base.miss_distance, places=9)
        p0 = cj.collision_probability(base).probability
        p1 = cj.collision_probability(rotated).probability
        self.assertAlmostEqual(p1, p0, delta=1.0e-9 * max(p0, 1.0e-12) + 1e-14)

    def test_zero_miss_picks_a_valid_frame(self) -> None:
        """A zero miss vector still produces an orthonormal frame."""
        g = cj.encounter_plane_geometry(
            (0.0, 0.0, 0.0), (0.0, 7000.0, 0.0), DIAG, SMALL, 1.0, 1.0
        )
        self.assertEqual(g.miss_distance, 0.0)
        assert g.basis is not None
        self.assertAlmostEqual(sum(x * x for x in g.basis[0]), 1.0, places=13)
        self.assertAlmostEqual(
            sum(x * y for x, y in zip(g.basis[0], g.basis[2], strict=True)), 0.0
        )


class ProbabilityClosedFormTest(unittest.TestCase):
    """Isotropic covariance gives the Rayleigh and Rice laws."""

    def test_zero_miss_isotropic_closed_form(self) -> None:
        """``Pc = 1 - exp(-R^2 / (2 sigma^2))`` for every method."""
        for sigma, radius in ((100.0, 10.0), (100.0, 100.0), (50.0, 140.0)):
            g = isotropic(sigma, 0.0, radius)
            expected = -math.expm1(-(radius**2) / (2.0 * sigma**2))
            for method in ("exact", "foster", "chan"):
                result = cj.collision_probability(g, method=method)
                self.assertTrue(result.converged, (method, sigma, radius))
                self.assertAlmostEqual(
                    result.probability, expected, delta=2.0e-10 * expected + 1e-15
                )

    def test_small_object_matches_leading_term(self) -> None:
        """For ``R << sigma`` the small-object value tracks ``Pc`` to its error."""
        g = isotropic(200.0, 150.0, 2.0)
        exact = cj.collision_probability(g, method="exact").probability
        small = cj.collision_probability(g, method="small-object")
        self.assertLess(
            abs(small.probability - exact), 5.0 * small.error_estimate + 1e-15
        )
        self.assertLess(abs(small.probability - exact) / exact, 1.0e-4)
        # Closed form of the leading term.
        density = math.exp(-0.5 * (150.0 / 200.0) ** 2) / (2.0 * math.pi * 200.0**2)
        self.assertAlmostEqual(small.probability, math.pi * 4.0 * density, delta=1e-15)

    def test_chan_series_matches_rice_law(self) -> None:
        """Chan's series agrees with the noncentral chi-square form."""
        for miss, sigma, radius in (
            (0.0, 30.0, 20.0),
            (60.0, 30.0, 20.0),
            (400.0, 100.0, 25.0),
            (1500.0, 100.0, 60.0),
        ):
            chan = cj.chan_series_probability(miss, sigma, radius)
            rice = isotropic_gaussian_ball_probability(miss, sigma, radius)
            self.assertAlmostEqual(
                chan.probability,
                rice.probability,
                delta=1.0e-12 * rice.probability + 1.0e-15,
                msg=str((miss, sigma, radius)),
            )

    def test_chan_series_extreme_tail_keeps_relative_accuracy(self) -> None:
        """A 1e-21 probability keeps its relative accuracy (no cancellation)."""
        chan = cj.chan_series_probability(900.0, 100.0, 5.0)
        # Reference: 50-digit mpmath quadrature of the Rice density,
        # int_0^R (r/s^2) exp(-(r^2+d^2)/(2 s^2)) I0(r d/s^2) dr.
        reference = 3.3010866019733359e-21
        self.assertAlmostEqual(chan.probability / reference, 1.0, delta=1.0e-12)
        exact = cj.collision_probability(isotropic(100.0, 900.0, 5.0)).probability
        self.assertAlmostEqual(chan.probability / exact, 1.0, delta=1.0e-10)


class MethodAgreementTest(unittest.TestCase):
    """Independent methods agree on anisotropic, off-centre geometries."""

    def test_exact_versus_foster_anisotropic(self) -> None:
        """Quadrature and polar integration agree to 1e-7 relative."""
        cases = (
            ((120.0, 60.0), ((8.0e3, 3.0e3), (3.0e3, 5.0e3)), 30.0),
            ((0.0, 250.0), ((1.0e5, -2.0e4), (-2.0e4, 4.0e4)), 80.0),
            ((40.0, 10.0), ((900.0, 0.0), (0.0, 400.0)), 25.0),
        )
        for miss, cov, radius in cases:
            g = cj.plane_encounter(miss, cov, radius)
            exact = cj.collision_probability(g, method="exact").probability
            foster = cj.collision_probability(g, method="foster")
            self.assertTrue(foster.converged, miss)
            self.assertAlmostEqual(foster.probability / exact, 1.0, delta=1.0e-7)

    def test_exact_versus_erf_integral_oracle(self) -> None:
        """Alfano's one-dimensional error-function form, written independently."""
        miss, cov, radius = (90.0, -40.0), ((6.0e3, 1.5e3), (1.5e3, 2.5e3)), 45.0
        det = cov[0][0] * cov[1][1] - cov[0][1] ** 2
        # Conditional law of y given x; integrate over x in [-R, R] (midpoint
        # rule with many panels is ample for a smooth integrand).
        n = 4000
        total = 0.0
        for i in range(n):
            x = -radius + (i + 0.5) * 2.0 * radius / n
            h = math.sqrt(radius**2 - x * x)
            sx2 = cov[0][0]
            density_x = math.exp(-0.5 * (x - miss[0]) ** 2 / sx2) / math.sqrt(
                2.0 * math.pi * sx2
            )
            cond_mean = miss[1] + cov[0][1] / sx2 * (x - miss[0])
            cond_var = det / sx2
            s = math.sqrt(2.0 * cond_var)
            chord = 0.5 * (
                math.erf((h - cond_mean) / s) - math.erf((-h - cond_mean) / s)
            )
            total += density_x * chord
        total *= 2.0 * radius / n
        g = cj.plane_encounter(miss, cov, radius)
        exact = cj.collision_probability(g).probability
        # The midpoint rule has a square-root endpoint singularity in h; its
        # error is about 1e-6 relative at n = 4000.
        self.assertAlmostEqual(exact / total, 1.0, delta=5.0e-6)

    def test_small_object_converges_with_radius(self) -> None:
        """The relative gap of the small-object value shrinks like ``R^2``."""
        gaps = []
        for radius in (8.0, 4.0, 2.0):
            g = cj.plane_encounter(
                (100.0, 50.0), ((6.0e3, 1.0e3), (1.0e3, 3.0e3)), radius
            )
            exact = cj.collision_probability(g).probability
            small = cj.collision_probability(g, method="small-object").probability
            gaps.append(abs(small - exact) / exact)
        self.assertLess(gaps[1], gaps[0] / 3.0)
        self.assertLess(gaps[2], gaps[1] / 3.0)

    def test_monotone_in_radius(self) -> None:
        """``Pc`` strictly increases with the hard-body radius."""
        values = [
            cj.collision_probability(
                cj.plane_encounter((300.0, 100.0), ((2.0e4, 0.0), (0.0, 5.0e3)), r)
            ).probability
            for r in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
        ]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:], strict=False)))
        self.assertLessEqual(values[-1], 1.0)


class DilutionTest(unittest.TestCase):
    """Scale factor, maximum and dilution flag."""

    def test_isotropic_small_disc_maximum(self) -> None:
        """The maximum is at ``c sigma^2 = d^2 / 2`` and equals ``R^2/(e d^2)``."""
        d, radius = 1000.0, 5.0
        g = isotropic(100.0, d, radius)
        analysis = cj.dilution_analysis(g)
        assert analysis.maximum_scale is not None
        assert analysis.maximum_probability is not None
        sigma_squared_star = analysis.maximum_scale * 100.0**2
        self.assertAlmostEqual(sigma_squared_star / (d * d / 2.0), 1.0, delta=2.0e-3)
        self.assertAlmostEqual(
            analysis.maximum_probability / (radius**2 / (math.e * d * d)),
            1.0,
            delta=2.0e-3,
        )

    def test_maximum_is_a_maximum_against_chan_scan(self) -> None:
        """No scale on a fine grid beats the reported maximum (Chan oracle)."""
        d, radius, sigma = 800.0, 40.0, 150.0
        g = isotropic(sigma, d, radius)
        analysis = cj.dilution_analysis(g)
        assert analysis.maximum_probability is not None
        best = max(
            cj.chan_series_probability(d, sigma * math.sqrt(c), radius).probability
            for c in (math.exp(-4.0 + 0.01 * k) for k in range(1200))
        )
        # The grid (step 0.01 in ln c) can only undershoot the true maximum.
        self.assertGreaterEqual(analysis.maximum_probability, best * (1.0 - 1.0e-12))
        self.assertLess(analysis.maximum_probability, best * (1.0 + 1.0e-4))

    def test_dilution_flag(self) -> None:
        """The flag is true past the maximum and false before it."""
        d, radius = 1000.0, 5.0
        base = isotropic(100.0, d, radius)
        c_star = cj.dilution_analysis(base).maximum_scale
        assert c_star is not None
        beyond = cj.dilution_analysis(base, scale=4.0 * c_star)
        before = cj.dilution_analysis(base, scale=0.25 * c_star)
        self.assertTrue(beyond.in_dilution_region)
        self.assertFalse(before.in_dilution_region)
        assert beyond.maximum_ratio is not None
        self.assertGreater(beyond.maximum_ratio, 1.0)
        self.assertGreater(before.maximum_ratio or 0.0, 1.0)
        # Past the maximum, Pc falls with the scale.
        curve = cj.dilution_analysis(
            base, scales=(4.0 * c_star, 8.0 * c_star, 16.0 * c_star)
        ).curve
        self.assertTrue(curve[0][1] > curve[1][1] > curve[2][1])

    def test_no_dilution_when_mean_inside_disc(self) -> None:
        """A miss inside the radius has no finite maximizer."""
        analysis = cj.dilution_analysis(isotropic(100.0, 3.0, 5.0))
        self.assertFalse(analysis.dilution_possible)
        self.assertIsNone(analysis.maximum_scale)
        self.assertIsNone(analysis.maximum_probability)
        self.assertFalse(analysis.in_dilution_region)

    def test_invalid_scale(self) -> None:
        """Non-positive scales are rejected."""
        g = isotropic(100.0, 1000.0, 5.0)
        with self.assertRaises(ValueError):
            cj.dilution_analysis(g, scale=0.0)
        with self.assertRaises(ValueError):
            cj.dilution_analysis(g, scales=(1.0, -2.0))


class SensitivityTest(unittest.TestCase):
    """Derivatives of ``Pc`` against closed forms."""

    def test_zero_miss_isotropic_derivatives(self) -> None:
        """Check ``dPc/dR`` and ``dPc/d ln c`` against the Rayleigh law."""
        sigma, radius = 100.0, 120.0
        g = isotropic(sigma, 0.0, radius)
        a = radius**2 / (2.0 * sigma**2)
        s = cj.collision_probability_sensitivity(g)
        self.assertAlmostEqual(s.probability, 1.0 - math.exp(-a), places=10)
        self.assertAlmostEqual(
            s.d_radius, radius / sigma**2 * math.exp(-a), delta=1.0e-7 * s.d_radius
        )
        self.assertAlmostEqual(
            s.d_log_scale, -a * math.exp(-a), delta=1.0e-7 * abs(s.d_log_scale)
        )
        assert s.radius_elasticity is not None
        self.assertAlmostEqual(
            s.radius_elasticity, radius * s.d_radius / s.probability, places=12
        )

    def test_small_object_radius_elasticity_is_two(self) -> None:
        """``Pc ~ R^2`` for a small object."""
        g = isotropic(300.0, 200.0, 1.0)
        s = cj.collision_probability_sensitivity(g)
        assert s.radius_elasticity is not None
        self.assertAlmostEqual(s.radius_elasticity, 2.0, delta=1.0e-3)

    def test_scale_derivative_vanishes_at_the_maximum(self) -> None:
        """At the dilution maximum ``dPc/d ln c = 0``."""
        g = isotropic(100.0, 1000.0, 5.0)
        c_star = cj.dilution_analysis(g).maximum_scale
        assert c_star is not None
        at_max = cj.collision_probability_sensitivity(
            cj.plane_encounter(
                g.miss_vector,
                ((c_star * 1.0e4, 0.0), (0.0, c_star * 1.0e4)),
                g.hard_body_radius,
            )
        )
        self.assertLess(abs(at_max.d_log_scale), 1.0e-5 * at_max.probability)

    def test_invalid_step(self) -> None:
        """The step must lie in (0, 0.1]."""
        g = isotropic(100.0, 10.0, 5.0)
        for bad in (0.0, -1.0, 0.5):
            with self.assertRaises(ValueError):
                cj.collision_probability_sensitivity(g, relative_step=bad)


class ScaleUncertaintyTest(unittest.TestCase):
    """Log-normal covariance scale."""

    def test_small_spread_recovers_nominal(self) -> None:
        """With a tiny ``s`` the mean equals the nominal ``Pc``."""
        g = isotropic(100.0, 300.0, 20.0)
        result = cj.pc_under_scale_uncertainty(g, 1.0e-4)
        self.assertAlmostEqual(
            result.mean_probability,
            result.nominal_probability,
            delta=1e-6 * result.nominal_probability,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.std_probability, 1.0e-3 * result.nominal_probability)

    def test_mean_against_independent_trapezoid(self) -> None:
        """Rayleigh case: compare with a fine trapezoid in the normal score."""
        sigma, radius, s = 100.0, 120.0, 0.7
        g = isotropic(sigma, 0.0, radius)
        a = radius**2 / (2.0 * sigma**2)
        step = 0.005
        total = 0.0
        second = 0.0
        z = -10.0
        while z <= 10.0 + 1e-12:
            density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            value = 1.0 - math.exp(-a * math.exp(-s * z))
            total += density * value
            second += density * value * value
            z += step
        mean, var = total * step, second * step - (total * step) ** 2
        result = cj.pc_under_scale_uncertainty(g, s)
        self.assertAlmostEqual(result.mean_probability, mean, delta=1.0e-7)
        self.assertAlmostEqual(result.std_probability, math.sqrt(var), delta=1.0e-6)
        self.assertTrue(result.converged)

    def test_median_scale_shifts_the_law(self) -> None:
        """Doubling the median scale equals doubling the base covariance."""
        a = isotropic(100.0, 400.0, 30.0)
        b = isotropic(100.0 * math.sqrt(2.0), 400.0, 30.0)
        ra = cj.pc_under_scale_uncertainty(a, 0.4, median_scale=2.0)
        rb = cj.pc_under_scale_uncertainty(b, 0.4)
        self.assertAlmostEqual(ra.mean_probability, rb.mean_probability, delta=1e-9)

    def test_monte_carlo_quantiles(self) -> None:
        """Sampled quantiles are ordered and the sample mean tracks quadrature."""
        g = isotropic(100.0, 600.0, 25.0)
        result = cj.pc_under_scale_uncertainty(g, 0.5, samples=400, seed=11)
        self.assertEqual(result.samples, 400)
        q = result.quantile_probabilities
        self.assertEqual(len(q), 3)
        self.assertTrue(q[0] <= q[1] <= q[2])
        assert (
            result.sample_mean is not None and result.sample_standard_error is not None
        )
        self.assertLess(
            abs(result.sample_mean - result.mean_probability),
            4.0 * result.sample_standard_error,
        )
        again = cj.pc_under_scale_uncertainty(g, 0.5, samples=400, seed=11)
        self.assertEqual(again.quantile_probabilities, q)

    def test_invalid_arguments(self) -> None:
        """Bad parameters are rejected."""
        g = isotropic(100.0, 600.0, 25.0)
        with self.assertRaises(ValueError):
            cj.pc_under_scale_uncertainty(g, 0.0)
        with self.assertRaises(ValueError):
            cj.pc_under_scale_uncertainty(g, 0.3, nodes=4)
        with self.assertRaises(ValueError):
            cj.pc_under_scale_uncertainty(g, 0.3, samples=-1)
        with self.assertRaises(ValueError):
            cj.pc_under_scale_uncertainty(g, 0.3, samples=5, quantile_levels=(1.2,))
        with self.assertRaises(TypeError):
            cj.pc_under_scale_uncertainty(g, 0.3, nodes=8.0)  # type: ignore[arg-type]


class MonteCarloTest(unittest.TestCase):
    """Crude Monte Carlo against the planar integral."""

    def scene(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return a relative state with a ``Pc`` of a few percent."""
        return (150.0, -200.0, 30.0), (2000.0, 3000.0, -6000.0)

    def test_agrees_with_planar_probability(self) -> None:
        """The MC estimate is within four standard errors of the integral."""
        r, v = self.scene()
        c1 = ((9.0e4, 1.0e4, 0.0), (1.0e4, 4.0e4, 5.0e3), (0.0, 5.0e3, 2.0e4))
        c2 = ((4.0e4, 0.0, 0.0), (0.0, 2.5e4, 0.0), (0.0, 0.0, 3.0e4))
        g = cj.encounter_plane_geometry(r, v, c1, c2, 150.0, 150.0)
        exact = cj.collision_probability(g).probability
        self.assertGreater(exact, 0.01)
        mc = cj.monte_carlo_collision_probability(
            r, v, c1, c2, 150.0, 150.0, samples=60_000, seed=2024
        )
        self.assertLess(abs(mc.probability - exact), 4.0 * mc.standard_error)
        self.assertLessEqual(mc.lower, exact)
        self.assertGreaterEqual(mc.upper, exact)
        self.assertEqual(mc.samples, 60_000)
        self.assertEqual(mc.method, "crude-wilson")

    def test_reproducible_and_seed_sensitive(self) -> None:
        """The same seed repeats; another seed changes the hit count."""
        r, v = self.scene()
        args = (r, v, DIAG, SMALL, 100.0, 100.0)
        a = cj.monte_carlo_collision_probability(*args, samples=3000, seed=1)
        b = cj.monte_carlo_collision_probability(*args, samples=3000, seed=1)
        c = cj.monte_carlo_collision_probability(*args, samples=3000, seed=2)
        self.assertEqual(a.hits, b.hits)
        self.assertNotEqual(a.hits, c.hits)

    def test_zero_hits_keeps_informative_interval(self) -> None:
        """No hit gives p = 0 but a positive Wilson upper endpoint."""
        mc = cj.monte_carlo_collision_probability(
            (5000.0, 0.0, 0.0),
            (0.0, 0.0, 7000.0),
            SMALL,
            SMALL,
            1.0,
            1.0,
            samples=2000,
            seed=3,
        )
        self.assertEqual(mc.hits, 0)
        self.assertEqual(mc.probability, 0.0)
        self.assertGreater(mc.upper, 0.0)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_slow_high_precision_agreement(self) -> None:
        """Slow tier: 2e6 samples agree with the integral to three SE."""
        r, v = self.scene()
        g = cj.encounter_plane_geometry(r, v, DIAG, DIAG, 150.0, 150.0)
        exact = cj.collision_probability(g).probability
        mc = cj.monte_carlo_collision_probability(
            r, v, DIAG, DIAG, 150.0, 150.0, samples=2_000_000, seed=5
        )
        self.assertLess(abs(mc.probability - exact), 3.0 * mc.standard_error)


class RealismTest(unittest.TestCase):
    """Covariance realism and its effect on ``Pc``."""

    def residuals(
        self, factor: float, count: int = 80
    ) -> tuple[list[tuple[float, ...]], list[tuple[tuple[float, ...], ...]]]:
        """Return residuals drawn with standard deviations scaled by ``factor``."""
        rng = random.Random(77)
        errors = []
        covs = []
        for _ in range(count):
            errors.append(
                tuple(
                    factor * math.sqrt(DIAG[i][i]) * rng.gauss(0, 1) for i in range(3)
                )
            )
            covs.append(DIAG)
        return errors, covs

    def test_consistent_covariances(self) -> None:
        """Correctly scaled residuals give a consistent verdict, ``s`` near 1."""
        g = isotropic(100.0, 600.0, 25.0)
        errors, covs = self.residuals(1.0)
        res = cj.conjunction_realism(g, errors, covs)
        self.assertEqual(res.realism.verdict, "consistent")
        self.assertAlmostEqual(res.realism.scale_factor, 1.0, delta=0.25)
        lo, hi = res.probability_range
        self.assertLessEqual(lo, res.nominal_probability + 1e-15)
        self.assertGreaterEqual(hi, res.nominal_probability - 1e-15)

    def test_optimistic_covariances_flagged(self) -> None:
        """Residuals twice as large as predicted: scale about 4, optimistic."""
        g = isotropic(100.0, 600.0, 25.0)
        errors, covs = self.residuals(2.0)
        res = cj.conjunction_realism(g, errors, covs)
        self.assertEqual(res.realism.verdict, "optimistic")
        self.assertAlmostEqual(res.realism.scale_factor, 4.0, delta=1.0)
        self.assertAlmostEqual(
            res.scaled_probability,
            cj.collision_probability(
                cj.plane_encounter(
                    g.miss_vector,
                    (
                        (res.realism.scale_factor * 1.0e4, 0.0),
                        (0.0, res.realism.scale_factor * 1.0e4),
                    ),
                    g.hard_body_radius,
                )
            ).probability,
            delta=1.0e-12,
        )

    def test_invalid_inputs(self) -> None:
        """Wrong types and shapes are rejected."""
        g = isotropic(100.0, 600.0, 25.0)
        errors, covs = self.residuals(1.0, count=5)
        with self.assertRaises(TypeError):
            cj.conjunction_realism("geometry", errors, covs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cj.conjunction_realism(g, errors, covs, scan_points=2)
        with self.assertRaises(ValueError):
            cj.conjunction_realism(g, errors, covs[:-1])


class FailureCaseTest(unittest.TestCase):
    """Adversarial and invalid input."""

    def test_zero_relative_velocity(self) -> None:
        """The encounter plane is undefined without relative velocity."""
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry(
                (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), DIAG, SMALL, 1.0, 1.0
            )
        with self.assertRaises(ValueError):
            cj.monte_carlo_collision_probability(
                (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), DIAG, SMALL, 1.0, 1.0, samples=10
            )

    def test_singular_covariance(self) -> None:
        """A zero or rank-deficient projected covariance is rejected."""
        zero = ((0.0,) * 3,) * 3
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry(
                (1.0, 2.0, 3.0), (0.0, 0.0, 7000.0), zero, zero, 1.0, 1.0
            )
        line = ((1.0e4, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry(
                (1.0, 2.0, 3.0), (0.0, 0.0, 7000.0), line, zero, 1.0, 1.0
            )
        with self.assertRaises(ValueError):
            cj.plane_encounter((1.0, 0.0), ((1.0, 1.0), (1.0, 1.0)), 1.0)
        with self.assertRaises(ValueError):
            cj.monte_carlo_collision_probability(
                (1.0, 2.0, 3.0), (0.0, 0.0, 7000.0), line, zero, 1.0, 1.0, samples=10
            )

    def test_invalid_shapes_and_values(self) -> None:
        """Shapes, signs, finiteness and types are validated."""
        ok = ((0.0, 0.0, 7000.0), DIAG, SMALL)
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry((1.0, 2.0), *ok, 1.0, 1.0)
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry((1.0, 2.0, 3.0), *ok, -1.0, 1.0)
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry((1.0, 2.0, 3.0), *ok, 0.0, 0.0)
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry(
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 7000.0),
                ((1.0, 2.0), (2.0, 1.0)),
                SMALL,
                1.0,
                1.0,
            )
        asym = ((1.0e4, 5.0e3, 0.0), (0.0, 1.0e4, 0.0), (0.0, 0.0, 1.0e4))
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry((1.0, 2.0, 3.0), *ok[:1], asym, SMALL, 1.0, 1.0)
        with self.assertRaises(ValueError):
            cj.encounter_plane_geometry((math.nan, 2.0, 3.0), *ok, 1.0, 1.0)
        with self.assertRaises(TypeError):
            cj.encounter_plane_geometry(("a", 2.0, 3.0), *ok, 1.0, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cj.plane_encounter((1.0,), ((1.0, 0.0), (0.0, 1.0)), 1.0)
        with self.assertRaises(ValueError):
            cj.plane_encounter((1.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 0.0)

    def test_method_errors(self) -> None:
        """Unknown methods, non-isotropic Chan, and bad tolerances are rejected."""
        g = cj.plane_encounter((10.0, 0.0), ((100.0, 0.0), (0.0, 400.0)), 3.0)
        with self.assertRaises(ValueError):
            cj.collision_probability(g, method="monte-carlo")
        with self.assertRaises(ValueError):
            cj.collision_probability(g, method="chan")
        with self.assertRaises(TypeError):
            cj.collision_probability((10.0, 0.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cj.foster_probability(
                (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 1.0, tolerance=2.0
            )
        with self.assertRaises(ValueError):
            cj.foster_probability(
                (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 1.0, max_nodes=8
            )
        with self.assertRaises(TypeError):
            cj.foster_probability(
                (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 1.0, max_nodes=64.0
            )  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cj.chan_series_probability(-1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            cj.chan_series_probability(1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            cj.chan_series_probability(1.0e7, 1.0, 1.0)

    def test_foster_reports_non_convergence(self) -> None:
        """A tiny covariance far inside a huge disc is flagged, not hidden."""
        result = cj.foster_probability(
            (0.0, 0.0), ((1.0e-4, 0.0), (0.0, 1.0e-4)), 1.0, max_nodes=32
        )
        self.assertFalse(result.converged)

    def test_monte_carlo_invalid_arguments(self) -> None:
        """Sample counts and confidence are validated."""
        args = ((1.0, 2.0, 3.0), (0.0, 0.0, 7000.0), DIAG, SMALL, 1.0, 1.0)
        with self.assertRaises(ValueError):
            cj.monte_carlo_collision_probability(*args, samples=0)
        with self.assertRaises(TypeError):
            cj.monte_carlo_collision_probability(*args, samples=10.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cj.monte_carlo_collision_probability(*args, samples=10, confidence=1.5)


if __name__ == "__main__":
    unittest.main()
