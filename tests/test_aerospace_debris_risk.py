"""Tests for MMOD penetration-risk algebra, uncertainty and design trade.

Order: closed-form Poisson/PNP checks on a hand-checkable single surface,
angle/speed discretisation against integrals, ballistic-limit regime
continuity, Jensen gap, Sobol ranking against a closed form, reliability
methods against an exact lognormal probability, thickness design against a
closed form, and the Gamma-Poisson update against grid integration.

Every number is synthetic and illustrative. The flux and the ballistic-limit
coefficients are invented for the tests; none is ORDEM/MASTER data or a
published equation.
"""

from __future__ import annotations

import math
import os
import pickle
import unittest

from probstat.aerospace import debris_risk as dr
from probstat.distributions import Gamma, LogNormal, Normal
from probstat.reliability import ModelLimitState, form_hlrf, subset_simulation
from probstat.sensitivity import sobol_analysis

RHO_P = 2700.0
RHO_T = 2800.0

# Hand-checkable limit: d_c = 0.5 t, independent of everything else.
HALF_THICKNESS = dr.PowerLawBallisticLimit(0.5, 1.0, 0.0, 0.0, 0.0, 0.0)
# Synthetic flux F(>d) = 1e-3 (d / 1 mm)^-2 per m^2 per year.
FLUX_Q2 = dr.PowerLawFlux(1.0e-3, 1.0e-3, 2.0)
ENV_1 = dr.ImpactEnvironment.single_bin(0.0, 1.0e4, RHO_P)


def plate(thickness: float, area: float = 10.0, flux=FLUX_Q2) -> dr.Surface:
    """Return a one-wall surface with the half-thickness limit."""
    return dr.Surface(
        "plate", area, dr.SingleWallShield(HALF_THICKNESS, thickness, RHO_T), flux
    )


def phi(x: float) -> float:
    """Return the standard normal CDF."""
    return Normal(0.0, 1.0).cdf(x)


class HandCaseTests(unittest.TestCase):
    """A single surface, one bin: every number is checkable by hand."""

    def test_single_surface_matches_hand_calculation(self) -> None:
        """d_c = 2 mm, F = 1e-3 / 4 = 2.5e-4, N = 10 * 5 * 2.5e-4 = 0.0125."""
        risk = dr.penetration_risk([plate(4.0e-3)], ENV_1, 5.0)
        self.assertAlmostEqual(risk.expected_penetrations, 0.0125, places=15)
        self.assertAlmostEqual(risk.pnp, math.exp(-0.0125), places=15)
        self.assertAlmostEqual(
            risk.penetration_probability, -math.expm1(-0.0125), places=15
        )
        self.assertEqual(risk.names, ("plate",))
        self.assertEqual(risk.bins, 1)

    def test_surfaces_add_and_pnp_multiplies(self) -> None:
        """N adds over surfaces, so the PNPs multiply (independent thinning)."""
        a, b = plate(4.0e-3, 10.0), plate(6.0e-3, 3.0)
        total = dr.penetration_risk([a, b], ENV_1, 5.0)
        ra = dr.penetration_risk([a], ENV_1, 5.0)
        rb = dr.penetration_risk([b], ENV_1, 5.0)
        self.assertAlmostEqual(
            total.expected_penetrations,
            ra.expected_penetrations + rb.expected_penetrations,
            places=15,
        )
        self.assertAlmostEqual(total.pnp, ra.pnp * rb.pnp, places=15)
        # N scales with area and time, and with d_c^-q for a power-law flux.
        self.assertAlmostEqual(
            rb.expected_penetrations, 3.0 * 5.0 * 1.0e-3 / 9.0, places=15
        )
        again = dr.penetration_risk([a], ENV_1, 10.0)
        self.assertAlmostEqual(
            again.expected_penetrations, 2.0 * ra.expected_penetrations, places=15
        )

    def test_epistemic_factors_scale_n_exactly(self) -> None:
        """N(f, g) = f g^-q N for a power-law flux with exponent q = 2."""
        base = dr.penetration_risk([plate(4.0e-3)], ENV_1, 5.0)
        moved = dr.penetration_risk(
            [plate(4.0e-3)], ENV_1, 5.0, flux_factor=3.0, limit_factor=1.5
        )
        self.assertAlmostEqual(
            moved.expected_penetrations,
            base.expected_penetrations * 3.0 / 1.5**2,
            places=15,
        )

    def test_poisson_counts_reproduce_pnp(self) -> None:
        """Simulated counts have mean N and P(0) = exp(-N) within MC bands."""
        n, size = 0.7, 20000
        counts = dr.simulate_penetration_counts(n, size, seed=11)
        mean = sum(counts) / size
        self.assertLess(abs(mean - n), 4.0 * math.sqrt(n / size))
        zero = sum(1 for c in counts if c == 0) / size
        p0 = math.exp(-n)
        self.assertLess(abs(zero - p0), 4.0 * math.sqrt(p0 * (1.0 - p0) / size))
        self.assertEqual(counts, dr.simulate_penetration_counts(n, size, seed=11))

    def test_tabulated_flux_is_exact_for_a_power_law(self) -> None:
        """Log-log interpolation of power-law nodes reproduces the law."""
        diam = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
        table = dr.TabulatedFlux(diam, tuple(FLUX_Q2.cumulative(d) for d in diam))
        for d in (2.0e-4, 1.0e-3, 3.3e-3, 0.05):
            self.assertAlmostEqual(
                table.cumulative(d) / FLUX_Q2.cumulative(d), 1.0, places=12
            )
        with self.assertRaises(ValueError):
            table.cumulative(5.0e-5)
        with self.assertRaises(ValueError):
            table.cumulative(0.5)
        extended = dr.TabulatedFlux(diam, table.fluxes, extrapolate=True)
        self.assertAlmostEqual(
            extended.cumulative(0.5) / FLUX_Q2.cumulative(0.5), 1.0, places=12
        )
        risk_t = dr.penetration_risk([plate(4.0e-3, flux=table)], ENV_1, 5.0)
        self.assertAlmostEqual(risk_t.expected_penetrations, 0.0125, places=12)

    def test_truncated_flux_has_no_larger_particles(self) -> None:
        """A maximum diameter removes the large-particle risk entirely."""
        cut = dr.PowerLawFlux(1.0e-3, 1.0e-3, 2.0, maximum_diameter=1.0e-3)
        self.assertEqual(cut.cumulative(2.0e-3), 0.0)
        risk = dr.penetration_risk([plate(4.0e-3, flux=cut)], ENV_1, 5.0)
        self.assertEqual(risk.pnp, 1.0)


class DiscretisationTests(unittest.TestCase):
    """Angle and speed bins against the integrals they approximate."""

    def test_isotropic_bins_recover_mean_cosine(self) -> None:
        """E[cos theta] = 2/3 for the density 2 sin cos; midpoint rule converges."""
        errors = []
        for n in (10, 100, 1000):
            angles, weights = dr.isotropic_angle_bins(n)
            self.assertAlmostEqual(sum(weights), 1.0, places=14)
            mean_cos = sum(
                w * math.cos(a) for a, w in zip(angles, weights, strict=True)
            )
            errors.append(abs(mean_cos - 2.0 / 3.0))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 1.0e-4)

    def test_angle_integral_of_the_risk_against_closed_form(self) -> None:
        """d_c ~ cos^e: N = N_normal E[cos^(-e q)] = N_normal 2/(2 - e q).

        With e = 0.25 and q = 2 the integrand ``u^-0.5`` is integrable but
        singular at grazing incidence, so the midpoint rule converges only
        like ``count^-0.75``; the band follows that rate.
        """
        limit = dr.PowerLawBallisticLimit(0.5, 1.0, 0.0, 0.0, 0.0, 0.25)
        shield = dr.SingleWallShield(limit, 4.0e-3, RHO_T)
        surface = dr.Surface("oblique", 10.0, shield, FLUX_Q2)
        exact = 0.0125 * 2.0 / (2.0 - 0.5)
        previous = math.inf
        for n in (20, 200, 2000):
            angles, weights = dr.isotropic_angle_bins(n)
            env = dr.ImpactEnvironment.product(angles, weights, (1.0e4,), (1.0,), RHO_P)
            risk = dr.penetration_risk([surface], env, 5.0)
            error = abs(risk.expected_penetrations / exact - 1.0)
            self.assertLess(error, previous)
            previous = error
        self.assertLess(previous, 0.01)

    def test_speed_bins_with_a_speed_power_law(self) -> None:
        """d_c ~ V^-0.5, q = 2: N = N_ref sum_k w_k (V_k / V_ref)^1 exactly."""
        limit = dr.PowerLawBallisticLimit(0.5, 1.0, 0.0, 0.0, -0.5, 0.0)
        shield = dr.SingleWallShield(limit, 4.0e-3, RHO_T)
        surface = dr.Surface("fast", 10.0, shield, FLUX_Q2)
        speeds, weights = (5.0e3, 1.0e4, 2.0e4), (0.2, 0.5, 0.3)
        env = dr.ImpactEnvironment.product((0.0,), (1.0,), speeds, weights, RHO_P)
        risk = dr.penetration_risk([surface], env, 5.0)
        reference = dr.penetration_risk(
            [surface], dr.ImpactEnvironment.single_bin(0.0, 1.0e4, RHO_P), 5.0
        )
        expected = reference.expected_penetrations * sum(
            w * v / 1.0e4 for v, w in zip(speeds, weights, strict=True)
        )
        self.assertAlmostEqual(risk.expected_penetrations / expected, 1.0, places=12)

    def test_zero_weight_bins_are_ignored(self) -> None:
        """A bin with zero weight cannot change N, even at a wild speed."""
        env = dr.ImpactEnvironment((0.0,), (1.0e4, 1.0e9), ((1.0, 0.0),), RHO_P)
        risk = dr.penetration_risk([plate(4.0e-3)], env, 5.0)
        self.assertAlmostEqual(risk.expected_penetrations, 0.0125, places=15)
        self.assertEqual(risk.bins, 1)


class BallisticLimitTests(unittest.TestCase):
    """Power-law family and the three-regime dual-wall structure."""

    def test_power_law_exponents_apply_as_documented(self) -> None:
        """d_c = K t^a rho_p^b rho_t^c V^d cos(theta)^e on a hand value."""
        limit = dr.PowerLawBallisticLimit(2.0, 1.0, -0.5, 0.25, -1.0, 0.5)
        value = limit.critical_diameter(
            thickness=3.0e-3,
            projectile_density=4.0,
            target_density=16.0,
            velocity=2.0,
            angle=math.pi / 3.0,
        )
        expected = 2.0 * 3.0e-3 * 4.0**-0.5 * 16.0**0.25 * 2.0**-1.0 * 0.5**0.5
        self.assertAlmostEqual(value / expected, 1.0, places=14)

    def test_demonstration_limits_are_labelled_and_positive(self) -> None:
        """The demonstration families are documented as synthetic."""
        for obj in (
            dr.demonstration_single_wall_limit,
            dr.demonstration_dual_wall_limit,
            dr.demonstration_flux,
        ):
            self.assertIn("SYNTHETIC", obj.__doc__ or "")
        d = dr.demonstration_single_wall_limit().critical_diameter(
            thickness=2.0e-3,
            projectile_density=RHO_P,
            target_density=RHO_T,
            velocity=1.0e4,
            angle=0.0,
        )
        self.assertTrue(0.0 < d < 2.0e-3)

    def dual(self, velocity: float, angle: float = 0.3) -> float:
        """Return the demonstration dual-wall d_c at one speed."""
        return dr.demonstration_dual_wall_limit().critical_diameter(
            bumper_thickness=1.0e-3,
            wall_thickness=3.0e-3,
            spacing=0.1,
            projectile_density=RHO_P,
            wall_density=RHO_T,
            velocity=velocity,
            angle=angle,
        )

    def test_regime_selection_and_continuity(self) -> None:
        """d_c is continuous across both boundaries; transition is linear."""
        limit = dr.demonstration_dual_wall_limit()
        vb, vh = limit.ballistic_limit_velocity, limit.hypervelocity_limit_velocity
        self.assertEqual(limit.regime(vb - 1.0), "ballistic")
        self.assertEqual(limit.regime(vb), "ballistic")
        self.assertEqual(limit.regime(0.5 * (vb + vh)), "transition")
        self.assertEqual(limit.regime(vh), "hypervelocity")
        for boundary in (vb, vh):
            at = self.dual(boundary)
            for eps in (1.0e-6, -1.0e-6):
                self.assertAlmostEqual(self.dual(boundary + eps) / at, 1.0, places=9)
        # Linear interpolation: the midpoint is the mean of the end values.
        self.assertAlmostEqual(
            self.dual(0.5 * (vb + vh)), 0.5 * (self.dual(vb) + self.dual(vh)), places=15
        )
        # Outside the transition the regimes are plain power laws.
        regime_b = limit.ballistic
        direct = regime_b.critical_diameter(
            bumper_thickness=1.0e-3,
            wall_thickness=3.0e-3,
            spacing=0.1,
            projectile_density=RHO_P,
            wall_density=RHO_T,
            velocity=2000.0,
            angle=0.3,
        )
        self.assertEqual(self.dual(2000.0), direct)

    def test_continuity_for_a_discontinuous_pair_of_regimes(self) -> None:
        """Even strongly different regimes join at V_b and V_h (by construction)."""
        low = dr.DualWallRegime(5.0, 0, 1, 0, 0, 0, -1.0, 0)
        high = dr.DualWallRegime(1.0e-3, 0, 1, 0, 0, 0, 1.0, 0)
        limit = dr.DualWallBallisticLimit(low, high, 2000.0, 9000.0)

        def d(v: float) -> float:
            return limit.critical_diameter(
                bumper_thickness=1.0e-3,
                wall_thickness=3.0e-3,
                spacing=0.1,
                projectile_density=RHO_P,
                wall_density=RHO_T,
                velocity=v,
                angle=0.0,
            )

        # d_b(V_b) differs from d_h(V_b) by orders of magnitude, but the
        # interpolation uses d_b(V_b) at V_b and d_h(V_h) at V_h.
        self.assertAlmostEqual(d(2000.0 + 1e-9) / d(2000.0), 1.0, places=8)
        self.assertAlmostEqual(d(9000.0 - 1e-9) / d(9000.0), 1.0, places=8)
        self.assertNotAlmostEqual(
            low.critical_diameter(
                bumper_thickness=1e-3,
                wall_thickness=3e-3,
                spacing=0.1,
                projectile_density=RHO_P,
                wall_density=RHO_T,
                velocity=2000.0,
                angle=0.0,
            )
            / high.critical_diameter(
                bumper_thickness=1e-3,
                wall_thickness=3e-3,
                spacing=0.1,
                projectile_density=RHO_P,
                wall_density=RHO_T,
                velocity=2000.0,
                angle=0.0,
            ),
            1.0,
            places=2,
        )

    def test_dual_wall_surface_risk_uses_the_regimes_per_bin(self) -> None:
        """Risk over speed bins in all three regimes equals the bin-by-bin sum."""
        shield = dr.DualWallShield(
            dr.demonstration_dual_wall_limit(), 1.0e-3, 3.0e-3, 0.1, RHO_T
        )
        surface = dr.Surface("dual", 5.0, shield, dr.demonstration_flux())
        speeds, weights = (2.0e3, 5.0e3, 9.0e3), (0.3, 0.3, 0.4)
        env = dr.ImpactEnvironment.product(
            (0.0, 0.5), (0.5, 0.5), speeds, weights, RHO_P
        )
        risk = dr.penetration_risk([surface], env, 4.0)
        flux = dr.demonstration_flux()
        manual = 0.0
        for a in (0.0, 0.5):
            for v, w in zip(speeds, weights, strict=True):
                d = self.dual(v, a)
                manual += 0.5 * w * 5.0 * 4.0 * flux.cumulative(d)
        self.assertAlmostEqual(risk.expected_penetrations / manual, 1.0, places=12)


def single_surface_case() -> tuple[list[dr.Surface], dr.ImpactEnvironment, float]:
    """Return the case with N0 = 0.02 used by the uncertainty tests."""
    # N0 = A T F0 (0.5 t / d0)^-2 ; choose A = 10, T = 5, t = 4 mm -> 0.0125;
    # scale the area by 1.6 to reach N0 = 0.02.
    return [plate(4.0e-3, area=16.0)], ENV_1, 5.0


class UncertaintyTests(unittest.TestCase):
    """Nested epistemic propagation, Jensen gap, Sobol and reliability methods."""

    def test_mean_pnp_matches_taylor_and_gap_is_positive(self) -> None:
        """E[exp(-N0 f)] expanded in moments of f; Jensen gap > 0.

        Fourth-order series; the first neglected term is
        N0^5 E[f^5] / 120 = 1e-15 exp(12.5 sigma^2) / 120 ~ 2.5e-14.
        """
        n0 = 1.0e-3
        surface = plate(4.0e-3, area=0.8)  # N0 = 0.001 at f = 1
        factors = dr.EpistemicFactors(0.8)
        q = dr.mean_pnp_quadrature([surface], ENV_1, 5.0, factors, order=40)
        s2 = factors.flux_sigma**2
        m1, m2, m3, m4 = (math.exp(c * s2) for c in (0.5, 2.0, 4.5, 8.0))
        taylor = 1.0 - n0 * m1 + n0**2 * m2 / 2.0 - n0**3 * m3 / 6.0 + n0**4 * m4 / 24.0
        self.assertAlmostEqual(q.mean_pnp, taylor, delta=1.0e-13)
        self.assertGreater(q.jensen_gap, 0.0)
        series_gap = (
            0.5 * n0**2 * (m2 - m1**2)
            - n0**3 * (m3 - m1**3) / 6.0
            + n0**4 * (m4 - m1**4) / 24.0
        )
        self.assertAlmostEqual(q.jensen_gap, series_gap, delta=1.0e-12)
        self.assertAlmostEqual(q.pnp_at_mean_inputs, math.exp(-n0 * m1), places=14)
        self.assertAlmostEqual(q.pnp_at_median_inputs, math.exp(-n0), places=14)

    def test_gamma_flux_closed_form_against_numerical_integral(self) -> None:
        """(b/(b+N0))^a equals the Simpson integral of exp(-N0 f) Gamma(f)."""
        a, b, n0 = 4.0, 4.0, 0.3
        law = Gamma(a, b)
        count, upper = 20000, 25.0
        h = upper / count
        total = 0.0
        for i in range(count + 1):
            f = i * h
            weight = 1.0 if i in (0, count) else (4.0 if i % 2 else 2.0)
            total += weight * math.exp(-n0 * f) * law.pdf(f)
        self.assertAlmostEqual(
            dr.mean_pnp_gamma_flux(n0, a, b), total * h / 3.0, places=9
        )

    def test_nested_distribution_agrees_with_quadrature(self) -> None:
        """Monte Carlo over (f, g) reproduces the deterministic mean within 4 SE."""
        surfaces, env, time = single_surface_case()
        factors = dr.EpistemicFactors(0.5, 0.1)
        dist = dr.pnp_distribution(surfaces, env, time, factors, samples=20000, seed=3)
        quad = dr.mean_pnp_quadrature(surfaces, env, time, factors, order=30)
        self.assertLess(
            abs(dist.mean_pnp - quad.mean_pnp), 4.0 * dist.mean_pnp_standard_error
        )
        # Measured decomposition. E[exp(-N)] >= exp(-E[N]) always (Poisson
        # Jensen term), but N is convex in the limit factor g (N ~ g^-2), so
        # E[N] > N(E f, E g) and the gap against PNP(E[inputs]) is NEGATIVE
        # for this case; the sign is a property of the model, not of exp.
        self.assertGreater(quad.poisson_jensen_gap, 0.0)
        self.assertLess(quad.jensen_gap, 0.0)
        self.assertAlmostEqual(
            quad.jensen_gap,
            quad.poisson_jensen_gap + (quad.pnp_at_mean_n - quad.pnp_at_mean_inputs),
            places=15,
        )
        self.assertLess(
            abs(dist.jensen_gap - quad.jensen_gap),
            4.0 * dist.mean_pnp_standard_error,
        )
        self.assertLess(
            abs(dist.poisson_jensen_gap - quad.poisson_jensen_gap),
            4.0 * dist.mean_pnp_standard_error,
        )
        self.assertAlmostEqual(
            dist.pnp_at_median_inputs, quad.pnp_at_median_inputs, places=14
        )
        # PNP is bounded by 1; the 10 % lower quantile lies below the mean.
        self.assertLessEqual(dist.pnp_sorted[-1], 1.0)
        self.assertLess(dist.pnp_lower, dist.mean_pnp)
        # The 90th percentile of N is the log-normal quantile of N0 f g^-2.
        sigma = math.hypot(0.5, 2.0 * 0.1)
        n_q = 0.02 * math.exp(Normal(0.0, 1.0).quantile(0.9) * sigma)
        self.assertAlmostEqual(
            dist.expected_penetrations_quantile / n_q, 1.0, delta=0.03
        )
        # P(PNP < 0.95) = P(N > -ln 0.95), exact for the log-normal product.
        target = -math.log(0.95)
        exact = 1.0 - phi((math.log(target / 0.02)) / sigma)
        se = math.sqrt(exact * (1.0 - exact) / dist.samples)
        self.assertLess(abs(dist.probability_below_requirement - exact), 4.0 * se)
        repeat = dr.pnp_distribution(
            surfaces, env, time, factors, samples=20000, seed=3
        )
        self.assertEqual(repeat.mean_pnp, dist.mean_pnp)

    def test_jensen_gap_sign_is_measured_for_a_large_effect(self) -> None:
        """E[PNP] > PNP(E[inputs]) by many standard errors when N0 f is large.

        N0 = 0.3 with a log-sd of 1 on the flux: the gap is of order 0.1, far
        above the Monte Carlo error of the sampled mean.
        """
        surface = plate(4.0e-3, area=240.0)  # N0 = 0.3
        factors = dr.EpistemicFactors(1.0)
        dist = dr.pnp_distribution(
            [surface], ENV_1, 5.0, factors, samples=20000, seed=8
        )
        quad = dr.mean_pnp_quadrature([surface], ENV_1, 5.0, factors, order=60)
        self.assertGreater(dist.jensen_gap, 20.0 * dist.mean_pnp_standard_error)
        self.assertGreater(quad.jensen_gap, 0.05)
        self.assertLess(
            abs(dist.mean_pnp - quad.mean_pnp), 4.0 * dist.mean_pnp_standard_error
        )
        # PNP at the mean flux factor understates the expected PNP.
        self.assertLess(dist.pnp_at_mean_inputs, dist.mean_pnp)

    def test_no_uncertainty_collapses_to_the_nominal_case(self) -> None:
        """With sigma = 0 the distribution is a point and the gap is zero."""
        surfaces, env, time = single_surface_case()
        dist = dr.pnp_distribution(
            surfaces, env, time, dr.EpistemicFactors(0.0), samples=50
        )
        nominal = dr.penetration_risk(surfaces, env, time).pnp
        self.assertAlmostEqual(dist.mean_pnp, nominal, places=15)
        self.assertEqual(dist.std_pnp, 0.0)
        self.assertAlmostEqual(dist.jensen_gap, 0.0, places=15)

    def test_model_matches_the_algebra_and_pickles(self) -> None:
        """The ProbabilisticModel reproduces penetration_risk at one point."""
        surfaces, env, time = single_surface_case()
        model = dr.mmod_risk_model(
            surfaces,
            env,
            flux_factor=LogNormal(0.0, 0.5),
            limit_factor=LogNormal(0.0, 0.1),
            exposure_time=time,
        )
        self.assertEqual(model.dimension, 2)
        out = model.evaluate({"flux_factor": 2.0, "limit_factor": 0.9})
        ref = dr.penetration_risk(
            surfaces, env, time, flux_factor=2.0, limit_factor=0.9
        )
        self.assertAlmostEqual(
            out["expected_penetrations"], ref.expected_penetrations, places=15
        )
        self.assertAlmostEqual(out["pnp"], ref.pnp, places=15)
        pickle.loads(pickle.dumps(model.model))

    def test_sobol_ranking_matches_the_lognormal_product_closed_form(self) -> None:
        """N = N0 f g^-2 with lognormal f, g: indices in closed form.

        With ``h = g^-2`` log-normal(0, 2 sigma_g): S_f = Var(f) E[h]^2 / V,
        S_g = E[f]^2 Var(h) / V, total effects ``1 - S_other``.
        """
        sf, sg, q = 0.5, 0.1, 2.0
        surfaces, env, time = single_surface_case()
        model = dr.mmod_risk_model(
            surfaces,
            env,
            flux_factor=LogNormal(0.0, sf),
            limit_factor=LogNormal(0.0, sg),
            exposure_time=time,
        )
        sh = q * sg
        ef, ef2 = math.exp(0.5 * sf**2), math.exp(2.0 * sf**2)
        eh, eh2 = math.exp(0.5 * sh**2), math.exp(2.0 * sh**2)
        var = ef2 * eh2 - (ef * eh) ** 2
        s_f = (ef2 - ef**2) * eh**2 / var
        s_g = ef**2 * (eh2 - eh**2) / var
        result = sobol_analysis(model, "expected_penetrations", 4096, seed=2)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        total = {
            n: i.estimate
            for n, i in zip(result.input_names, result.total_effect, strict=True)
        }
        self.assertGreater(first["flux_factor"], first["limit_factor"])
        self.assertGreater(total["flux_factor"], total["limit_factor"])
        self.assertAlmostEqual(first["flux_factor"], s_f, delta=0.05)
        self.assertAlmostEqual(first["limit_factor"], s_g, delta=0.05)
        self.assertAlmostEqual(total["flux_factor"], 1.0 - s_g, delta=0.05)
        self.assertAlmostEqual(total["limit_factor"], 1.0 - s_f, delta=0.05)

    def test_reliability_methods_agree_on_missing_the_requirement(self) -> None:
        """P(PNP < 0.95): exact vs FORM vs subset simulation vs crude MC.

        ``{PNP < 0.95} = {ln f - 2 ln g > ln(-ln 0.95 / N0)}`` is a half-space
        in the standard-normal space, so FORM is exact up to its tolerance.
        Subset simulation is compared in three combined standard errors.
        """
        sf, sg = 0.5, 0.1
        surfaces, env, time = single_surface_case()
        model = dr.mmod_risk_model(
            surfaces,
            env,
            flux_factor=LogNormal(0.0, sf),
            limit_factor=LogNormal(0.0, sg),
            exposure_time=time,
        )
        sigma = math.hypot(sf, 2.0 * sg)
        exact = 1.0 - phi(math.log(-math.log(0.95) / 0.02) / sigma)
        state = ModelLimitState(model, "pnp", 0.95, "below")
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertAlmostEqual(form.probability, exact, delta=2.0e-3)
        # Importance factors are the variance shares of ln N: sf^2 : 4 sg^2.
        self.assertAlmostEqual(
            form.importance_factors["flux_factor"], sf**2 / sigma**2, delta=0.02
        )
        subset = subset_simulation(state, samples_per_level=400, seed=5)
        self.assertTrue(subset.converged)
        band = 3.0 * subset.coefficient_of_variation_upper * subset.probability
        self.assertLess(abs(subset.probability - exact), max(band, 0.02))

    def test_invalid_uncertainty_inputs(self) -> None:
        """Bad factors, levels and sample counts are rejected."""
        surfaces, env, time = single_surface_case()
        with self.assertRaises(ValueError):
            dr.EpistemicFactors(-0.1)
        with self.assertRaises(ValueError):
            dr.EpistemicFactors(0.1, flux_median=0.0)
        with self.assertRaises(ValueError):
            dr.pnp_distribution(
                surfaces, env, time, dr.EpistemicFactors(0.1), samples=5
            )
        with self.assertRaises(ValueError):
            dr.pnp_distribution(
                surfaces, env, time, dr.EpistemicFactors(0.1), requirement=1.0
            )
        with self.assertRaises(TypeError):
            dr.pnp_distribution(surfaces, env, time, 0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dr.mmod_risk_model(
                surfaces, env, flux_factor=1.0, limit_factor=1.0, exposure_time=5.0
            )
        with self.assertRaises(TypeError):
            dr.mmod_risk_model(
                surfaces,
                env,
                flux_factor=True,
                limit_factor=LogNormal(0.0, 0.1),
                exposure_time=5.0,
            )


class DesignTradeTests(unittest.TestCase):
    """Required thickness against a closed form, nominal and with margin."""

    A, T, R, LEVEL = 10.0, 5.0, 0.99, 0.9
    SF, SG = 0.5, 0.1

    @classmethod
    def build(cls, t: float) -> list[dr.Surface]:
        """Return the surface for design variable ``t``."""
        return [plate(t, area=cls.A)]

    @classmethod
    def t_nominal(cls) -> float:
        """Closed-form thickness: N = A T F0 (0.5 t / d0)^-2 = -ln R."""
        target = -math.log(cls.R)
        return 2.0e-3 * math.sqrt(cls.A * cls.T * 1.0e-3 / target)

    def test_nominal_thickness_matches_closed_form(self) -> None:
        """The monotone root find recovers the analytic thickness."""
        design = dr.required_thickness(
            self.build, ENV_1, self.T, self.R, bracket=(1.0e-4, 1.0e-1)
        )
        self.assertTrue(design.converged)
        self.assertAlmostEqual(design.thickness / self.t_nominal(), 1.0, places=8)
        self.assertAlmostEqual(design.nominal_pnp, self.R, places=8)
        self.assertEqual(design.basis, "nominal")
        # A thicker wall is safer (monotone), a thinner one misses.
        thinner = dr.penetration_risk(self.build(0.9 * design.thickness), ENV_1, self.T)
        self.assertLess(thinner.pnp, self.R)

    def test_epistemic_margin_adds_thickness(self) -> None:
        """Sizing on the 90th percentile of N needs exp(z s / q) more wall.

        N = N(t) f g^-2 is lognormal with log-sd s = hypot(sf, 2 sg); the
        requirement ``N_q <= -ln R`` rescales the thickness by ``M^(1/q)`` with
        ``M = exp(z_0.9 s)``. The empirical quantile of 20000 draws carries
        about 1 % error in ``N``, i.e. 0.5 % in ``t``; the band is 1.5 %.
        """
        factors = dr.EpistemicFactors(self.SF, self.SG)
        design = dr.required_thickness(
            self.build,
            ENV_1,
            self.T,
            self.R,
            bracket=(1.0e-4, 1.0),
            basis="quantile",
            factors=factors,
            level=self.LEVEL,
            samples=20000,
            seed=4,
        )
        s = math.hypot(self.SF, 2.0 * self.SG)
        margin = math.exp(Normal(0.0, 1.0).quantile(self.LEVEL) * s / 2.0)
        self.assertGreater(design.thickness, self.t_nominal())
        self.assertAlmostEqual(
            design.thickness / (self.t_nominal() * margin), 1.0, delta=0.015
        )
        self.assertAlmostEqual(design.achieved_metric, self.R, places=6)
        self.assertGreater(design.nominal_pnp, self.R)

    def test_mean_pnp_basis_is_consistent_with_quadrature(self) -> None:
        """Sizing on the epistemic mean PNP: quadrature at t* returns ~R."""
        factors = dr.EpistemicFactors(self.SF, self.SG)
        design = dr.required_thickness(
            self.build,
            ENV_1,
            self.T,
            self.R,
            bracket=(1.0e-4, 1.0),
            basis="mean_pnp",
            factors=factors,
            samples=20000,
            seed=4,
        )
        quad = dr.mean_pnp_quadrature(
            self.build(design.thickness), ENV_1, self.T, factors, order=30
        )
        self.assertAlmostEqual(quad.mean_pnp, self.R, delta=2.0e-3)
        self.assertTrue(design.converged)

    def test_design_failure_cases(self) -> None:
        """Brackets that do not enclose the root and missing factors fail loudly."""
        with self.assertRaises(ValueError):  # already met at the lower end
            dr.required_thickness(
                self.build, ENV_1, self.T, self.R, bracket=(0.05, 0.1)
            )
        with self.assertRaises(ValueError):  # not met at the upper end
            dr.required_thickness(
                self.build, ENV_1, self.T, self.R, bracket=(1.0e-4, 1.0e-3)
            )
        with self.assertRaises(ValueError):
            dr.required_thickness(self.build, ENV_1, self.T, self.R, bracket=(1.0, 0.5))
        with self.assertRaises(ValueError):
            dr.required_thickness(
                self.build,
                ENV_1,
                self.T,
                self.R,
                bracket=(1e-4, 1e-1),
                basis="quantile",
            )
        with self.assertRaises(ValueError):
            dr.required_thickness(
                self.build,
                ENV_1,
                self.T,
                self.R,
                bracket=(1e-4, 1e-1),
                basis="bogus",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            dr.required_thickness(self.build, ENV_1, self.T, 1.0, bracket=(1e-4, 1e-1))


class BayesianUpdateTests(unittest.TestCase):
    """Gamma-Poisson conjugate update against grid integration."""

    A0, B0 = 3.0, 3.0  # mean 1, cv 1/sqrt(3)
    COUNTS, MU = (2, 0, 5), (0.7, 1.1, 2.3)

    def test_conjugate_update_is_exact(self) -> None:
        """Posterior = Gamma(a0 + sum n, b0 + sum mu); moments by formula."""
        post = dr.update_flux_factor(self.A0, self.B0, self.COUNTS, self.MU)
        self.assertEqual(post.shape, 3.0 + 7)
        self.assertAlmostEqual(post.rate, 3.0 + 4.1, places=14)
        self.assertEqual(post.total_count, 7)
        self.assertAlmostEqual(post.mean, 10.0 / 7.1, places=14)
        self.assertAlmostEqual(post.variance, 10.0 / 7.1**2, places=14)
        a, b = dr.gamma_prior_from_mean_cv(1.0, 1.0 / math.sqrt(3.0))
        self.assertAlmostEqual(a, 3.0, places=12)
        self.assertAlmostEqual(b, 3.0, places=12)
        # Sequential updating equals batch updating.
        step = dr.update_flux_factor(self.A0, self.B0, self.COUNTS[:1], self.MU[:1])
        step = dr.update_flux_factor(
            step.shape, step.rate, self.COUNTS[1:], self.MU[1:]
        )
        self.assertAlmostEqual(step.shape, post.shape, places=14)
        self.assertAlmostEqual(step.rate, post.rate, places=14)

    def grid(self) -> tuple[list[float], list[float]]:
        """Return a grid and normalised unnormalised posterior weights."""
        prior = Gamma(self.A0, self.B0)
        count, upper = 40000, 12.0
        xs = [(i + 0.5) * upper / count for i in range(count)]
        w = []
        for f in xs:
            like = 1.0
            for n, mu in zip(self.COUNTS, self.MU, strict=True):
                like *= math.exp(-f * mu + n * math.log(f * mu) - math.lgamma(n + 1))
            w.append(prior.pdf(f) * like)
        total = sum(w)
        return xs, [v / total for v in w]

    def test_posterior_and_predictive_pnp_match_grid_integration(self) -> None:
        """Brute-force Bayes on a grid reproduces mean, quantile and PNP."""
        post = dr.update_flux_factor(self.A0, self.B0, self.COUNTS, self.MU)
        xs, w = self.grid()
        mean = sum(x * v for x, v in zip(xs, w, strict=True))
        self.assertAlmostEqual(mean, post.mean, places=6)
        n0 = 0.4
        grid_pnp = sum(math.exp(-n0 * x) * v for x, v in zip(xs, w, strict=True))
        self.assertAlmostEqual(post.pnp_mean(n0), grid_pnp, places=6)
        q = post.quantile(0.9)
        cum = sum(v for x, v in zip(xs, w, strict=True) if x <= q)
        self.assertAlmostEqual(cum, 0.9, places=3)
        bound = post.pnp_credible_bound(n0, 0.9)
        self.assertAlmostEqual(bound, math.exp(-n0 * q), places=14)
        below = sum(v for x, v in zip(xs, w, strict=True) if math.exp(-n0 * x) >= bound)
        self.assertAlmostEqual(below, 0.9, places=3)

    def test_predictive_count_is_negative_binomial(self) -> None:
        """The pmf sums to one and has mean mu a / b (grid-free identities)."""
        post = dr.update_flux_factor(self.A0, self.B0, self.COUNTS, self.MU)
        mu = 1.7
        pmf = [post.predictive_count_pmf(k, mu) for k in range(400)]
        self.assertAlmostEqual(sum(pmf), 1.0, places=10)
        mean = sum(k * p for k, p in enumerate(pmf))
        self.assertAlmostEqual(mean, mu * post.mean, places=8)
        # P(K = 0) is the posterior predictive PNP for N0 = mu.
        self.assertAlmostEqual(pmf[0], post.pnp_mean(mu), places=14)

    def test_data_shift_the_posterior_toward_the_observations(self) -> None:
        """Counts above (below) the prediction raise (lower) the flux factor."""
        high = dr.update_flux_factor(self.A0, self.B0, (10,), (2.0,))
        low = dr.update_flux_factor(self.A0, self.B0, (0,), (2.0,))
        self.assertGreater(high.mean, 1.0)
        self.assertLess(low.mean, 1.0)
        self.assertLess(high.pnp_mean(0.5), low.pnp_mean(0.5))

    def test_baseline_count_and_failure_cases(self) -> None:
        """Baseline exposure formula and invalid updating inputs."""
        self.assertAlmostEqual(
            dr.baseline_impact_count(2.0, 5.0, FLUX_Q2, 2.0e-3), 2.5e-3, places=15
        )
        with self.assertRaises(ValueError):
            dr.update_flux_factor(1.0, 1.0, (1, 2), (1.0,))
        with self.assertRaises(ValueError):
            dr.update_flux_factor(1.0, 1.0, (), ())
        with self.assertRaises(ValueError):
            dr.update_flux_factor(1.0, 1.0, (-1,), (1.0,))
        with self.assertRaises(ValueError):
            dr.update_flux_factor(1.0, 1.0, (1,), (0.0,))
        with self.assertRaises(TypeError):
            dr.update_flux_factor(1.0, 1.0, (1.5,), (1.0,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dr.update_flux_factor(0.0, 1.0, (1,), (1.0,))


class FailureCaseTests(unittest.TestCase):
    """Adversarial and invalid input."""

    def test_invalid_geometry_and_laws(self) -> None:
        """Out-of-range angles, weights, shapes and types are rejected."""
        with self.assertRaises(ValueError):
            dr.ImpactEnvironment.single_bin(math.pi / 2.0, 1.0e4, RHO_P)
        with self.assertRaises(ValueError):
            dr.ImpactEnvironment.single_bin(-0.1, 1.0e4, RHO_P)
        with self.assertRaises(ValueError):
            dr.ImpactEnvironment((0.0,), (1.0e4,), ((0.9,),), RHO_P)
        with self.assertRaises(ValueError):
            dr.ImpactEnvironment((0.0, 0.1), (1.0e4,), ((1.0,),), RHO_P)
        with self.assertRaises(ValueError):
            dr.ImpactEnvironment((0.0,), (1.0e4,), ((-0.5,),), RHO_P)
        with self.assertRaises(ValueError):
            dr.PowerLawFlux(0.0, 1.0e-3, 2.0)
        with self.assertRaises(ValueError):
            dr.PowerLawFlux(1.0e-3, 1.0e-3, math.nan)
        with self.assertRaises(TypeError):
            dr.PowerLawFlux(True, 1.0e-3, 2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dr.TabulatedFlux((1.0e-3, 1.0e-2), (1.0e-5, 1.0e-4))  # increasing flux
        with self.assertRaises(ValueError):
            dr.TabulatedFlux((1.0e-2, 1.0e-3), (1.0e-4, 1.0e-5))
        with self.assertRaises(ValueError):
            dr.TabulatedFlux((1.0e-3,), (1.0e-5,))
        with self.assertRaises(ValueError):
            dr.PowerLawBallisticLimit(-1.0, 1, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            dr.DualWallBallisticLimit(
                dr.demonstration_dual_wall_limit().ballistic,
                dr.demonstration_dual_wall_limit().hypervelocity,
                7000.0,
                3000.0,
            )
        with self.assertRaises(ValueError):
            dr.isotropic_angle_bins(0)
        with self.assertRaises(ValueError):
            dr.penetration_risk([], ENV_1, 5.0)
        with self.assertRaises(TypeError):
            dr.penetration_risk([plate(4e-3)], "env", 5.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dr.penetration_risk([plate(4e-3)], ENV_1, 0.0)
        with self.assertRaises(ValueError):
            dr.penetration_risk([plate(4e-3)], ENV_1, 5.0, flux_factor=-1.0)
        with self.assertRaises(ValueError):
            dr.Surface(
                "", 1.0, dr.SingleWallShield(HALF_THICKNESS, 1e-3, RHO_T), FLUX_Q2
            )
        with self.assertRaises(ValueError):
            dr.simulate_penetration_counts(500.0, 10)
        with self.assertRaises(ValueError):
            HALF_THICKNESS.critical_diameter(
                thickness=1e-3,
                projectile_density=RHO_P,
                target_density=RHO_T,
                velocity=1e4,
                angle=2.0,
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowTierTests(unittest.TestCase):
    """Larger samples: tighter agreement of the sampled and exact PNP mean."""

    def test_large_sample_mean_pnp(self) -> None:
        """200k nested draws agree with quadrature within 3 SE."""
        surfaces, env, time = single_surface_case()
        factors = dr.EpistemicFactors(0.8, 0.15)
        dist = dr.pnp_distribution(surfaces, env, time, factors, samples=200000, seed=9)
        quad = dr.mean_pnp_quadrature(surfaces, env, time, factors, order=60)
        self.assertLess(
            abs(dist.mean_pnp - quad.mean_pnp), 3.0 * dist.mean_pnp_standard_error
        )


if __name__ == "__main__":
    unittest.main()
