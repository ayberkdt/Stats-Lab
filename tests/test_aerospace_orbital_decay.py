"""Tests for orbital lifetime under drag, with uncertainty.

Order: closed-form lifetime in an exponential atmosphere with constant scale
height (rigorous bounds and an independent Simpson quadrature), the synthetic
density and solar-activity relations, monotonicity (lifetime increases with
altitude and with the ballistic coefficient, also pathwise with a solar
cycle), the orbit-averaged eccentric equations against an independent
true-anomaly quadrature, reliability methods (FORM, subset simulation, Monte
Carlo) on the lifetime model, a measured Sobol ranking, the monotone root of
the highest compliant altitude and the effect of a drag sail, and failure
cases.

Every number is synthetic and illustrative. The thermosphere, the activity
cycle and the noise are invented for the tests; none is NRLMSISE, Jacchia,
solar-cycle data, or a statement about a real spacecraft.
"""

from __future__ import annotations

import math
import os
import pickle
import unittest

from probstat.aerospace import orbital_decay as od
from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS
from probstat.distributions import LogNormal, Normal, Uniform
from probstat.sensitivity import sobol_analysis

YEAR = od.SECONDS_PER_YEAR
TH = od.demonstration_thermosphere()
# Single exponential layer, constant scale height 60 km, rho = 1e-9 kg/m^3 at
# 150 km (no activity dependence): the closed-form case.
TH1 = od.SyntheticThermosphere((150.0e3,), (60.0e3,), 1.0e-9)
DEADLINE = 25.0


def simpson_lifetime(h0, ballistic, thermosphere, h_re, activity=1.0, panels=20000):
    """Independent composite Simpson evaluation of the static lifetime integral."""
    ratio = activity / thermosphere.reference_activity
    cuts = [h_re] + [z for z in thermosphere.base_altitudes if h_re < z < h0] + [h0]
    total = 0.0

    def f(h):
        rho = thermosphere._density(h, ratio)
        return ballistic / (math.sqrt(EARTH_MU * (EARTH_RADIUS + h)) * rho)

    for lo, hi in zip(cuts, cuts[1:], strict=False):
        w = (hi - lo) / panels
        s = f(lo) + f(hi)
        s += sum((4.0 if k % 2 else 2.0) * f(lo + k * w) for k in range(1, panels))
        total += s * w / 3.0
    return total / YEAR


def lifetime_model(
    altitude, *, area_to_mass=None, phase=0.3, knots=0, sigma=0.0, full=False
):
    """Return the lifetime model used by the reliability tests.

    ``full`` makes the drag coefficient, attitude, phase and amplitude random.
    """
    kwargs = dict(
        initial_altitude=Normal(altitude, 3000.0),
        area_to_mass=area_to_mass or LogNormal(math.log(0.01), 0.2),
        density_factor=LogNormal(0.0, 0.3),
        altitude_fraction=0.1,
        noise_knots=knots,
        noise_sigma=sigma,
    )
    if full:
        kwargs.update(
            drag_coefficient=LogNormal(math.log(2.2), 0.1),
            attitude_factor=Uniform(0.7, 1.0),
            solar_phase=Uniform(0.0, 1.0),
            solar_amplitude=Uniform(0.2, 0.5),
        )
    else:
        kwargs.update(
            drag_coefficient=2.2,
            attitude_factor=0.85,
            solar_phase=phase,
            solar_amplitude=0.4,
        )
    return od.orbital_lifetime_model(TH, DEADLINE, **kwargs)


class ClosedFormTests(unittest.TestCase):
    """Closed form in the constant-scale-height exponential atmosphere."""

    def test_bounds_bracket_the_integration_and_the_quadrature(self) -> None:
        """T lies between the frozen-sqrt(mu a) closed forms (a rigorous bracket).

        With ``sqrt(mu a)`` frozen the equation integrates exactly to
        ``T = H B/c (1/rho_0 - 1/rho_e)``; the true rate is largest at the
        start, so freezing there bounds T below, at the reentry radius above.
        The relative gap is about ``(h0 - he)/(2 R)``; the integration error
        (step <= 0.05 H, RK4) is about 1e-5 and must be far below it.
        """
        b = 50.0
        h0, h_re = 500.0e3, 200.0e3
        bounds = od.exponential_lifetime_bounds(
            h0, b, 150.0e3, 1.0e-9, 60.0e3, reentry_altitude=h_re
        )
        gap = (bounds.upper_years - bounds.lower_years) / bounds.midpoint_years
        expected_gap = 0.5 * (h0 - h_re) / (EARTH_RADIUS + 0.5 * (h0 + h_re))
        self.assertAlmostEqual(gap, expected_gap, delta=0.02 * expected_gap)
        result = od.orbital_decay(h0, b, TH1, None, reentry_altitude=h_re)
        self.assertTrue(result.reentered)
        self.assertEqual(result.stop_reason, "reentry")
        self.assertGreater(result.lifetime_years, bounds.lower_years)
        self.assertLess(result.lifetime_years, bounds.upper_years)
        quad = od.static_lifetime(h0, b, TH1, reentry_altitude=h_re)
        self.assertGreater(quad, bounds.lower_years)
        self.assertLess(quad, bounds.upper_years)
        # Hand formula (independent of the module) for the midpoint closed form.
        rho_e = 1.0e-9 * math.exp(-(h_re - 150.0e3) / 60.0e3)
        rho_0 = 1.0e-9 * math.exp(-(h0 - 150.0e3) / 60.0e3)
        c = math.sqrt(EARTH_MU * (EARTH_RADIUS + 0.5 * (h0 + h_re)))
        hand = 60.0e3 * b / c * (1.0 / rho_0 - 1.0 / rho_e) / YEAR
        self.assertAlmostEqual(bounds.midpoint_years, hand, delta=1e-9 * hand)
        # The weight 1/rho concentrates the integral near the start, so the
        # exact value sits between the bounds; the mid-radius closed form is
        # within half the bracket (measured 0.68 % against a 2.2 % gap).
        self.assertAlmostEqual(bounds.midpoint_years, quad, delta=0.5 * gap * quad)

    def test_integration_matches_independent_simpson_quadrature(self) -> None:
        """Time stepping vs Simpson on the separated integral, relative 5e-5."""
        for h0 in (350.0e3, 500.0e3, 650.0e3):
            with self.subTest(h0=h0):
                ref = simpson_lifetime(h0, 50.0, TH1, 120.0e3)
                run = od.orbital_decay(h0, 50.0, TH1, None, horizon_years=1.0e4)
                self.assertAlmostEqual(run.lifetime_years / ref, 1.0, delta=5.0e-5)
                quad = od.static_lifetime(h0, 50.0, TH1)
                self.assertAlmostEqual(quad / ref, 1.0, delta=1.0e-9)

    def test_layered_activity_dependent_atmosphere_against_simpson(self) -> None:
        """Three layers with activity scaling: quadrature splits at the bases."""
        for activity in (0.6, 1.0, 1.8):
            with self.subTest(activity=activity):
                ref = simpson_lifetime(650.0e3, 40.0, TH, 120.0e3, activity)
                quad = od.static_lifetime(650.0e3, 40.0, TH, activity)
                self.assertAlmostEqual(quad / ref, 1.0, delta=1.0e-7)
                run = od.orbital_decay(650.0e3, 40.0, TH, activity, horizon_years=2000)
                self.assertAlmostEqual(run.lifetime_years / ref, 1.0, delta=5.0e-5)

    def test_step_size_convergence(self) -> None:
        """Refining the altitude fraction reduces the error toward the finest run."""
        runs = [
            od.orbital_decay(
                500.0e3, 50.0, TH, None, altitude_fraction=f
            ).lifetime_years
            for f in (0.1, 0.05, 0.025)
        ]
        e1 = abs(runs[0] - runs[2])
        e2 = abs(runs[1] - runs[2])
        self.assertLess(e1 / runs[2], 2.0e-4)
        self.assertLess(e2, e1)

    def test_ballistic_coefficient(self) -> None:
        """B = m / (C_D A)."""
        self.assertEqual(od.ballistic_coefficient(100.0, 2.0, 0.5), 100.0)


class AtmosphereAndSolarTests(unittest.TestCase):
    """The synthetic density relation and the synthetic activity process."""

    def test_density_is_continuous_and_scales_with_activity(self) -> None:
        """Continuity at layer bases for every F; rho_0 s^p at the first base."""
        for f in (0.5, 1.0, 2.5):
            self.assertAlmostEqual(
                TH.density(150.0e3, f) / (2.0e-9 * f), 1.0, delta=1e-12
            )
            for base in (300.0e3, 500.0e3):
                below = TH.density(base - 1e-3, f)
                above = TH.density(base + 1e-3, f)
                self.assertAlmostEqual(below / above, 1.0, delta=1e-6)
        # Higher activity: more density above the first base, larger scale height.
        self.assertGreater(TH.density(500.0e3, 1.5), TH.density(500.0e3, 1.0))
        self.assertAlmostEqual(TH.scale_height(400.0e3, 4.0), 55.0e3 * 2.0, delta=1e-9)
        values = [TH.density(h * 1.0e3, 1.0) for h in range(120, 900, 10)]
        self.assertTrue(all(b < a for a, b in zip(values, values[1:], strict=False)))

    def test_single_layer_closed_form_density(self) -> None:
        """Single layer: rho_0 exp(-(h - h_0)/H), also below its base."""
        self.assertAlmostEqual(
            TH1.density(450.0e3, 3.0) / (1.0e-9 * math.exp(-5.0)), 1.0, delta=1e-12
        )
        self.assertAlmostEqual(
            TH1.density(90.0e3, 1.0) / (1.0e-9 * math.exp(1.0)), 1.0, delta=1e-12
        )

    def test_solar_cycle_and_noise(self) -> None:
        """Mean over a period, the extremes, and exact interpolation at knots."""
        cycle = od.demonstration_solar_activity(phase=0.1, amplitude=0.4)
        n = 4000
        mean = sum(cycle.index(11.0 * (k + 0.5) / n) for k in range(n)) / n
        self.assertAlmostEqual(mean, 1.0, delta=1e-9)
        peak = max(cycle.index(11.0 * k / n) for k in range(n))
        low = min(cycle.index(11.0 * k / n) for k in range(n))
        self.assertAlmostEqual(peak, 1.4, delta=1e-4)
        self.assertAlmostEqual(low, 0.6, delta=1e-4)
        noise = od.SolarActivity(
            mean=2.0,
            noise_sigma=0.3,
            noise_values=(0.0, 1.0, -1.0),
            noise_interval_years=2.0,
        )
        self.assertAlmostEqual(noise.index(0.0), 2.0, delta=1e-12)
        self.assertAlmostEqual(noise.index(2.0), 2.0 * math.exp(0.3), delta=1e-12)
        self.assertAlmostEqual(noise.index(3.0), 2.0 * math.exp(0.0), delta=1e-12)
        self.assertAlmostEqual(noise.index(50.0), 2.0 * math.exp(-0.3), delta=1e-12)
        draws = od.synthetic_noise_values(2000, seed=3)
        self.assertEqual(draws, od.synthetic_noise_values(2000, seed=3))
        self.assertAlmostEqual(sum(draws) / 2000, 0.0, delta=4.0 / math.sqrt(2000))


class MonotonicityTests(unittest.TestCase):
    """Lifetime grows with altitude and with the ballistic coefficient."""

    def test_monotone_in_altitude_and_ballistic_with_a_solar_cycle(self) -> None:
        """Pathwise monotone, also with a cycle and noise (no trajectory crossing)."""
        noise = od.synthetic_noise_values(12, seed=7)
        for phase in (0.0, 0.3, 0.7):
            solar = od.demonstration_solar_activity(
                phase=phase,
                amplitude=0.5,
                noise_sigma=0.3,
                noise_values=noise,
                noise_interval_years=3.0,
            )
            with self.subTest(phase=phase):
                previous = 0.0
                for h in range(400, 801, 50):
                    life = od.orbital_decay(
                        h * 1.0e3, 45.0, TH, solar, horizon_years=3000
                    ).lifetime_years
                    self.assertGreater(life, previous)
                    previous = life
                previous = 0.0
                for b in (10.0, 20.0, 40.0, 80.0, 160.0):
                    life = od.orbital_decay(
                        550.0e3, b, TH, solar, horizon_years=3000
                    ).lifetime_years
                    self.assertGreater(life, previous)
                    previous = life

    def test_density_factor_is_an_exact_rescaling_of_the_ballistic_coefficient(
        self,
    ) -> None:
        """At fixed activity T(k rho, B) = T(rho, B / k): same ODE, rescaled."""
        a = od.static_lifetime(550.0e3, 40.0, TH, density_factor=2.0)
        b = od.static_lifetime(550.0e3, 20.0, TH)
        self.assertAlmostEqual(a / b, 1.0, delta=1e-12)

    def test_horizon_censoring_and_static_tail(self) -> None:
        """A long-lived orbit is flagged and optionally continued."""
        censored = od.orbital_decay(750.0e3, 45.0, TH, None, horizon_years=10.0)
        self.assertFalse(censored.reentered)
        self.assertEqual(censored.stop_reason, "horizon")
        self.assertAlmostEqual(censored.lifetime_years, 10.0, delta=1e-9)
        tail = od.orbital_decay(
            750.0e3, 45.0, TH, None, horizon_years=10.0, static_tail=True
        )
        full = od.orbital_decay(750.0e3, 45.0, TH, None, horizon_years=3000.0)
        self.assertEqual(tail.stop_reason, "horizon_extrapolated")
        self.assertAlmostEqual(
            tail.lifetime_years / full.lifetime_years, 1.0, delta=1e-4
        )


class EccentricTests(unittest.TestCase):
    """Orbit-averaged (a, e) equations."""

    def test_averaged_rates_against_true_anomaly_quadrature(self) -> None:
        """Eccentric-anomaly average vs Gauss-equation average over true anomaly.

        Independent oracle: ``da/dt = 2 a^2 v f / mu`` and
        ``de/dt = 2 (e + cos nu) f / v`` with ``f = -rho v^2 / (2 B)``,
        averaged with ``dt = r^2 d nu / h`` over a revolution (Simpson, 20000
        panels), at a frozen activity.
        """
        b, a, e = 30.0, EARTH_RADIUS + 450.0e3, 0.08
        ctx = od._Context(TH, None, 1.0, 1.0, b, 128)
        da, de = ctx.averaged_rates(0.0, a, e)
        p = a * (1.0 - e * e)
        h_mom = math.sqrt(EARTH_MU * p)
        period = 2.0 * math.pi * math.sqrt(a**3 / EARTH_MU)
        n = 20000
        width = 2.0 * math.pi / n

        def integrands(nu):
            r = p / (1.0 + e * math.cos(nu))
            v = math.sqrt(EARTH_MU * (2.0 / r - 1.0 / a))
            f = -TH._density(r - EARTH_RADIUS, 1.0) * v * v / (2.0 * b)
            weight = r * r / h_mom
            return (
                2.0 * a * a * v * f / EARTH_MU * weight,
                2.0 * (e + math.cos(nu)) * f / v * weight,
            )

        sum_a = sum_e = 0.0
        for k in range(n + 1):
            wk = 1.0 if k in (0, n) else (4.0 if k % 2 else 2.0)
            ia, ie = integrands(k * width)
            sum_a += wk * ia
            sum_e += wk * ie
        ref_a = sum_a * width / 3.0 / period
        ref_e = sum_e * width / 3.0 / period
        self.assertAlmostEqual(da / ref_a, 1.0, delta=1e-6)
        self.assertAlmostEqual(de / ref_e, 1.0, delta=1e-6)

    def test_small_eccentricity_recovers_the_circular_lifetime(self) -> None:
        """e0 = 1e-6 differs from the circular case by ~ (a e / H) (mean altitude)."""
        circular = od.orbital_decay(500.0e3, 45.0, TH, None).lifetime_years
        near = od.orbital_decay(
            500.0e3, 45.0, TH, None, eccentricity=1.0e-6
        ).lifetime_years
        self.assertAlmostEqual(near / circular, 1.0, delta=5.0e-4)

    def test_perigee_nearly_constant_while_apogee_falls_and_e_decays(self) -> None:
        """Qualitative law of drag on an eccentric orbit, measured on a run.

        Early on the apogee distance falls several times faster than the
        perigee distance and the eccentricity decreases monotonically.
        """
        run = od.orbital_decay(
            500.0e3, 45.0, TH, None, eccentricity=0.1, horizon_years=500, record=True
        )
        self.assertTrue(run.reentered)
        ecc = run.eccentricities
        self.assertTrue(all(b <= a + 1e-15 for a, b in zip(ecc, ecc[1:], strict=False)))
        k = next(i for i, e in enumerate(ecc) if e < 0.07)
        perigee = EARTH_RADIUS + run.altitudes[k]
        apogee = perigee * (1.0 + ecc[k]) / (1.0 - ecc[k])
        perigee0 = EARTH_RADIUS + 500.0e3
        apogee0 = perigee0 * 1.1 / 0.9
        self.assertGreater(apogee0 - apogee, 0.0)
        self.assertLess(perigee0 - perigee, 0.2 * (apogee0 - apogee))

    def test_quadrature_points_converge(self) -> None:
        """The orbit-average quadrature converges in the node count."""
        values = [
            od.orbital_decay(
                300.0e3,
                45.0,
                TH,
                None,
                eccentricity=0.3,
                horizon_years=500,
                quadrature_points=n,
            ).lifetime_years
            for n in (32, 64, 128, 256)
        ]
        self.assertLess(abs(values[2] - values[3]), abs(values[0] - values[3]))
        self.assertAlmostEqual(values[2] / values[3], 1.0, delta=1.0e-3)

    def test_circularised_tail_is_continuous_with_the_full_run(self) -> None:
        """The re-phased circular tail continues a run in a cycle with noise."""
        noise = od.synthetic_noise_values(6, seed=2)
        solar = od.demonstration_solar_activity(
            phase=0.7,
            amplitude=0.5,
            noise_sigma=0.1,
            noise_values=noise,
            noise_interval_years=2.0,
        )
        full = od.orbital_decay(600.0e3, 45.0, TH, solar, horizon_years=200)
        half = od.orbital_decay(600.0e3, 45.0, TH, solar, horizon_years=1.3)
        ctx = od._Context(TH, solar, 1.0, 1.0, 45.0, 64)
        tail = od._shifted_tail(
            ctx,
            1.3 * YEAR,
            EARTH_RADIUS + half.final_altitude,
            120.0e3,
            200.0 * YEAR,
            0.05,
            0.7 * YEAR,
            10**6,
            half.steps,
        )
        self.assertFalse(half.reentered)
        self.assertTrue(tail.reentered)
        self.assertAlmostEqual(
            tail.lifetime_years / full.lifetime_years, 1.0, delta=2e-3
        )


class ReliabilityTests(unittest.TestCase):
    """P(lifetime > deadline) by FORM, subset simulation and Monte Carlo."""

    def test_model_structure_and_pickling(self) -> None:
        """Inputs, outputs, margin identity and picklability."""
        model = lifetime_model(600.0e3, full=True, knots=3, sigma=0.2)
        self.assertEqual(model.dimension, 7 + 3)
        point = {
            "initial_altitude": 600.0e3,
            "drag_coefficient": 2.2,
            "area_to_mass": 0.01,
            "attitude_factor": 0.85,
            "solar_phase": 0.3,
            "solar_amplitude": 0.4,
            "density_factor": 1.0,
            "solar_noise_0": 0.0,
            "solar_noise_1": 0.0,
            "solar_noise_2": 0.0,
        }
        out = model.evaluate(point)
        self.assertAlmostEqual(
            out["lifetime_years"] + out["compliance_margin_years"],
            DEADLINE,
            delta=1e-12,
        )
        self.assertAlmostEqual(out["log_lifetime"], math.log(out["lifetime_years"]))
        ref = od.orbital_decay(
            600.0e3,
            1.0 / (2.2 * 0.85 * 0.01),
            TH,
            od.demonstration_solar_activity(phase=0.3, amplitude=0.4),
            horizon_years=2.0 * DEADLINE,
            altitude_fraction=0.1,
            static_tail=True,
        )
        self.assertAlmostEqual(out["lifetime_years"], ref.lifetime_years, delta=1e-9)
        pickle.loads(pickle.dumps(model.model))

    def test_form_subset_and_monte_carlo_agree_with_a_fixed_phase(self) -> None:
        """Smooth case (solar phase fixed): the three methods agree.

        Measured in development at 640 km: FORM 0.177, Monte Carlo 0.167
        (se 0.012, 1000 samples), subset 0.167. Bands: FORM within 0.06 of
        Monte Carlo (FORM linearises a curved limit state), subset within
        4 combined standard errors.
        """
        model = lifetime_model(640.0e3)
        form = od.noncompliance_probability(model, DEADLINE, method="form")
        mc = od.noncompliance_probability(
            model, DEADLINE, method="monte_carlo", samples=600, seed=11
        )
        sub = od.noncompliance_probability(
            model, DEADLINE, method="subset", samples=300, seed=5
        )
        self.assertTrue(form.converged)
        self.assertTrue(sub.converged)
        self.assertGreater(mc.probability, 0.05)
        self.assertLess(abs(form.probability - mc.probability), 0.06)
        band = 4.0 * math.hypot(mc.standard_error, sub.standard_error)
        self.assertLess(abs(sub.probability - mc.probability), band)

    def test_subset_matches_monte_carlo_with_a_random_phase(self) -> None:
        """Full model (phase, amplitude, C_D, attitude random): MC vs subset.

        FORM is *not* asserted here: the lifetime is periodic in the phase,
        which a linearisation in standard-normal space cannot represent
        (development run at 640 km: FORM 0.407 against Monte Carlo 0.318).
        """
        model = lifetime_model(640.0e3, full=True)
        mc = od.noncompliance_probability(
            model, DEADLINE, method="monte_carlo", samples=600, seed=4
        )
        sub = od.noncompliance_probability(
            model, DEADLINE, method="subset", samples=300, seed=6
        )
        band = 4.0 * math.hypot(mc.standard_error, sub.standard_error)
        self.assertLess(abs(sub.probability - mc.probability), band)
        self.assertGreater(mc.probability, 0.1)
        self.assertLess(mc.probability, 0.6)


class SensitivityTests(unittest.TestCase):
    """Sobol ranking of the (log) lifetime, measured."""

    @staticmethod
    def total_effects(altitude: float, size: int, seed: int) -> dict[str, float]:
        """Return total-effect indices of ``log_lifetime``."""
        model = lifetime_model(altitude, full=True)
        result = sobol_analysis(model, "log_lifetime", size, seed=seed)
        return {
            name: item.estimate
            for name, item in zip(result.input_names, result.total_effect, strict=True)
        }

    def test_phase_and_density_factor_lead_at_a_low_altitude(self) -> None:
        """At 550 km (lifetime ~ 2 cycles) the phase competes with the density error.

        Measured in development (size 256, seed 1): total effects solar_phase
        0.51, density_factor 0.38, area_to_mass 0.19, solar_amplitude 0.16,
        attitude_factor 0.07, drag_coefficient 0.05, initial_altitude 0.01.
        The test asserts the top two only.
        """
        total = self.total_effects(550.0e3, 256, 1)
        ranked = sorted(total, key=total.__getitem__, reverse=True)
        self.assertEqual(set(ranked[:2]), {"solar_phase", "density_factor"})
        self.assertGreater(total["solar_phase"], 0.25)
        self.assertLess(total["initial_altitude"], 0.1)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_density_factor_dominates_at_a_high_altitude(self) -> None:
        """At 750 km the lifetime spans many cycles: the phase washes out.

        Measured (size 512, seed 1): density_factor 0.47, solar_amplitude
        0.21, area_to_mass 0.21, attitude_factor 0.06, drag_coefficient 0.06,
        solar_phase 0.01.
        """
        total = self.total_effects(750.0e3, 512, 1)
        ranked = sorted(total, key=total.__getitem__, reverse=True)
        self.assertEqual(ranked[0], "density_factor")
        self.assertLess(total["solar_phase"], 0.1)


def sail_builder(sail: float):
    """Return ``build(h)`` with the area-to-mass ratio scaled by ``sail``."""

    def build(h: float):
        return lifetime_model(h, area_to_mass=LogNormal(math.log(0.01 * sail), 0.2))

    return build


class DesignTests(unittest.TestCase):
    """Highest compliant altitude and the effect of a drag sail."""

    def test_probability_is_monotone_in_altitude_with_common_random_numbers(
        self,
    ) -> None:
        """Same seed: the Monte Carlo probability never decreases with altitude."""
        values = [
            od.noncompliance_probability(
                lifetime_model(h * 1.0e3),
                DEADLINE,
                method="monte_carlo",
                samples=150,
                seed=9,
            ).probability
            for h in (560, 600, 630, 660, 700)
        ]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:], strict=False)))
        self.assertGreater(values[-1], values[0])

    def test_maximum_altitude_root_and_drag_sail(self) -> None:
        """Bisection root of P(h) = 0.1; a 4x area sail raises it a lot.

        Measured in development (FORM, 3 km tolerance, altitude_fraction 0.1):
        630 km without a sail, about 764 km with a 4x area-to-mass sail. The
        root is checked against an independent Monte Carlo probability within
        3.5 standard errors plus a FORM bias band of 0.03.
        """
        target = 0.1
        plain = od.maximum_compliant_altitude(
            sail_builder(1.0),
            DEADLINE,
            target,
            (560.0e3, 700.0e3),
            altitude_tolerance=3000.0,
        )
        sail = od.maximum_compliant_altitude(
            sail_builder(4.0),
            DEADLINE,
            target,
            (640.0e3, 820.0e3),
            altitude_tolerance=3000.0,
        )
        for factor, result in ((1.0, plain), (4.0, sail)):
            self.assertLessEqual(result.probability, target)
            self.assertLess(result.upper_altitude - result.altitude, 3000.0 + 1e-9)
            above = od.noncompliance_probability(
                sail_builder(factor)(result.upper_altitude), DEADLINE
            ).probability
            self.assertGreater(above, target)
        self.assertGreater(sail.altitude, plain.altitude + 50.0e3)
        mc = od.noncompliance_probability(
            sail_builder(1.0)(plain.altitude),
            DEADLINE,
            method="monte_carlo",
            samples=600,
            seed=3,
        )
        self.assertLess(abs(mc.probability - target), 3.5 * mc.standard_error + 0.03)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_monte_carlo_root_agrees_with_form_root(self) -> None:
        """The two root finders agree within 25 km (MC noise, FORM bias)."""
        form = od.maximum_compliant_altitude(
            sail_builder(1.0),
            DEADLINE,
            0.1,
            (560.0e3, 700.0e3),
            altitude_tolerance=3000.0,
        )
        mc = od.maximum_compliant_altitude(
            sail_builder(1.0),
            DEADLINE,
            0.1,
            (560.0e3, 700.0e3),
            method="monte_carlo",
            samples=200,
            seed=1,
            altitude_tolerance=3000.0,
        )
        self.assertLess(abs(form.altitude - mc.altitude), 25.0e3)


class FailureCaseTests(unittest.TestCase):
    """Invalid inputs are rejected; nothing is clipped."""

    def test_orbit_already_below_the_reentry_altitude(self) -> None:
        """Initial (or perigee) altitude must exceed the reentry altitude."""
        with self.assertRaises(ValueError):
            od.orbital_decay(100.0e3, 45.0, TH, None)
        with self.assertRaises(ValueError):
            od.orbital_decay(120.0e3, 45.0, TH, None)
        with self.assertRaises(ValueError):
            od.static_lifetime(110.0e3, 45.0, TH)
        with self.assertRaises(ValueError):
            od.exponential_lifetime_bounds(100.0e3, 45.0, 150.0e3, 1.0e-9, 60.0e3)
        with self.assertRaises(ValueError):
            od.orbital_decay(110.0e3, 45.0, TH, None, eccentricity=0.1)
        with self.assertRaises(ValueError):
            od.orbital_lifetime_model(
                TH,
                DEADLINE,
                initial_altitude=100.0e3,
                drag_coefficient=LogNormal(0.0, 0.1),
                area_to_mass=0.01,
            )

    def test_invalid_scalars(self) -> None:
        """Non-positive B, bad types, bad ranges."""
        for bad in (0.0, -5.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                od.orbital_decay(500.0e3, bad, TH, None)
        with self.assertRaises(TypeError):
            od.orbital_decay(500.0e3, True, TH, None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            od.orbital_decay(500.0e3, 45.0, TH, None, density_factor=0.0)
        with self.assertRaises(ValueError):
            od.orbital_decay(500.0e3, 45.0, TH, None, eccentricity=1.0)
        with self.assertRaises(ValueError):
            od.orbital_decay(500.0e3, 45.0, TH, -1.0)
        with self.assertRaises(TypeError):
            od.orbital_decay(500.0e3, 45.0, "atmosphere", None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            od.ballistic_coefficient(10.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            od.SolarActivity(amplitude=1.0)
        with self.assertRaises(ValueError):
            od.SolarActivity(noise_sigma=-0.1)
        with self.assertRaises(ValueError):
            od.SyntheticThermosphere((300.0e3, 200.0e3), (50.0e3, 50.0e3), 1e-9)
        with self.assertRaises(ValueError):
            od.SyntheticThermosphere((200.0e3,), (0.0,), 1e-9)
        with self.assertRaises(ValueError):
            TH.density(300.0e3, 0.0)

    def test_budget_exhaustion_is_reported_not_hidden(self) -> None:
        """A step budget that is too small ends with stop_reason max_steps."""
        run = od.orbital_decay(500.0e3, 45.0, TH, None, max_steps=5)
        self.assertFalse(run.reentered)
        self.assertEqual(run.stop_reason, "max_steps")
        self.assertEqual(run.steps, 5)

    def test_model_and_design_validation(self) -> None:
        """Model construction and the root finder reject bad arguments."""
        with self.assertRaises(ValueError):
            od.orbital_lifetime_model(
                TH,
                DEADLINE,
                initial_altitude=600.0e3,
                drag_coefficient=2.2,
                area_to_mass=0.01,
            )
        with self.assertRaises(ValueError):
            od.orbital_lifetime_model(
                TH,
                DEADLINE,
                initial_altitude=Normal(600.0e3, 1000.0),
                drag_coefficient=2.2,
                area_to_mass=0.01,
                noise_knots=1,
            )
        with self.assertRaises(TypeError):
            od.orbital_lifetime_model(
                TH,
                DEADLINE,
                initial_altitude=True,  # type: ignore[arg-type]
                drag_coefficient=2.2,
                area_to_mass=0.01,
            )
        # An amplitude above one drawn from a wide law raises in the forward map.
        model = od.orbital_lifetime_model(
            TH,
            DEADLINE,
            initial_altitude=600.0e3,
            drag_coefficient=Uniform(2.0, 2.4),
            area_to_mass=0.01,
            solar_amplitude=Uniform(0.0, 2.0),
        )
        with self.assertRaises(ValueError):
            model.evaluate({"solar_amplitude": 1.5, "drag_coefficient": 2.2})
        build = sail_builder(1.0)
        with self.assertRaises(ValueError):
            od.maximum_compliant_altitude(build, DEADLINE, 0.1, (700.0e3, 560.0e3))
        with self.assertRaises(ValueError):  # P already above target at the low end
            od.maximum_compliant_altitude(build, DEADLINE, 0.1, (690.0e3, 700.0e3))
        with self.assertRaises(ValueError):  # P below target at the high end
            od.maximum_compliant_altitude(build, DEADLINE, 0.1, (500.0e3, 520.0e3))
        with self.assertRaises(ValueError):
            od.maximum_compliant_altitude(build, DEADLINE, 1.5, (500.0e3, 700.0e3))
        with self.assertRaises(ValueError):
            od.noncompliance_probability(
                build(600.0e3),
                DEADLINE,
                method="bogus",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            od.noncompliance_probability("model", DEADLINE)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
