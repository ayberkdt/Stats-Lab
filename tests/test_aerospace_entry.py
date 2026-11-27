"""Tests for ``probstat.aerospace.entry``: capsule entry, heating and TPS sizing.

Verification order: closed forms first (Sutton-Graves value, Allen-Eggers
formulas against their own consistency identities and against quadrature,
the integrator against Allen-Eggers in the regime where its assumptions hold,
Kepler conservation laws without drag), then exact scaling identities of the
model, then the heat-shield formulas, then the uncertainty-quantification
demonstrations (Sobol ranking, FORM / subset simulation / Monte Carlo
agreement, the steep-versus-shallow trade-off, the entry corridor) and the
failure cases. All vehicle and material numbers are synthetic.

Tolerances are stated next to each assertion with their justification. The
studies of the slow tier run only with ``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS, STANDARD_GRAVITY
from probstat.aerospace.entry import (
    SUTTON_GRAVES_EARTH_K,
    AllenEggersEntry,
    EntryVehicle,
    ExponentialAtmosphere,
    HeatShield,
    entry_corridor,
    entry_model,
    heat_shield_sizing,
    simulate_entry,
    sutton_graves_heat_flux,
)
from probstat.distributions import Normal
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis_all

# Synthetic Apollo-/Orion-like capsule class (order of magnitude only):
# 5000 kg, C_D = 1.3, A = 12 m^2 -> beta = 320.5 kg/m^2, R_n = 4.7 m, L/D = 0.3.
_VEHICLE = EntryVehicle.from_mass(5000.0, 1.30, 12.0, 4.7, 0.3)
_BALLISTIC = EntryVehicle.from_mass(5000.0, 1.30, 12.0, 4.7, 0.0)
# Synthetic ablator: rho = 500 kg/m^3, H_eff = 2.5e7 J/kg, f = 1.2,
# alpha = 2.7e-7 m^2/s, T_s = 2500 K, T_bond = 450 K, T_0 = 250 K.
_DESIGN = 0.027


def _shield(design: float = _DESIGN) -> HeatShield:
    """Synthetic shield with a given design thickness.

    Args:
        design: Installed thickness in m.

    Returns:
        The shield.

    References:
        [EQ-AERO-ENTRY-006]. Synthetic numbers.

    """
    return HeatShield(500.0, 2.5e7, 1.2, 2.7e-7, 2500.0, 450.0, 250.0, design)


_FAST = {"max_step": 3.0, "resolution": 0.5}


def _model(
    *,
    fix_scales: bool = False,
    sigma_gamma_deg: float = 0.12,
    gamma_deg: float = -7.9,
    design: float = _DESIGN,
    on_failure: str = "raise",
):  # type: ignore[no-untyped-def]
    """Lunar-return-speed entry model with the standard synthetic dispersions.

    Args:
        fix_scales: Fix the Sutton-Graves scale and the heat of ablation.
        sigma_gamma_deg: Flight-path-angle standard deviation in degrees.
        gamma_deg: Mean flight-path angle in degrees.
        design: Design thickness in m.
        on_failure: Failure policy.

    Returns:
        The model.

    References:
        [EQ-AERO-ENTRY-009]. Synthetic numbers.

    """
    return entry_model(
        _VEHICLE,
        _shield(design),
        entry_speed=Normal(11000.0, 100.0),
        flight_path_angle=Normal(
            math.radians(gamma_deg), math.radians(sigma_gamma_deg)
        ),
        ballistic_coefficient=Normal(320.0, 25.0),
        nose_radius=Normal(4.7, 0.1),
        density_scale=Normal(1.0, 0.1),
        heating_scale=1.0 if fix_scales else Normal(1.0, 0.1),
        heat_of_ablation=2.5e7 if fix_scales else Normal(2.5e7, 2.5e6),
        on_failure=on_failure,  # type: ignore[arg-type]
        **_FAST,
    )


class SuttonGravesTests(unittest.TestCase):
    """Tests of ``probstat.aerospace.entry``: SuttonGravesTests."""

    def test_closed_value_and_scalings(self) -> None:
        """Check ``q = k sqrt(rho/R) V^3`` exactly, with the documented constant."""
        q = sutton_graves_heat_flux(1.0e-4, 1.0e4, 1.0)
        expected = SUTTON_GRAVES_EARTH_K * 1.0e-2 * 1.0e12
        self.assertAlmostEqual(q, expected, delta=1e-6 * q)
        # Table II of the report: K(air) = 0.1113 kg/(s m^1.5 atm^0.5).
        self.assertAlmostEqual(SUTTON_GRAVES_EARTH_K, 1.7483e-4, delta=1e-8)
        # Scalings: V^3, sqrt(rho), 1/sqrt(R).
        self.assertAlmostEqual(
            sutton_graves_heat_flux(1e-4, 2e4, 1.0) / q, 8.0, places=12
        )
        self.assertAlmostEqual(
            sutton_graves_heat_flux(4e-4, 1e4, 1.0) / q, 2.0, places=12
        )
        self.assertAlmostEqual(
            sutton_graves_heat_flux(1e-4, 1e4, 4.0) / q, 0.5, places=12
        )
        self.assertEqual(sutton_graves_heat_flux(0.0, 1e4, 1.0), 0.0)

    def test_validation(self) -> None:
        """Negative density, speed or non-positive radius are rejected."""
        with self.assertRaises(ValueError):
            sutton_graves_heat_flux(-1.0, 1e4, 1.0)
        with self.assertRaises(ValueError):
            sutton_graves_heat_flux(1e-4, 1e4, 0.0)
        with self.assertRaises(ValueError):
            sutton_graves_heat_flux(1e-4, -1.0, 1.0)
        with self.assertRaises(TypeError):
            sutton_graves_heat_flux("a", 1e4, 1.0)  # type: ignore[arg-type]


class AllenEggersTests(unittest.TestCase):
    """The closed-form solution against its own identities and quadrature."""

    def setUp(self) -> None:
        """Build a steep ballistic case."""
        self.ae = AllenEggersEntry(
            300.0, math.radians(-35.0), 7500.0, 4.7, entry_altitude=160e3
        )

    def test_peak_identities(self) -> None:
        """Deceleration and heat flux evaluated at the stated peaks equal the peaks."""
        ae = self.ae
        h_a = ae.peak_deceleration_altitude
        self.assertAlmostEqual(
            ae.deceleration(h_a),
            ae.peak_deceleration,
            delta=1e-12 * ae.peak_deceleration,
        )
        self.assertAlmostEqual(
            ae.speed(h_a), ae.peak_deceleration_speed, delta=1e-9 * ae.speed(h_a)
        )
        h_q = ae.peak_heat_flux_altitude
        self.assertAlmostEqual(
            ae.heat_flux(h_q), ae.peak_heat_flux, delta=1e-12 * ae.peak_heat_flux
        )
        self.assertAlmostEqual(
            ae.speed(h_q), ae.peak_heat_flux_speed, delta=1e-9 * ae.speed(h_q)
        )
        # Peak heating occurs H ln 3 above the peak deceleration.
        self.assertAlmostEqual(
            h_q - h_a, ae.atmosphere.scale_height * math.log(3.0), places=9
        )
        # Peak deceleration formula V_e^2 sin|g| / (2 e H).
        expected = (
            ae.infinity_speed**2
            * abs(math.sin(ae.flight_path_angle))
            / (2.0 * math.e * ae.atmosphere.scale_height)
        )
        self.assertAlmostEqual(ae.peak_deceleration, expected, delta=1e-12 * expected)

    def test_peaks_are_maxima_on_a_scan(self) -> None:
        """A dense altitude scan finds no larger value than the closed-form peaks."""
        ae = self.ae
        heights = [160e3 - 20.0 * i for i in range(7000)]
        a_scan = max(ae.deceleration(h) for h in heights)
        q_scan = max(ae.heat_flux(h) for h in heights)
        # Scan resolution 20 m of a 7.2 km scale height: second-order error.
        self.assertLessEqual(a_scan, ae.peak_deceleration * (1.0 + 1e-12))
        self.assertGreater(a_scan, ae.peak_deceleration * (1.0 - 1e-5))
        self.assertLessEqual(q_scan, ae.peak_heat_flux * (1.0 + 1e-12))
        self.assertGreater(q_scan, ae.peak_heat_flux * (1.0 - 1e-5))

    def test_heat_load_against_quadrature(self) -> None:
        """Q = int q dh / (V sin|g|) by Simpson's rule equals the erf formula."""
        ae = self.ae
        s = abs(math.sin(ae.flight_path_angle))
        h_end = 20e3
        n = 20000  # even; the integrand is smooth on a 140 km range
        width = (160e3 - h_end) / n
        total = 0.0
        for i in range(n + 1):
            h = 160e3 - i * width
            weight = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            total += weight * ae.heat_flux(h) / (ae.speed(h) * s)
        total *= width / 3.0
        self.assertAlmostEqual(total, ae.heat_load(h_end), delta=1e-9 * total)

    def test_total_heat_load_and_trade_off_scalings(self) -> None:
        """Q_inf = k V^2 sqrt(pi beta H / (R s)); peak g ~ s; Q ~ 1/sqrt(s)."""
        atm = ExponentialAtmosphere()

        def make(angle_deg: float) -> AllenEggersEntry:
            return AllenEggersEntry(300.0, math.radians(angle_deg), 7500.0, 4.7, atm)

        a, b = make(-20.0), make(-50.0)
        s_a = math.sin(math.radians(20.0))
        s_b = math.sin(math.radians(50.0))
        q_inf = (
            SUTTON_GRAVES_EARTH_K
            * 7500.0**2
            * math.sqrt(math.pi * 300.0 * atm.scale_height / (4.7 * s_a))
        )
        # The ground is deep inside the atmosphere: erf(sqrt(c rho_0)) = 1 to
        # machine precision, so heat_load(0) is the total.
        self.assertAlmostEqual(a.heat_load(0.0), q_inf, delta=1e-12 * q_inf)
        self.assertAlmostEqual(
            b.peak_deceleration / a.peak_deceleration, s_b / s_a, places=12
        )
        self.assertAlmostEqual(
            b.heat_load(0.0) / a.heat_load(0.0), math.sqrt(s_a / s_b), places=12
        )

    def test_validation(self) -> None:
        """Ascending, level or non-positive parameters are rejected."""
        with self.assertRaises(ValueError):
            AllenEggersEntry(300.0, 0.1, 7500.0, 4.7)
        with self.assertRaises(ValueError):
            AllenEggersEntry(-300.0, -0.1, 7500.0, 4.7)
        with self.assertRaises(ValueError):
            AllenEggersEntry(300.0, -0.1, 7500.0, 0.0)
        with self.assertRaises(TypeError):
            AllenEggersEntry(300.0, -0.1, 7500.0, 4.7, atmosphere="us76")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ExponentialAtmosphere(scale_height=0.0)


class IntegratorAgainstClosedFormTests(unittest.TestCase):
    """The RK4 integrator against Allen-Eggers where its assumptions hold."""

    ATM = ExponentialAtmosphere()

    def _pair(self, beta: float, radius: float, speed: float, angle_deg: float):  # type: ignore[no-untyped-def]
        """Return ``(ae, idealised, full)`` for one case (terminal altitude 20 km)."""
        gam = math.radians(angle_deg)
        ae = AllenEggersEntry(beta, gam, speed, radius, self.ATM, entry_altitude=160e3)
        vehicle = EntryVehicle(beta, radius)
        kwargs = {
            "altitude": 160e3,
            "speed": speed,
            "flight_path_angle": gam,
            "atmosphere": self.ATM,
            "terminal_altitude": 20e3,
            "max_step": 1.0,
        }
        ideal = simulate_entry(vehicle, idealised=True, **kwargs)  # type: ignore[arg-type]
        full = simulate_entry(vehicle, **kwargs)  # type: ignore[arg-type]
        return ae, ideal, full

    def test_idealised_run_reproduces_the_closed_form(self) -> None:
        """Straight line, no gravity: agreement is the integration error alone.

        Measured (this file's cases, resolution 0.2): peak values within 1e-4,
        peak altitudes within 10 m of a 7200 m scale height, heat load within
        3e-7. The tolerances below are 5 to 10 times the measured errors.
        """
        for beta, radius, speed, angle in (
            (300.0, 4.7, 7500.0, -20.0),
            (300.0, 4.7, 7500.0, -45.0),
            (300.0, 4.7, 7500.0, -90.0),
            (100.0, 1.0, 7500.0, -60.0),
        ):
            with self.subTest(beta=beta, angle=angle):
                ae, ideal, _ = self._pair(beta, radius, speed, angle)
                self.assertEqual(ideal.status, "landed")
                self.assertTrue(ideal.peaks_interior)
                self.assertAlmostEqual(
                    ideal.peak_deceleration_g * STANDARD_GRAVITY / ae.peak_deceleration,
                    1.0,
                    delta=5e-4,
                )
                self.assertAlmostEqual(
                    ideal.peak_heat_flux / ae.peak_heat_flux, 1.0, delta=5e-4
                )
                self.assertAlmostEqual(
                    ideal.peak_deceleration_altitude,
                    ae.peak_deceleration_altitude,
                    delta=60.0,
                )
                self.assertAlmostEqual(
                    ideal.peak_heat_flux_altitude,
                    ae.peak_heat_flux_altitude,
                    delta=60.0,
                )
                self.assertAlmostEqual(
                    ideal.heat_load / ae.heat_load(ideal.altitude), 1.0, delta=2e-6
                )
                self.assertAlmostEqual(
                    ideal.speed / ae.speed(ideal.altitude), 1.0, delta=2e-6
                )

    def test_velocity_altitude_curve_along_the_whole_path(self) -> None:
        """V(h) from the integrator equals the closed form at every grid point."""
        gam = math.radians(-30.0)
        ae = AllenEggersEntry(300.0, gam, 7500.0, 4.7, self.ATM, entry_altitude=160e3)
        run = simulate_entry(
            EntryVehicle(300.0, 4.7),
            altitude=160e3,
            speed=7500.0,
            flight_path_angle=gam,
            atmosphere=self.ATM,
            terminal_altitude=20e3,
            max_step=1.0,
            idealised=True,
            record=True,
        )
        worst = max(abs(v / ae.speed(h) - 1.0) for _, h, v, _, _ in run.trajectory)
        self.assertLess(worst, 2e-6)  # measured ~1e-7; RK4 on smooth exp(-x)
        # Flight-path angle is exactly constant when gravity and curvature are off.
        self.assertLess(max(abs(g - gam) for _, _, _, g, _ in run.trajectory), 1e-15)

    def test_error_decreases_with_the_step(self) -> None:
        """Halving the step cuts the heat-load error by about 16 (RK4, order 4).

        Measured heat-load errors for resolution 2, 1, 0.5, 0.25, 0.125:
        1.6e-3, 1.1e-4, 7.4e-6, 4.8e-7, 3.0e-8 (ratios 15, 15, 15.5, 15.7). The
        peak load error is not monotone in the step (the parabola vertex
        depends on the grid phase) but is below 1e-4 from resolution 0.5 on.
        """
        gam = math.radians(-30.0)
        ae = AllenEggersEntry(300.0, gam, 7500.0, 4.7, self.ATM, entry_altitude=160e3)
        heat_errors = []
        peak_errors = []
        for resolution in (1.0, 0.5, 0.25, 0.125):
            run = simulate_entry(
                EntryVehicle(300.0, 4.7),
                altitude=160e3,
                speed=7500.0,
                flight_path_angle=gam,
                atmosphere=self.ATM,
                terminal_altitude=20e3,
                max_step=100.0,
                resolution=resolution,
                idealised=True,
            )
            heat_errors.append(abs(run.heat_load / ae.heat_load(run.altitude) - 1.0))
            peak_errors.append(
                abs(
                    run.peak_deceleration_g * STANDARD_GRAVITY / ae.peak_deceleration
                    - 1.0
                )
            )
        for coarse, fine in zip(heat_errors, heat_errors[1:], strict=False):
            self.assertGreater(coarse / fine, 12.0)
        self.assertTrue(all(e < 1e-4 for e in peak_errors[1:]))
        self.assertLess(peak_errors[-1], peak_errors[0] / 50.0)

    def test_gravity_and_curvature_effect_is_the_neglected_term(self) -> None:
        """With gravity on, peaks move by a few percent, as the speed gain predicts.

        Gravity adds ``g sin|gamma| t`` to the speed over the descent to the
        peak (about 27 s at 7.5 km/s and 30 deg), i.e. 1 to 2 percent of the
        speed and 2 to 4 percent of ``V^2``; the peak load and heat flux move
        by 5 to 7 percent in the measured cases. The tolerance of 10 percent is
        the justified envelope of that neglected term, not a fit.
        """
        for beta, radius, speed, angle in (
            (300.0, 4.7, 7500.0, -20.0),
            (300.0, 4.7, 7500.0, -45.0),
            (300.0, 4.7, 7500.0, -90.0),
        ):
            with self.subTest(angle=angle):
                ae, ideal, full = self._pair(beta, radius, speed, angle)
                load = full.peak_deceleration_g * STANDARD_GRAVITY
                self.assertAlmostEqual(load / ae.peak_deceleration, 1.0, delta=0.10)
                self.assertAlmostEqual(
                    full.peak_heat_flux / ae.peak_heat_flux, 1.0, delta=0.10
                )
                self.assertAlmostEqual(
                    full.heat_load / ae.heat_load(full.altitude), 1.0, delta=0.08
                )
                # Gravity speeds the vehicle up: every peak is larger, never smaller.
                self.assertGreater(full.peak_deceleration_g, ideal.peak_deceleration_g)
                self.assertGreater(full.peak_heat_flux, ideal.peak_heat_flux)
                self.assertGreater(full.heat_load, ideal.heat_load)

    def test_standard_atmosphere_is_close_in_order_of_magnitude(self) -> None:
        """US76 against the exponential closed form: same order, not equal.

        The exponential atmosphere (rho_0 = 1.225, H = 7200 m) is not a fit to
        the 1976 model, whose local scale height varies between about 5.5 and
        8 km over 40 to 90 km. Measured differences: 11 to 13 percent in peak
        load and 1 to 2 km in the peak altitude.
        """
        gam = math.radians(-30.0)
        ae = AllenEggersEntry(300.0, gam, 7800.0, 4.7, self.ATM, entry_altitude=120e3)
        run = simulate_entry(
            EntryVehicle(300.0, 4.7),
            altitude=120e3,
            speed=7800.0,
            flight_path_angle=gam,
        )
        self.assertEqual(run.status, "landed")
        load = run.peak_deceleration_g * STANDARD_GRAVITY
        self.assertAlmostEqual(load / ae.peak_deceleration, 1.0, delta=0.25)
        self.assertAlmostEqual(
            run.peak_deceleration_altitude, ae.peak_deceleration_altitude, delta=3e3
        )

    def test_kepler_invariants_without_drag(self) -> None:
        """With a vanishing density, energy and angular momentum are conserved.

        The exact flow conserves ``E = V^2/2 - mu/r`` and ``H = r V cos(gamma)``;
        the RK4 drift measures the integration error (measured below 1e-9
        relative over the 10 min descent).
        """
        run = simulate_entry(
            EntryVehicle(300.0, 4.7),
            altitude=120e3,
            speed=7000.0,
            flight_path_angle=math.radians(-4.0),
            density_scale=1e-12,
            terminal_altitude=10e3,
            max_step=1.0,
            resolution=1.0,
            record=True,
        )
        self.assertEqual(run.status, "landed")

        def invariants(h: float, v: float, g: float) -> tuple[float, float]:
            r = EARTH_RADIUS + h
            return 0.5 * v * v - EARTH_MU / r, r * v * math.cos(g)

        e0, h0 = invariants(*run.trajectory[0][1:4])
        worst_e = worst_h = 0.0
        for _, h, v, g, _ in run.trajectory:
            e, mom = invariants(h, v, g)
            worst_e = max(worst_e, abs(e / e0 - 1.0))
            worst_h = max(worst_h, abs(mom / h0 - 1.0))
        self.assertLess(worst_e, 1e-8)
        self.assertLess(worst_h, 1e-8)
        # The integrated heat load of a vacuum-like run is negligible.
        self.assertLess(run.heat_load, 1e5)  # vs about 1e8 J/m^2 with the atmosphere


class ModelIdentityTests(unittest.TestCase):
    """Exact identities of the equations of motion in the idealised mode."""

    ATM = ExponentialAtmosphere()

    def _run(self, vehicle: EntryVehicle, **overrides):  # type: ignore[no-untyped-def]
        """Idealised run in the exponential atmosphere with optional overrides."""
        kwargs = {
            "altitude": 160e3,
            "speed": 7500.0,
            "flight_path_angle": math.radians(-40.0),
            "atmosphere": self.ATM,
            "terminal_altitude": 20e3,
            "max_step": 1.0,
            "idealised": True,
        }
        kwargs.update(overrides)
        return simulate_entry(vehicle, **kwargs)  # type: ignore[arg-type]

    def test_density_scale_shifts_the_altitude_not_the_load(self) -> None:
        """Scaling the density by k shifts the altitude by H ln k, not the load."""
        base = self._run(EntryVehicle(300.0, 4.7))
        scaled = self._run(EntryVehicle(300.0, 4.7), density_scale=2.0)
        shift = self.ATM.scale_height * math.log(2.0)
        self.assertAlmostEqual(
            scaled.peak_deceleration_g,
            base.peak_deceleration_g,
            delta=2e-4 * base.peak_deceleration_g,
        )
        self.assertAlmostEqual(
            scaled.peak_deceleration_altitude - base.peak_deceleration_altitude,
            shift,
            delta=40.0,
        )
        # The scaled density is also the heating density, so the whole
        # trajectory and the heat flux are those of the base case, shifted.
        self.assertAlmostEqual(
            scaled.peak_heat_flux / base.peak_heat_flux, 1.0, delta=5e-4
        )
        self.assertAlmostEqual(
            scaled.peak_heat_flux_altitude - base.peak_heat_flux_altitude,
            shift,
            delta=40.0,
        )

    def test_peak_load_is_independent_of_the_ballistic_coefficient(self) -> None:
        """Allen-Eggers: a_max does not depend on beta; its altitude does."""
        light = self._run(EntryVehicle(150.0, 4.7))
        heavy = self._run(EntryVehicle(600.0, 4.7))
        self.assertAlmostEqual(
            heavy.peak_deceleration_g,
            light.peak_deceleration_g,
            delta=3e-4 * light.peak_deceleration_g,
        )
        # Altitude difference H ln(beta ratio) = H ln 4.
        self.assertAlmostEqual(
            light.peak_deceleration_altitude - heavy.peak_deceleration_altitude,
            self.ATM.scale_height * math.log(4.0),
            delta=60.0,
        )

    def test_heating_scale_is_exactly_linear(self) -> None:
        """The heating scale multiplies q and Q and leaves the trajectory alone."""
        base = self._run(EntryVehicle(300.0, 4.7))
        hot = self._run(EntryVehicle(300.0, 4.7), heating_scale=1.5)
        self.assertEqual(hot.altitude, base.altitude)
        self.assertEqual(hot.time, base.time)
        self.assertAlmostEqual(hot.peak_heat_flux / base.peak_heat_flux, 1.5, places=12)
        self.assertAlmostEqual(hot.heat_load / base.heat_load, 1.5, places=12)
        self.assertEqual(hot.peak_deceleration_g, base.peak_deceleration_g)


class HeatShieldTests(unittest.TestCase):
    """Tests of ``probstat.aerospace.entry``: HeatShieldTests."""

    def test_recession_and_insulation_closed_forms(self) -> None:
        """Recession f Q / (rho H_eff); insulation 2 sqrt(alpha t) erfc^-1(theta)."""
        theta = math.erfc(1.0)  # erfc^-1(theta) = 1 exactly
        t_i, t_s = 250.0, 2500.0
        shield = HeatShield(
            500.0, 2.5e7, 1.2, 2.7e-7, t_s, t_i + theta * (t_s - t_i), t_i, 0.03
        )
        heat_load, peak = 1.0e8, 2.0e6
        sizing = heat_shield_sizing(shield, heat_load, peak)
        self.assertAlmostEqual(sizing.heating_time, 50.0, places=12)
        self.assertAlmostEqual(
            sizing.recession, 1.2 * 1.0e8 / (500.0 * 2.5e7), delta=1e-15
        )
        self.assertAlmostEqual(
            sizing.insulation_thickness,
            2.0 * math.sqrt(2.7e-7 * 50.0),
            delta=1e-12,
        )
        self.assertAlmostEqual(
            sizing.required_thickness,
            sizing.recession + sizing.insulation_thickness,
            delta=1e-15,
        )
        self.assertAlmostEqual(
            sizing.margin, 0.03 - sizing.required_thickness, delta=1e-15
        )

    def test_insulation_satisfies_the_conduction_solution(self) -> None:
        """At the computed depth the semi-infinite solution hits the bond limit."""
        shield = _shield()
        sizing = heat_shield_sizing(shield, 1.2e8, 2.0e6)
        x = sizing.insulation_thickness
        t_h = sizing.heating_time
        theta = math.erfc(x / (2.0 * math.sqrt(2.7e-7 * t_h)))
        bond = 250.0 + (2500.0 - 250.0) * theta
        self.assertAlmostEqual(bond, 450.0, delta=1e-8)

    def test_override_and_zero_load(self) -> None:
        """The heat of ablation can be overridden; no heat means no thickness."""
        shield = _shield()
        base = heat_shield_sizing(shield, 1e8, 2e6)
        weak = heat_shield_sizing(shield, 1e8, 2e6, heat_of_ablation=1.25e7)
        self.assertAlmostEqual(weak.recession / base.recession, 2.0, places=12)
        self.assertEqual(weak.insulation_thickness, base.insulation_thickness)
        zero = heat_shield_sizing(shield, 0.0, 0.0)
        self.assertEqual(zero.required_thickness, 0.0)
        self.assertEqual(zero.margin, shield.design_thickness)

    def test_validation(self) -> None:
        """Unordered temperatures, bad factors and bad inputs are rejected."""
        with self.assertRaises(ValueError):
            HeatShield(500.0, 2.5e7, 1.2, 2.7e-7, 2500.0, 300.0, 400.0, 0.03)
        with self.assertRaises(ValueError):
            HeatShield(500.0, 2.5e7, 0.9, 2.7e-7, 2500.0, 450.0, 250.0, 0.03)
        with self.assertRaises(ValueError):
            HeatShield(500.0, 0.0, 1.2, 2.7e-7, 2500.0, 450.0, 250.0, 0.03)
        with self.assertRaises(TypeError):
            heat_shield_sizing("shield", 1e8, 2e6)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            heat_shield_sizing(_shield(), 1e8, 0.0)
        with self.assertRaises(ValueError):
            heat_shield_sizing(_shield(), -1.0, 1.0)


class SimulateEntryTests(unittest.TestCase):
    """Tests of ``probstat.aerospace.entry``: SimulateEntryTests."""

    def test_landing_and_outputs(self) -> None:
        """A lunar-return entry lands at 10 km with consistent outputs."""
        run = simulate_entry(
            _VEHICLE,
            altitude=120e3,
            speed=11000.0,
            flight_path_angle=math.radians(-9.0),
            record=True,
        )
        self.assertEqual(run.status, "landed")
        self.assertAlmostEqual(run.altitude, 10e3, delta=1e-3)
        self.assertTrue(run.peaks_interior)
        self.assertEqual(len(run.trajectory), run.steps + 1)
        self.assertAlmostEqual(run.downrange, EARTH_RADIUS * run.downrange_angle)
        # Measured: 20.4 g peak, peak heating 3 km above the peak load, speed at
        # 10 km near the (low-altitude) terminal value for beta = 320 kg/m^2.
        self.assertGreater(run.peak_deceleration_g, 10.0)
        self.assertLess(run.peak_deceleration_g, 30.0)
        self.assertGreater(run.peak_heat_flux_altitude, run.peak_deceleration_altitude)
        self.assertLess(run.speed, 300.0)
        # The peak altitude lies on the recorded path.
        heights = [p[1] for p in run.trajectory]
        self.assertGreater(run.peak_deceleration_altitude, min(heights))
        self.assertLess(run.peak_deceleration_altitude, max(heights))
        # The trajectory descends monotonically while the vehicle is steep.
        self.assertLess(run.trajectory[10][1], run.trajectory[0][1])

    def test_skip_out_is_reported_not_raised(self) -> None:
        """At 11 km/s a 5 degree entry with lift up skips back out."""
        run = simulate_entry(
            _VEHICLE,
            altitude=120e3,
            speed=11000.0,
            flight_path_angle=math.radians(-5.0),
        )
        self.assertEqual(run.status, "skip_out")
        self.assertGreater(run.altitude, 120e3)
        self.assertIn("above the entry altitude", run.stop_reason)
        # A ballistic vehicle (no lift) at the same angle is captured.
        ballistic = simulate_entry(
            EntryVehicle.from_mass(5000.0, 1.30, 12.0, 4.7, 0.0),
            altitude=120e3,
            speed=11000.0,
            flight_path_angle=math.radians(-7.0),
        )
        self.assertEqual(ballistic.status, "landed")

    def test_timeout_and_validation(self) -> None:
        """A short time limit stops the run; invalid arguments raise."""
        run = simulate_entry(
            _VEHICLE,
            altitude=120e3,
            speed=7800.0,
            flight_path_angle=math.radians(-5.0),
            max_time=50.0,
        )
        self.assertEqual(run.status, "timeout")
        self.assertLess(run.altitude, 120e3)
        bad = {"altitude": 120e3, "speed": 7800.0, "flight_path_angle": -0.1}
        with self.assertRaises(ValueError):
            simulate_entry(_VEHICLE, **{**bad, "flight_path_angle": 0.1})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_entry(_VEHICLE, **{**bad, "speed": 0.0})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_entry(_VEHICLE, **{**bad, "altitude": 5e3})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_entry(_VEHICLE, density_scale=-1.0, **bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_entry(_VEHICLE, idealised=True, **bad)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            simulate_entry("capsule", **bad)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            simulate_entry(_BALLISTIC, atmosphere="mars", **bad)  # type: ignore[arg-type]

    def test_numerical_failure_is_a_status(self) -> None:
        """A step far too coarse for a very light vehicle fails numerically."""
        run = simulate_entry(
            EntryVehicle(0.01, 4.7),
            altitude=120e3,
            speed=7800.0,
            flight_path_angle=math.radians(-30.0),
            max_step=50.0,
            resolution=50.0,
        )
        self.assertEqual(run.status, "numerical_failure")

    def test_lift_extends_the_range_and_lowers_the_load(self) -> None:
        """At fixed entry, lift up lowers the peak load and lengthens the range."""
        results = [
            simulate_entry(
                EntryVehicle(320.0, 4.7, ld),
                altitude=120e3,
                speed=7800.0,
                flight_path_angle=math.radians(-6.0),
            )
            for ld in (0.0, 0.15, 0.3, 0.45)
        ]
        loads = [r.peak_deceleration_g for r in results]
        ranges = [r.downrange for r in results]
        self.assertTrue(all(a > b for a, b in zip(loads, loads[1:], strict=False)))
        self.assertTrue(all(a < b for a, b in zip(ranges, ranges[1:], strict=False)))


class EntryModelTests(unittest.TestCase):
    """Tests of ``probstat.aerospace.entry``: EntryModelTests."""

    def test_nominal_matches_the_direct_simulation(self) -> None:
        """The model's nominal output equals simulate_entry plus the sizing."""
        model = _model()
        out = model.evaluate(model.nominal())
        run = simulate_entry(
            EntryVehicle(320.0, 4.7, 0.3),
            altitude=120e3,
            speed=11000.0,
            flight_path_angle=math.radians(-7.9),
            **_FAST,
        )
        self.assertAlmostEqual(out["peak_deceleration_g"], run.peak_deceleration_g)
        self.assertAlmostEqual(out["heat_load"], run.heat_load)
        sizing = heat_shield_sizing(_shield(), run.heat_load, run.peak_heat_flux)
        self.assertAlmostEqual(out["required_thickness"], sizing.required_thickness)
        self.assertAlmostEqual(
            out["thickness_margin"], _DESIGN - out["required_thickness"]
        )
        self.assertAlmostEqual(out["downrange"], run.downrange)

    def test_construction_and_validation(self) -> None:
        """No random input, bad types and bad vehicles are rejected."""
        with self.assertRaises(ValueError):
            entry_model(
                _VEHICLE, _shield(), entry_speed=11000.0, flight_path_angle=-0.13
            )
        with self.assertRaises(TypeError):
            entry_model(
                "v",  # type: ignore[arg-type]
                _shield(),
                entry_speed=Normal(11000.0, 1.0),
                flight_path_angle=-0.13,
            )
        with self.assertRaises(TypeError):
            entry_model(
                _VEHICLE,
                _shield(),
                entry_speed=True,
                flight_path_angle=Normal(-0.13, 0.01),
            )
        model = _model()
        self.assertEqual(
            model.inputs.names,
            (
                "flight_path_angle",
                "entry_speed",
                "ballistic_coefficient",
                "nose_radius",
                "density_scale",
                "heating_scale",
                "heat_of_ablation",
            ),
        )

    def test_failure_policy_handles_skip_out(self) -> None:
        """A dispersion straddling the skip-out boundary fails runs per policy."""
        # Skip-out boundary at 11 km/s is about -7.1 deg: mean -7.3, sd 0.3.
        wide = _model(gamma_deg=-7.3, sigma_gamma_deg=0.3)
        point = list(wide.nominal())
        point[0] = math.radians(-6.0)
        with self.assertRaises(RuntimeError):
            wide.evaluate(point)
        skipping = _model(gamma_deg=-7.3, sigma_gamma_deg=0.3, on_failure="skip")
        with self.assertRaises(Exception) as caught:  # noqa: B017
            skipping.evaluate(point)
        self.assertIn("skip_out", str(caught.exception.__cause__))
        self.assertEqual(skipping.failed_evaluations, 1)
        # Over a sample, a sizeable fraction fails; the rest succeed.
        failures = 0
        draws = skipping.inputs.sample(60, seed=4)
        for x in draws:
            try:
                skipping.evaluate(x)
            except Exception:  # noqa: BLE001
                failures += 1
        self.assertGreater(failures, 5)
        self.assertLess(failures, 55)

    def test_sobol_ranking_measured(self) -> None:
        """Sobol ranking of peak load, heat flux and heat load (measured, N = 32).

        Measured at N = 64 (not assumed): with all seven inputs uncertain the
        Sutton-Graves scale (10 percent) dominates the peak heat flux and the
        heat load (first-order index about 0.8), the ballistic coefficient is
        second (about 0.2) and the flight-path angle carries only 0.03; the
        flight-path angle dominates the peak load factor (index above 0.9).
        With the correlation scale and the heat of ablation fixed, the
        ballistic coefficient (7.8 percent) ranks first for heat flux and heat
        load (about 0.5), then the entry speed. The flight-path angle matters
        little for the heat quantities at a 0.12 degree dispersion because the
        peak heat flux scales as ``sqrt(sin|gamma|)`` and the load as
        ``1 / sqrt(sin|gamma|)``: weak powers of a small relative change.
        """
        names = ("peak_deceleration_g", "peak_heat_flux", "heat_load")
        full = sobol_analysis_all(_model(), 32, quantities=names, seed=1)

        def first(result, quantity):  # type: ignore[no-untyped-def]
            sobol = result[quantity]
            return dict(
                zip(
                    sobol.input_names,
                    (i.estimate for i in sobol.first_order),
                    strict=True,
                )
            )

        g = first(full, "peak_deceleration_g")
        q = first(full, "peak_heat_flux")
        load = first(full, "heat_load")
        self.assertEqual(max(g, key=lambda k: g[k]), "flight_path_angle")
        self.assertGreater(g["flight_path_angle"], 0.8)
        self.assertEqual(max(q, key=lambda k: q[k]), "heating_scale")
        self.assertGreater(q["heating_scale"], 0.55)
        self.assertEqual(max(load, key=lambda k: load[k]), "heating_scale")
        self.assertGreater(load["heating_scale"], 0.45)
        self.assertLess(q["heat_of_ablation"], 0.1)  # irrelevant to the flight
        self.assertLess(q["flight_path_angle"], 0.25)
        # Physical inputs only.
        physical = sobol_analysis_all(
            _model(fix_scales=True), 32, quantities=names, seed=1
        )
        qp = first(physical, "peak_heat_flux")
        lp = first(physical, "heat_load")
        self.assertEqual(max(qp, key=lambda k: qp[k]), "ballistic_coefficient")
        self.assertEqual(max(lp, key=lambda k: lp[k]), "ballistic_coefficient")
        self.assertGreater(qp["ballistic_coefficient"], 0.3)
        self.assertGreater(qp["ballistic_coefficient"], qp["flight_path_angle"])

    def test_exceedance_probability_form_subset_and_mc(self) -> None:
        """P(required thickness > design) by FORM, subset simulation and MC.

        Reference (slow tier, 3000 MC runs of the same model): 0.0363 with
        standard error 0.0034; FORM gives 0.0350 (see ``SlowStudies``).
        The fast bands: 300-run MC (standard error about 0.01 at p = 0.035),
        FORM within 3 standard errors, subset simulation (100 samples per
        level) within 3 combined standard errors.
        """
        state = ModelLimitState(_model(), "required_thickness", _DESIGN, "above")
        mc = model_monte_carlo(state, 300, seed=3)
        self.assertGreater(mc.probability, 0.01)
        self.assertLess(mc.probability, 0.1)
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertLess(abs(form.probability - mc.probability), 3.0 * mc.standard_error)
        subset = subset_simulation(state, samples_per_level=100, seed=5)
        self.assertTrue(subset.converged)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)
        # The model-form scales carry the failure; the flight-path angle does not.
        weights = form.importance_factors
        self.assertGreater(weights["heat_of_ablation"] + weights["heating_scale"], 0.6)
        self.assertLess(weights["flight_path_angle"], 0.15)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_steep_versus_shallow_trade_off_is_monotone(self) -> None:
        """Steeper entries raise the peak load and lower the heat load, monotonically.

        Ballistic capsule at 7.8 km/s from 8 to 60 degrees: peak load grows
        like ``sin|gamma|`` (log-log slope measured 0.92 to 1.04, closed form
        1) and the heat load falls like ``sin^-1/2`` (slope measured -0.46 to
        -0.51, closed form -0.5); the low-angle cases deviate because gravity
        and curvature are no longer negligible.
        """
        angles = (-8.0, -10.0, -15.0, -20.0, -30.0, -45.0, -60.0)
        runs = [
            simulate_entry(
                EntryVehicle(300.0, 4.7),
                altitude=120e3,
                speed=7800.0,
                flight_path_angle=math.radians(a),
                **_FAST,
            )
            for a in angles
        ]
        self.assertTrue(all(r.status == "landed" for r in runs))
        loads = [r.peak_deceleration_g for r in runs]
        heats = [r.heat_load for r in runs]
        fluxes = [r.peak_heat_flux for r in runs]
        self.assertTrue(all(a < b for a, b in zip(loads, loads[1:], strict=False)))
        self.assertTrue(all(a > b for a, b in zip(heats, heats[1:], strict=False)))
        self.assertTrue(all(a < b for a, b in zip(fluxes, fluxes[1:], strict=False)))
        sines = [math.sin(math.radians(-a)) for a in angles]
        for i in range(len(angles) - 1):
            ratio = math.log(sines[i + 1] / sines[i])
            slope_g = math.log(loads[i + 1] / loads[i]) / ratio
            slope_q = math.log(heats[i + 1] / heats[i]) / ratio
            self.assertAlmostEqual(slope_g, 1.0, delta=0.12)
            self.assertAlmostEqual(slope_q, -0.5, delta=0.1)

    def test_lifting_corridor_is_bracketed_by_bisection(self) -> None:
        """Corridor bounds at 11 km/s: skip-out and 15 g, found by bisection."""
        corridor = entry_corridor(
            _VEHICLE, entry_speed=11000.0, g_limit=15.0, tolerance=1e-4, **_FAST
        )
        self.assertLess(corridor.steep_angle, corridor.shallow_angle)
        self.assertGreater(corridor.width, 0.0)
        # Measured: about -7.1 deg (skip-out) to -8.05 deg (15 g).
        self.assertAlmostEqual(math.degrees(corridor.shallow_angle), -7.08, delta=0.1)
        self.assertAlmostEqual(math.degrees(corridor.steep_angle), -8.05, delta=0.1)
        self.assertLessEqual(corridor.peak_g_at_steep, 15.0)
        self.assertAlmostEqual(corridor.peak_g_at_steep, 15.0, delta=0.05)
        # Just shallower than the boundary skips, at the boundary it lands.
        common = {
            "altitude": 120e3,
            "speed": 11000.0,
            **_FAST,
        }
        inside = simulate_entry(
            _VEHICLE, flight_path_angle=corridor.shallow_angle, **common
        )
        outside = simulate_entry(
            _VEHICLE, flight_path_angle=corridor.shallow_angle + 3e-4, **common
        )
        self.assertEqual(inside.status, "landed")
        self.assertEqual(outside.status, "skip_out")
        # Just steeper than the load bound exceeds the limit.
        over = simulate_entry(
            _VEHICLE, flight_path_angle=corridor.steep_angle - 3e-4, **common
        )
        self.assertGreater(over.peak_deceleration_g, 15.0)
        # The trade-off across the corridor: shallow edge = more heat, less load.
        self.assertGreater(corridor.heat_load_at_shallow, corridor.heat_load_at_steep)
        self.assertLess(corridor.peak_g_at_shallow, corridor.peak_g_at_steep)
        self.assertLess(corridor.evaluations, 60)

    def test_lift_widens_the_corridor_and_shifts_it(self) -> None:
        """The ballistic corridor lies at shallower angles and has a different width.

        Measured at 11 km/s, 15 g: ballistic -5.3 to -6.4 deg, L/D = 0.3
        -7.1 to -8.05 deg. Constant lift-up pushes the skip boundary steeper.
        """
        ballistic = entry_corridor(
            _BALLISTIC, entry_speed=11000.0, g_limit=15.0, tolerance=1e-4, **_FAST
        )
        lifting = entry_corridor(
            _VEHICLE, entry_speed=11000.0, g_limit=15.0, tolerance=1e-4, **_FAST
        )
        self.assertGreater(ballistic.shallow_angle, lifting.shallow_angle)
        self.assertGreater(ballistic.steep_angle, lifting.steep_angle)
        self.assertAlmostEqual(math.degrees(ballistic.shallow_angle), -5.30, delta=0.1)

    def test_corridor_validation(self) -> None:
        """Unordered or non-bracketing brackets raise instead of lying."""
        base = {"entry_speed": 11000.0, "g_limit": 15.0, **_FAST}
        with self.assertRaises(ValueError):
            entry_corridor(_VEHICLE, shallow_bracket=-0.2, steep_bracket=-0.1, **base)
        with self.assertRaises(ValueError):
            # -12 degrees captures: no skip-out boundary in the bracket.
            entry_corridor(_VEHICLE, shallow_bracket=math.radians(-12.0), **base)
        with self.assertRaises(ValueError):
            # -9 degrees is still below 15 g? It is above (20 g); use -7.5.
            entry_corridor(
                _VEHICLE,
                steep_bracket=math.radians(-7.5),
                shallow_bracket=math.radians(-3.0),
                **base,
            )
        with self.assertRaises(TypeError):
            entry_corridor("v", **base)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            entry_corridor(_VEHICLE, **{**base, "g_limit": 0.0})


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudies(unittest.TestCase):
    """Larger studies, run only with PROBSTAT_SLOW=1."""

    def test_exceedance_probability_reference(self) -> None:
        """3000-run MC reference with FORM and subset simulation (500 per level)."""
        state = ModelLimitState(_model(), "required_thickness", _DESIGN, "above")
        mc = model_monte_carlo(state, 3000, seed=11)
        form = form_hlrf(state)
        subset = subset_simulation(state, samples_per_level=500, seed=7)
        self.assertLess(abs(form.probability - mc.probability), 3.0 * mc.standard_error)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)

    def test_sobol_ranking_stable_with_more_samples(self) -> None:
        """At N = 128 the rankings of the fast test persist."""
        names = ("peak_deceleration_g", "peak_heat_flux", "heat_load")
        result = sobol_analysis_all(_model(), 128, quantities=names, seed=9)
        for quantity, leader in (
            ("peak_deceleration_g", "flight_path_angle"),
            ("peak_heat_flux", "heating_scale"),
            ("heat_load", "heating_scale"),
        ):
            sobol = result[quantity]
            firsts = {
                n: i.estimate
                for n, i in zip(sobol.input_names, sobol.first_order, strict=True)
            }
            self.assertEqual(max(firsts, key=lambda k: firsts[k]), leader)


if __name__ == "__main__":
    unittest.main()
