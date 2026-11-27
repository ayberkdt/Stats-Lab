"""Tests for regenerative cooling and wall-temperature uncertainty.

All engine numbers are synthetic and of order-of-magnitude realism for a
regeneratively cooled kerosene-class chamber (about 6 MPa, 100 kN class). They
are not data from any real engine.
"""

from __future__ import annotations

import math
import os
import statistics
import unittest
import warnings

from probstat.aerospace.liquid_engine import area_ratio_from_mach
from probstat.aerospace.nozzle_cooling import (
    ChannelGeometry,
    CoolantProperties,
    GasProperties,
    NozzleContour,
    WallCorrection,
    adiabatic_wall_temperature,
    bartz_base_coefficient,
    bartz_heat_transfer_coefficient,
    bartz_property_correction,
    conical_contour,
    coolant_side_coefficient,
    local_mach,
    nozzle_cooling_model,
    solve_regenerative_cooling,
    solve_wall_network,
    station_profile,
)
from probstat.distributions import LogNormal, Normal, TruncatedNormal
from probstat.propagation import make_design
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis
from probstat.surrogates import fit_gaussian_process_to_model, validate_gp_emulator

GAMMA = 1.2
RT = 0.05


def _area_ratio_closed_form(gamma: float, mach: float) -> float:
    """Independent isentropic ``A/A*`` for a gas with ratio ``gamma``."""
    z = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / mach) * (2.0 / (gamma + 1.0) * z) ** exponent


def _contour() -> NozzleContour:
    """Return the synthetic conical contour (r_t = 50 mm, epsilon = 15)."""
    return conical_contour(
        throat_radius=RT,
        chamber_radius=0.09,
        cylinder_length=0.20,
        expansion_ratio=15.0,
    )


def _gas() -> GasProperties:
    """Return synthetic hot-gas properties."""
    return GasProperties(GAMMA, 23.0, 3400.0, 8e-5)


def _coolant() -> CoolantProperties:
    """Return synthetic kerosene-class coolant properties."""
    return CoolantProperties(2100.0, 0.12, 3.5e-4, 700.0)


def _channel() -> ChannelGeometry:
    """Return the synthetic channel geometry."""
    return ChannelGeometry(100, 1.2e-3, 2.0e-3, 3.0)


def _operating(**overrides: float) -> dict[str, float]:
    """Return the nominal synthetic operating point."""
    point = {
        "chamber_pressure": 6.0e6,
        "characteristic_velocity": 1750.0,
        "coolant_mass_flow": 11.0,
        "coolant_inlet_temperature": 290.0,
        "wall_conductivity": 320.0,
        "wall_thickness": 1.0e-3,
    }
    point.update(overrides)
    return point


def _solve(stations: int = 41, **kwargs: object):  # noqa: ANN202
    """Solve the nominal case with optional overrides."""
    flow = kwargs.pop("flow", "counter")
    operating = _operating(**{k: float(v) for k, v in kwargs.items()})  # type: ignore[arg-type]
    return solve_regenerative_cooling(
        station_profile(_contour(), GAMMA, stations),
        gas=_gas(),
        coolant=_coolant(),
        channel=_channel(),
        flow=flow,  # type: ignore[arg-type]
        **operating,
    )


class ContourAndMachTests(unittest.TestCase):
    """Geometry, area ratio and the two Mach branches."""

    def test_contour_geometry(self) -> None:
        """Throat, chamber and exit radii; one sharp corner only."""
        c = _contour()
        self.assertAlmostEqual(c.throat_radius, RT, places=15)
        self.assertAlmostEqual(c.radius[0], 0.09, places=15)
        self.assertAlmostEqual(c.radius[-1], RT * math.sqrt(15.0), places=12)
        self.assertEqual(c.x[0], 0.0)
        slopes = [
            math.atan2(c.radius[i + 1] - c.radius[i], c.x[i + 1] - c.x[i])
            for i in range(len(c.x) - 1)
        ]
        jumps = [abs(b - a) for a, b in zip(slopes, slopes[1:], strict=False)]
        # the cylinder-cone corner is sharp (30 degrees); every other joint is
        # tangent up to the table spacing (< 0.2 degrees per sample)
        sharp = [j for j in jumps if j > 0.05]
        self.assertEqual(len(sharp), 1)
        self.assertAlmostEqual(sharp[0], math.radians(30.0), delta=2e-3)
        self.assertLess(max(j for j in jumps if j <= 0.05), math.radians(0.2))
        # arc radii: the curvature radius is the mean of the two arcs
        self.assertAlmostEqual(c.curvature_radius, 0.5 * (1.5 + 0.382) * RT)

    def test_contour_rejects_bad_tables(self) -> None:
        """A non convergent-divergent or non-monotone table is rejected."""
        with self.assertRaises(ValueError):
            NozzleContour((0, 1, 2, 3, 4), (1, 2, 3, 4, 5), 1.0)  # no throat
        with self.assertRaises(ValueError):
            NozzleContour((0, 1, 1, 3, 4), (3, 2, 1, 2, 3), 1.0)  # x repeated
        with self.assertRaises(ValueError):
            NozzleContour((0, 1, 2, 3, 4), (3, 1, 2, 1.5, 3), 1.0)  # two throats
        with self.assertRaises(ValueError):
            conical_contour(
                throat_radius=RT,
                chamber_radius=0.06,  # smaller than the upstream arc end
                cylinder_length=0.2,
                expansion_ratio=15.0,
            )
        with self.assertRaises(ValueError):
            _contour().radius_at(10.0)

    def test_stations_include_throat(self) -> None:
        """The throat is always a station; the count is as requested."""
        c = _contour()
        for n in (5, 21, 40, 41, 100):
            xs = c.stations(n)
            self.assertEqual(len(xs), n)
            self.assertIn(c.x_throat, xs)
            self.assertEqual((xs[0], xs[-1]), (c.x[0], c.x[-1]))
        with self.assertRaises(ValueError):
            c.stations(4)

    def test_mach_matches_independent_area_relation(self) -> None:
        """Both branches invert the closed-form A/A* at every station."""
        profile = station_profile(_contour(), GAMMA, 81)
        for ratio, mach, x in zip(
            profile.area_ratio, profile.mach, profile.x, strict=True
        ):
            if x < profile.contour.x_throat:
                self.assertLessEqual(mach, 1.0)
            elif x > profile.contour.x_throat:
                self.assertGreaterEqual(mach, 1.0)
            if mach != 1.0:
                self.assertAlmostEqual(
                    _area_ratio_closed_form(GAMMA, mach) / ratio, 1.0, delta=1e-9
                )
        self.assertEqual(profile.mach[profile.throat_index], 1.0)
        # chamber end: A/A_t = (0.09/0.05)^2 = 3.24, subsonic
        self.assertAlmostEqual(profile.area_ratio[0], 3.24, places=12)
        self.assertAlmostEqual(
            area_ratio_from_mach(GAMMA, profile.mach[0]), 3.24, delta=1e-8
        )
        self.assertLess(profile.mach[0], 0.25)

    def test_local_mach_errors(self) -> None:
        """Invalid arguments are rejected."""
        with self.assertRaises(ValueError):
            local_mach(GAMMA, 0.9, supersonic=True)
        with self.assertRaises(ValueError):
            local_mach(1.0, 2.0, supersonic=False)
        self.assertEqual(local_mach(GAMMA, 1.0, supersonic=False), 1.0)


class GasSideTests(unittest.TestCase):
    """Bartz, sigma and the recovery temperature."""

    def test_recovery_temperature_closed_forms(self) -> None:
        """T_aw = T_c at rest and for r = 1; hand value at M = 1."""
        gas = _gas()
        self.assertEqual(adiabatic_wall_temperature(gas, 0.0), 3400.0)
        unit = GasProperties(GAMMA, 23.0, 3400.0, 8e-5, recovery_factor=1.0)
        for mach in (0.3, 1.0, 2.5):
            self.assertAlmostEqual(adiabatic_wall_temperature(unit, mach), 3400.0)
        pr = 4 * GAMMA / (9 * GAMMA - 5)
        self.assertAlmostEqual(gas.prandtl_number, pr, places=15)
        r = pr ** (1.0 / 3.0)
        by_hand = 3400.0 * (1.0 + r * 0.1) / 1.1  # M = 1: Z = 1 + 0.1
        self.assertAlmostEqual(adiabatic_wall_temperature(gas, 1.0), by_hand, places=9)
        # monotone decrease with Mach for r < 1
        temps = [adiabatic_wall_temperature(gas, m) for m in (0.0, 0.5, 1.0, 2.0, 3.0)]
        self.assertTrue(all(b < a for a, b in zip(temps, temps[1:], strict=False)))

    def test_specific_heat(self) -> None:
        """The specific heat is gamma R / (gamma - 1) with R = 8314.46 / M."""
        expected = GAMMA / (GAMMA - 1.0) * 8314.462618 / 23.0
        self.assertAlmostEqual(_gas().specific_heat, expected, places=9)

    def test_sigma_limits_and_monotonicity(self) -> None:
        """Sigma is 1 at rest and T_w = T_c, falls with T_w and exceeds 1 when cold."""
        gas = _gas()
        self.assertAlmostEqual(bartz_property_correction(gas, 0.0, 3400.0), 1.0)
        values = [bartz_property_correction(gas, 1.0, t) for t in (300, 600, 900, 1500)]
        self.assertTrue(all(b < a for a, b in zip(values, values[1:], strict=False)))
        self.assertGreater(values[0], 1.0)
        # hand value at M = 1, T_w = 600 K, omega = 0.6
        z = 1.1
        by_hand = 1.0 / ((0.5 * 600.0 / 3400.0 * z + 0.5) ** 0.68 * z**0.12)
        self.assertAlmostEqual(
            bartz_property_correction(gas, 1.0, 600.0), by_hand, places=14
        )

    def test_bartz_reduces_to_pipe_correlation(self) -> None:
        """Without curvature and area factors Bartz is 0.026 (k/D) Re^0.8 Pr^0.4."""
        gas = _gas()
        pc, cstar, dt = 6.0e6, 1750.0, 0.1
        h = bartz_base_coefficient(
            gas=gas,
            chamber_pressure=pc,
            characteristic_velocity=cstar,
            throat_diameter=dt,
            curvature_radius=dt,  # (D_t / r_c)^0.1 = 1
            area_ratio=1.0,
        )
        g = pc / cstar  # throat mass flux, kg/(m^2 s)
        reynolds = g * dt / gas.viscosity
        conductivity = gas.viscosity * gas.specific_heat / gas.prandtl_number
        pipe = 0.026 * conductivity / dt * reynolds**0.8 * gas.prandtl_number**0.4
        self.assertAlmostEqual(h / pipe, 1.0, delta=1e-12)
        # order of magnitude of a 100 kN-class throat (synthetic): 1e4 W/m^2/K
        self.assertGreater(h, 5e3)
        self.assertLess(h, 5e4)

    def test_bartz_scaling_laws(self) -> None:
        """Scaling exponents 0.8, -0.8, -0.2 and -0.9 at fixed sigma."""
        gas = _gas()
        base = {
            "gas": gas,
            "chamber_pressure": 6.0e6,
            "characteristic_velocity": 1750.0,
            "throat_diameter": 0.1,
            "curvature_radius": 0.047,
            "area_ratio": 1.7,
            "mach": 0.6,
            "wall_temperature": 800.0,
        }

        def h(**change: float) -> float:
            """Bartz coefficient with the named arguments replaced."""
            return bartz_heat_transfer_coefficient(**{**base, **change})  # type: ignore[arg-type]

        h0 = h()

        def slope(value: float) -> float:
            """Log-log slope for a change by a factor 2 of one argument."""
            return math.log2(value / h0)

        self.assertAlmostEqual(slope(h(chamber_pressure=12.0e6)), 0.8, places=12)
        self.assertAlmostEqual(
            slope(h(characteristic_velocity=3500.0)), -0.8, places=12
        )
        # D_t at fixed D_t / r_c: exponent -0.2 (fixed r_c: -0.2 + 0.1 = -0.1)
        self.assertAlmostEqual(
            slope(h(throat_diameter=0.2, curvature_radius=0.094)), -0.2, places=12
        )
        self.assertAlmostEqual(slope(h(throat_diameter=0.2)), -0.1, places=12)
        self.assertAlmostEqual(slope(h(area_ratio=3.4)), -0.9, places=12)
        self.assertAlmostEqual(slope(h(multiplier=2.0)), 1.0, places=12)
        # sigma is independent of p_c, c*, D_t and A at fixed T_w and M
        self.assertAlmostEqual(
            h()
            / bartz_base_coefficient(
                **{
                    k: base[k]
                    for k in (
                        "gas",
                        "chamber_pressure",
                        "characteristic_velocity",
                        "throat_diameter",
                        "curvature_radius",
                        "area_ratio",
                    )
                }
            ),  # type: ignore[arg-type]
            bartz_property_correction(gas, 0.6, 800.0),
            places=12,
        )

    def test_gas_validation(self) -> None:
        """Invalid gas and coefficient arguments are rejected."""
        with self.assertRaises(ValueError):
            GasProperties(1.0, 23.0, 3400.0, 8e-5)
        with self.assertRaises(ValueError):
            GasProperties(GAMMA, 23.0, -3400.0, 8e-5)
        with self.assertRaises(ValueError):
            GasProperties(GAMMA, 23.0, 3400.0, 8e-5, recovery_factor=2.0)
        with self.assertRaises(TypeError):
            GasProperties(True, 23.0, 3400.0, 8e-5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bartz_property_correction(_gas(), -1.0, 600.0)
        with self.assertRaises(TypeError):
            adiabatic_wall_temperature("gas", 1.0)  # type: ignore[arg-type]


class CoolantSideTests(unittest.TestCase):
    """Dittus-Boelter coefficient and channel geometry."""

    def test_dittus_boelter_hand_value(self) -> None:
        """h_c from Re, Pr and D_h against the formula evaluated by hand."""
        channel, coolant = _channel(), _coolant()
        d_h = 2 * 1.2e-3 * 2.0e-3 / (1.2e-3 + 2.0e-3)
        self.assertAlmostEqual(channel.hydraulic_diameter, d_h, places=18)
        self.assertAlmostEqual(channel.flow_area, 100 * 1.2e-3 * 2.0e-3, places=18)
        h_eff, reynolds = coolant_side_coefficient(channel, coolant, 11.0)
        re = 11.0 / (100 * 1.2e-3 * 2.0e-3) * d_h / 3.5e-4
        pr = 2100.0 * 3.5e-4 / 0.12
        self.assertAlmostEqual(reynolds / re, 1.0, delta=1e-13)
        nu = 0.023 * re**0.8 * pr**0.4
        self.assertAlmostEqual(h_eff / (3.0 * nu * 0.12 / d_h), 1.0, delta=1e-13)
        # multiplier scales linearly; flow scales as mdot^0.8
        h2, _ = coolant_side_coefficient(channel, coolant, 11.0, multiplier=1.25)
        self.assertAlmostEqual(h2 / h_eff, 1.25, places=13)
        h3, _ = coolant_side_coefficient(channel, coolant, 22.0)
        self.assertAlmostEqual(math.log2(h3 / h_eff), 0.8, places=12)

    def test_validation(self) -> None:
        """Invalid geometry and coolant data are rejected."""
        with self.assertRaises(ValueError):
            ChannelGeometry(0, 1e-3, 1e-3)
        with self.assertRaises(TypeError):
            ChannelGeometry(2.5, 1e-3, 1e-3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ChannelGeometry(10, -1e-3, 1e-3)
        with self.assertRaises(ValueError):
            CoolantProperties(2100.0, 0.12, 3.5e-4, 700.0, heating_exponent=2.0)
        with self.assertRaises(ValueError):
            coolant_side_coefficient(_channel(), _coolant(), -1.0)


class StationNetworkTests(unittest.TestCase):
    """The series resistance network at one station."""

    def test_closed_form_hand_case(self) -> None:
        """Constant h_g: q = dT / (1/h_g + t/k + 1/h_c), checked by hand.

        T_aw = 3000 K, T_b = 400 K, h_g = 2e4, t/k = 1e-3 / 300, h_c = 5e4:
        sum of resistances = 5e-5 + 3.3333e-6 + 2e-5 = 7.33333e-5 m^2 K / W,
        q = 2600 / 7.33333e-5 = 3.54545e7 W/m^2, T_wg = 3000 - q 5e-5 =
        1227.27 K, T_wc = T_wg - q 3.3333e-6 = 1109.09 K.
        """
        sol = solve_wall_network(
            recovery_temperature=3000.0,
            coolant_temperature=400.0,
            gas_coefficient=2e4,
            wall_resistance=1e-3 / 300.0,
            coolant_coefficient=5e4,
        )
        self.assertAlmostEqual(sol.heat_flux, 3.5454545e7, delta=1.0)
        self.assertAlmostEqual(sol.gas_wall_temperature, 1227.2727, delta=1e-3)
        self.assertAlmostEqual(sol.coolant_wall_temperature, 1109.0909, delta=1e-3)
        q = sol.heat_flux
        self.assertAlmostEqual(q, 2e4 * (3000.0 - sol.gas_wall_temperature), delta=1e-4)
        self.assertAlmostEqual(
            q,
            300.0 / 1e-3 * (sol.gas_wall_temperature - sol.coolant_wall_temperature),
            delta=1e-4,
        )
        self.assertAlmostEqual(
            q, 5e4 * (sol.coolant_wall_temperature - 400.0), delta=1e-4
        )
        self.assertEqual(sol.iterations, 0)

    def test_sigma_network_matches_bisection_oracle(self) -> None:
        """With sigma(T_w) the Newton root equals a plain bisection root."""
        gas = _gas()
        mach, taw = 1.0, adiabatic_wall_temperature(gas, 1.0)
        z = 1.0 + 0.5 * (GAMMA - 1.0) * mach * mach
        base, r_wall, h_c, tb = 1.4e4, 3.3e-6, 1.6e5, 500.0
        sol = solve_wall_network(
            recovery_temperature=taw,
            coolant_temperature=tb,
            gas_coefficient=base,
            wall_resistance=r_wall,
            coolant_coefficient=h_c,
            correction=WallCorrection(3400.0, z, 0.6),
        )
        self.assertTrue(sol.converged)

        def residual(t: float) -> float:
            """Wall-temperature equation using the public sigma."""
            h = base * bartz_property_correction(gas, mach, t)
            return h * (taw - t) - (t - tb) / (r_wall + 1.0 / h_c)

        low, high = tb, taw
        for _ in range(200):
            mid = 0.5 * (low + high)
            if residual(mid) > 0.0:
                low = mid
            else:
                high = mid
        self.assertAlmostEqual(sol.gas_wall_temperature, 0.5 * (low + high), delta=1e-8)
        h_here = base * bartz_property_correction(gas, mach, sol.gas_wall_temperature)
        self.assertAlmostEqual(sol.gas_coefficient / h_here, 1.0, delta=1e-12)
        q = sol.heat_flux
        self.assertAlmostEqual(
            q / (h_here * (taw - sol.gas_wall_temperature)), 1.0, delta=1e-12
        )
        self.assertAlmostEqual(
            q / (h_c * (sol.coolant_wall_temperature - tb)), 1.0, delta=1e-8
        )
        # a hotter wall than a constant-coefficient network predicts at the
        # unit-sigma value: the property factor changes the answer
        plain = solve_wall_network(
            recovery_temperature=taw,
            coolant_temperature=tb,
            gas_coefficient=base,
            wall_resistance=r_wall,
            coolant_coefficient=h_c,
        )
        self.assertGreater(
            abs(plain.gas_wall_temperature - sol.gas_wall_temperature), 10.0
        )

    def test_failure_is_reported_not_hidden(self) -> None:
        """A cap of one iteration reports converged False; bad input raises."""
        kwargs = {
            "recovery_temperature": 3000.0,
            "coolant_temperature": 500.0,
            "gas_coefficient": 1.4e4,
            "wall_resistance": 3.3e-6,
            "coolant_coefficient": 1.6e5,
            "correction": WallCorrection(3400.0, 1.1, 0.6),
        }
        capped = solve_wall_network(**kwargs, guess=600.0, max_iterations=1)  # type: ignore[arg-type]
        self.assertFalse(capped.converged)
        with self.assertRaises(ValueError):
            solve_wall_network(**{**kwargs, "coolant_temperature": 3000.0})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            solve_wall_network(**{**kwargs, "wall_resistance": -1.0})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            solve_wall_network(**{**kwargs, "correction": 1.0})  # type: ignore[arg-type]


class MarchingTests(unittest.TestCase):
    """Energy conservation, the throat, direction and convergence."""

    def test_energy_conservation_both_directions(self) -> None:
        """Coolant gain = trapezoid of q' ds, and per segment."""
        for flow in ("counter", "co"):
            sol = _solve(41, flow=flow)
            self.assertTrue(sol.converged, sol.stop_reason)
            profile = station_profile(_contour(), GAMMA, 41)
            trapezoid = sum(
                0.5 * ds * (sol.linear_heat_rate[i] + sol.linear_heat_rate[i + 1])
                for i, ds in enumerate(profile.segment_length)
            )
            self.assertAlmostEqual(trapezoid / sol.total_heat_load, 1.0, delta=1e-11)
            self.assertAlmostEqual(
                sol.coolant_heat_gain / sol.total_heat_load, 1.0, delta=1e-9
            )
            mcp = 11.0 * 2100.0
            for i, ds in enumerate(profile.segment_length):
                gain = abs(sol.coolant_temperature[i + 1] - sol.coolant_temperature[i])
                seg = 0.5 * ds * (sol.linear_heat_rate[i] + sol.linear_heat_rate[i + 1])
                self.assertAlmostEqual(gain * mcp / seg, 1.0, delta=1e-8)

    def test_flow_direction(self) -> None:
        """Counter-flow enters at the exit; co-flow at the injector face."""
        counter, co = _solve(41, flow="counter"), _solve(41, flow="co")
        self.assertEqual(counter.coolant_temperature[-1], 290.0)
        self.assertEqual(co.coolant_temperature[0], 290.0)
        self.assertEqual(
            counter.coolant_outlet_temperature, counter.coolant_temperature[0]
        )
        self.assertEqual(co.coolant_outlet_temperature, co.coolant_temperature[-1])
        self.assertTrue(
            all(
                a > b
                for a, b in zip(
                    counter.coolant_temperature,
                    counter.coolant_temperature[1:],
                    strict=False,
                )
            )
        )
        self.assertTrue(
            all(
                a < b
                for a, b in zip(
                    co.coolant_temperature, co.coolant_temperature[1:], strict=False
                )
            )
        )
        # Measured for this synthetic case: counter-flow peak is 30+ K hotter.
        self.assertGreater(
            counter.peak_wall_temperature - co.peak_wall_temperature, 10.0
        )
        with self.assertRaises(ValueError):
            _solve(41, flow="parallel")

    def test_throat_is_the_peak_heat_flux_location(self) -> None:
        """The heat-flux maximum sits at the throat (measured, then asserted)."""
        coarse = _solve(81)
        self.assertEqual(coarse.peak_heat_flux_index, coarse.throat_index)
        fine = _solve(641)
        x_throat = _contour().x_throat
        offset = fine.x[fine.peak_heat_flux_index] - x_throat
        # measured offset 1.3 mm = 0.026 r_t, from the unequal throat arcs
        self.assertLess(abs(offset), 0.1 * RT)
        self.assertGreater(
            fine.heat_flux[fine.throat_index], 0.999 * fine.peak_heat_flux
        )
        # the flux is several times larger than in the chamber and at the exit
        self.assertGreater(fine.peak_heat_flux, 2.0 * fine.heat_flux[0])
        self.assertGreater(fine.peak_heat_flux, 5.0 * fine.heat_flux[-1])
        # the peak wall temperature is near the throat (measured 5.2 mm = 0.10 r_t)
        self.assertLess(abs(fine.x[fine.peak_wall_index] - x_throat), 0.15 * RT)

    def test_convergence_with_station_count(self) -> None:
        """Heat load and outlet temperature converge as stations are added.

        Reference: 1025 stations. Measured relative errors of the total heat
        load at 17, 33, 65 and 129 stations: 1.7e-3, 1.2e-3, 5.0e-4, 1.6e-4.
        The contour has curvature jumps at the throat arcs, so the observed
        rate is between first and second order, not clean second order.
        """
        reference = _solve(1025)
        errors = []
        outlet = []
        for n in (17, 33, 65, 129):
            sol = _solve(n)
            errors.append(
                abs(sol.total_heat_load - reference.total_heat_load)
                / reference.total_heat_load
            )
            outlet.append(
                abs(
                    sol.coolant_outlet_temperature
                    - reference.coolant_outlet_temperature
                )
            )
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:], strict=False)))
        self.assertTrue(all(b < a for a, b in zip(outlet, outlet[1:], strict=False)))
        self.assertLess(errors[0], 3e-3)
        self.assertLess(errors[-1], 3e-4)
        self.assertLess(errors[-1], errors[0] / 3.0)
        # outlet error follows the same law (it is Q / (mdot cp))
        self.assertLess(outlet[-1], 0.05)
        # the peak wall temperature is converged to 3 K at 65 stations
        self.assertAlmostEqual(
            _solve(65).peak_wall_temperature, reference.peak_wall_temperature, delta=3.0
        )

    def test_physical_trends(self) -> None:
        """More coolant cools; thicker walls and a hotter gas side heat."""
        base = _solve(41)
        self.assertLess(
            _solve(41, coolant_mass_flow=13.0).peak_wall_temperature,
            base.peak_wall_temperature,
        )
        self.assertLess(
            _solve(41, coolant_mass_flow=13.0).coolant_outlet_temperature,
            base.coolant_outlet_temperature,
        )
        self.assertGreater(
            _solve(41, wall_thickness=2e-3).peak_wall_temperature,
            base.peak_wall_temperature,
        )
        self.assertGreater(
            _solve(41, chamber_pressure=7e6).peak_heat_flux, base.peak_heat_flux
        )
        self.assertGreater(
            _solve(41, coolant_inlet_temperature=330.0).peak_wall_temperature,
            base.peak_wall_temperature,
        )
        profile = station_profile(_contour(), GAMMA, 41)
        hot = solve_regenerative_cooling(
            profile,
            gas=_gas(),
            coolant=_coolant(),
            channel=_channel(),
            bartz_multiplier=1.2,
            **_operating(),
        )
        self.assertGreater(hot.peak_wall_temperature, base.peak_wall_temperature)
        self.assertGreater(hot.total_heat_load, base.total_heat_load)
        # nominal order of magnitude of this synthetic case
        self.assertGreater(base.peak_heat_flux, 1e7)
        self.assertLess(base.peak_heat_flux, 1e8)
        self.assertTrue(base.correlation_valid)

    def test_boiling_flag_and_validity_flags(self) -> None:
        """Starving the coolant raises the boiling flag; low Re voids validity."""
        base = _solve(41)
        self.assertFalse(base.boiling_flag)
        self.assertLess(base.max_coolant_wall_temperature, 700.0)
        starved = _solve(41, coolant_mass_flow=5.0)
        self.assertTrue(starved.boiling_flag)
        self.assertGreater(starved.max_coolant_wall_temperature, 700.0)
        self.assertTrue(starved.correlation_valid)
        laminar = _solve(41, coolant_mass_flow=0.3)
        self.assertFalse(laminar.correlation_valid)
        self.assertLess(laminar.coolant_reynolds, 1e4)

    def test_input_validation(self) -> None:
        """Non-physical inputs raise; a wrong profile type raises."""
        with self.assertRaises(ValueError):
            _solve(41, wall_thickness=-1e-3)
        with self.assertRaises(ValueError):
            _solve(41, coolant_mass_flow=0.0)
        with self.assertRaises(TypeError):
            solve_regenerative_cooling(
                "profile",  # type: ignore[arg-type]
                gas=_gas(),
                coolant=_coolant(),
                channel=_channel(),
                **_operating(),
            )


def _model(**overrides: object):  # noqa: ANN202
    """Build the synthetic UQ model (8 uncertain inputs, 41 stations)."""
    config: dict[str, object] = {
        "bartz_multiplier": LogNormal(0.0, 0.15),
        "coolant_multiplier": LogNormal(0.0, 0.10),
        "wall_conductivity": TruncatedNormal(320.0, 20.0, lower=200.0),
        "wall_thickness": TruncatedNormal(1.0e-3, 5.0e-5, lower=5e-4, upper=1.5e-3),
        "coolant_mass_flow": TruncatedNormal(11.0, 0.4, lower=8.0),
        "chamber_pressure": Normal(6.0e6, 1.5e5),
        "characteristic_velocity": Normal(1750.0, 20.0),
        "coolant_inlet_temperature": Normal(290.0, 5.0),
    }
    config.update(overrides)
    return nozzle_cooling_model(
        profile=station_profile(_contour(), GAMMA, 41),
        gas=_gas(),
        coolant=_coolant(),
        channel=_channel(),
        wall_temperature_limit=900.0,
        **config,  # type: ignore[arg-type]
    )


class ModelTests(unittest.TestCase):
    """Model construction, evaluation and failure policy."""

    def test_nominal_evaluation_matches_direct_solver(self) -> None:
        """The model reproduces solve_regenerative_cooling at one point."""
        model = _model()
        self.assertEqual(model.dimension, 8)
        point = (1.0, 1.0, 320.0, 1.0e-3, 11.0, 6.0e6, 1750.0, 290.0)
        out = model.evaluate(point)
        direct = _solve(41)
        self.assertAlmostEqual(
            out["peak_wall_temperature"], direct.peak_wall_temperature, places=9
        )
        self.assertAlmostEqual(out["total_heat_load"], direct.total_heat_load, places=3)
        self.assertAlmostEqual(
            out["wall_temperature_margin"],
            900.0 - direct.peak_wall_temperature,
            places=9,
        )
        self.assertAlmostEqual(
            out["coolant_wall_margin"],
            700.0 - direct.max_coolant_wall_temperature,
            places=9,
        )
        self.assertAlmostEqual(
            out["peak_wall_location"], direct.x[direct.peak_wall_index], places=12
        )
        names = set(model.quantity_names)
        self.assertEqual(
            names,
            {
                "peak_wall_temperature",
                "peak_wall_location",
                "coolant_outlet_temperature",
                "total_heat_load",
                "wall_temperature_margin",
                "peak_heat_flux",
                "coolant_wall_margin",
            },
        )

    def test_construction_errors(self) -> None:
        """Bad configuration is rejected."""
        profile = station_profile(_contour(), GAMMA, 21)
        common = {
            "profile": profile,
            "gas": _gas(),
            "coolant": _coolant(),
            "channel": _channel(),
        }
        fixed = {
            "chamber_pressure": 6e6,
            "characteristic_velocity": 1750.0,
            "coolant_mass_flow": 11.0,
            "coolant_inlet_temperature": 290.0,
            "wall_conductivity": 320.0,
            "wall_thickness": 1e-3,
        }
        with self.assertRaises(ValueError):  # nothing uncertain
            nozzle_cooling_model(**common, **fixed)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            nozzle_cooling_model(**common, **{**fixed, "chamber_pressure": True})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            nozzle_cooling_model(
                **common,
                **{**fixed, "wall_thickness": -1e-3},  # type: ignore[arg-type]
                bartz_multiplier=LogNormal(0.0, 0.1),
            )
        with self.assertRaises(TypeError):
            nozzle_cooling_model(
                **{**common, "profile": "x"},
                **fixed,  # type: ignore[arg-type]
                bartz_multiplier=LogNormal(0.0, 0.1),
            )
        with self.assertRaises(ValueError):
            nozzle_cooling_model(
                **common,
                **fixed,
                flow="x",  # type: ignore[arg-type]
                bartz_multiplier=LogNormal(0.0, 0.1),
            )

    def test_non_physical_draw_fails_the_run(self) -> None:
        """A negative thickness is a run failure, counted under 'skip'."""
        bad = (1.0, 1.0, 320.0, -1.0e-3, 11.0, 6.0e6, 1750.0, 290.0)
        good = (1.0, 1.0, 320.0, 1.0e-3, 11.0, 6.0e6, 1750.0, 290.0)
        with self.assertRaises(Exception) as caught:  # noqa: B017
            _model().evaluate(bad)
        self.assertIn("wall_thickness", str(caught.exception))
        model = _model(on_failure="skip")
        batch = model.evaluate_many([good, bad, good])
        self.assertEqual(len(batch["peak_wall_temperature"]), 2)
        self.assertEqual(model.failed_evaluations, 1)

    def test_boiling_flag_in_model_output(self) -> None:
        """A starved coolant gives a negative coolant-side margin."""
        out = _model().evaluate((1.0, 1.0, 320.0, 1.0e-3, 5.0, 6.0e6, 1750.0, 290.0))
        self.assertLess(out["coolant_wall_margin"], 0.0)


class UncertaintyTests(unittest.TestCase):
    """Sobol ranking, reliability methods and the GP emulator."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the shared model."""
        warnings.simplefilter("ignore")
        cls.model = _model()

    def test_sobol_ranking_of_peak_wall_temperature(self) -> None:
        """The Bartz multiplier dominates; the coolant multiplier is second.

        Measured (size 128, seed 1): first-order 0.88 and 0.062, total 0.882
        and 0.070; every other first-order index is below 0.02.
        """
        result = sobol_analysis(self.model, "peak_wall_temperature", 128, seed=1)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        total = {
            n: i.estimate
            for n, i in zip(result.input_names, result.total_effect, strict=True)
        }
        ranked = sorted(total, key=lambda k: total[k], reverse=True)
        self.assertEqual(ranked[0], "bartz_multiplier")
        self.assertEqual(ranked[1], "coolant_multiplier")
        self.assertGreater(first["bartz_multiplier"], 0.75)
        self.assertLess(first["bartz_multiplier"], 0.95)
        self.assertGreater(first["coolant_multiplier"], 0.02)
        for name in ("wall_conductivity", "wall_thickness", "chamber_pressure"):
            self.assertLess(total[name], 0.05)
        self.assertAlmostEqual(sum(first.values()), 1.0, delta=0.1)

    def test_exceedance_probability_form_subset_and_monte_carlo(self) -> None:
        """P(T_wg,max > 880 K) by crude MC, FORM and subset simulation.

        880 K is a synthetic limit on the synthetic case. Measured: MC (2000,
        seed 3) 0.0605 with standard error 0.0053; FORM 0.0700; subset
        simulation (300 per level, seed 5) 0.077 with upper c.o.v. 0.23.
        Bands: 3 combined standard errors for subset simulation, 0.03 absolute
        for first-order FORM (the failure boundary is mildly curved in the
        standard normal space because the multipliers are log-normal).
        """
        state = ModelLimitState(self.model, "peak_wall_temperature", 880.0, "above")
        mc = model_monte_carlo(state, 1500, seed=3)
        self.assertGreater(mc.probability, 0.03)
        self.assertLess(mc.probability, 0.12)
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.03)
        subset = subset_simulation(state, samples_per_level=300, seed=5)
        self.assertTrue(subset.converged)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)
        self.assertGreater(form.importance_factors["bartz_multiplier"], 0.7)
        self.assertLess(form.importance_factors["wall_thickness"], 0.05)

    def test_gp_emulator_validated_on_held_out_runs(self) -> None:
        """A GP of the peak wall temperature, validated on 40 held-out runs.

        Measured (48 LHS training runs, seed 2, one restart; 40 held-out LHS
        runs, seed 11): RMSE 0.34 K against a held-out standard deviation of
        60 K, credible-interval coverage 1.0. The library's Mahalanobis check
        calls the emulator conservative (distance 18.6, too small for
        chi-square(40)); that is the only flag and it is not a failure of
        accuracy.
        """
        gp = fit_gaussian_process_to_model(
            self.model,
            "peak_wall_temperature",
            size=48,
            design="lhs",
            seed=2,
            restarts=1,
        )
        unit = make_design("lhs", 40, self.model.dimension, seed=11)
        points = self.model.inputs.sample_design(unit)
        values = self.model.evaluate_many(points)["peak_wall_temperature"]
        report = validate_gp_emulator(gp, points, values)
        errors = [gp.predict(x) - y for x, y in zip(points, values, strict=True)]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        spread = statistics.pstdev(values)
        self.assertGreater(spread, 40.0)
        self.assertLess(rmse, 0.02 * spread)
        self.assertLess(max(abs(e) for e in errors), 0.08 * spread)
        self.assertGreaterEqual(report.coverage, 0.9)
        self.assertTrue(report.full_rank or report.coverage >= 0.9)
        for flag in report.flags:
            self.assertIn("too small", flag)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowUncertaintyTests(unittest.TestCase):
    """Tighter agreement of the three exceedance estimators (slow tier)."""

    def test_exceedance_probability_with_many_samples(self) -> None:
        """Crude MC with 12000 samples against FORM and subset simulation."""
        warnings.simplefilter("ignore")
        state = ModelLimitState(_model(), "peak_wall_temperature", 900.0, "above")
        mc = model_monte_carlo(state, 12000, seed=7)
        form = form_hlrf(state)
        subset = subset_simulation(state, samples_per_level=1000, seed=9)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.015)


if __name__ == "__main__":
    unittest.main()
