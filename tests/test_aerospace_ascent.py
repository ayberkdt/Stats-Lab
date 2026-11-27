"""Tests for the planar ascent model and its dispersion analysis.

All vehicle numbers are illustrative and synthetic; none describes a real
launcher. Closed forms first (Tsiolkovsky with a uniform-gravity vertical
burn, standard-atmosphere tables, Kepler relations), then independent
oracles (hydrostatic integration, a Kepler period), then the dispersion
workflows and the failure cases.
"""

from __future__ import annotations

import math
import os
import unittest
from collections.abc import Mapping

from probstat.aerospace.ascent import (
    EARTH_MU,
    EARTH_RADIUS,
    STANDARD_GRAVITY,
    DragModel,
    GuidanceSpec,
    InsertionWindow,
    StageSpec,
    VehicleSpec,
    ascent_model,
    atmosphere,
    coast_arc,
    dispersion_budget,
    dispersion_ellipse,
    dispersion_intervals,
    geometric_from_geopotential,
    orbital_elements,
    rocket_equation_delta_v,
    simulate_ascent,
    specific_energy,
    stage_from_engine,
    vertical_burn_closed_form,
)
from probstat.aerospace.liquid_engine import (
    DEMONSTRATION_MIXTURE_TABLE,
    ideal_rocket_performance,
)
from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.numerics import integrate_fixed_step
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis
from probstat.uq import QuantityOfInterest

_DRAG = DragModel.tabulated(
    [(0.0, 0.3), (0.8, 0.3), (1.2, 0.5), (2.0, 0.4), (5.0, 0.25), (50.0, 0.2)]
)


def _vehicle() -> VehicleSpec:
    """Illustrative two-stage vehicle (synthetic numbers, not a real launcher)."""
    first = StageSpec(4000.0, 42000.0, 1.4e6, 300.0, 1.0, 3.0, coast_after=2.0)
    second = StageSpec(800.0, 4500.0, 60e3, 340.0, 0.1, 3.0)
    return VehicleSpec((first, second), 500.0)


_GUIDANCE = GuidanceSpec(8.0, 0.5)
_WINDOW = InsertionWindow(100e3, 2000e3, 8000e3)


def _model(scale: float = 1.0, **overrides: object) -> ProbabilisticModel:
    """Dispersion model; ``scale`` multiplies every standard deviation."""
    config: dict[str, object] = {
        "thrust_scale": Normal(1.0, 0.002 * scale),
        "isp_scale": Normal(1.0, 0.002 * scale),
        "propellant_scale": Normal(1.0, 0.002 * scale),
        "dry_mass_scale": Normal(1.0, 0.01 * scale),
        "drag_scale": Normal(1.0, 0.05 * scale),
        "density_scale": Normal(1.0, 0.03 * scale),
        "pitch_kick_angle": Normal(0.5, 0.004 * scale),
        "pitch_kick_time": Normal(8.0, 0.1 * scale),
        "window": _WINDOW,
        "max_step": 4.0,
    }
    config.update(overrides)
    return ascent_model(_vehicle(), _GUIDANCE, _DRAG, **config)  # type: ignore[arg-type]


class AtmosphereTests(unittest.TestCase):
    """U.S. Standard Atmosphere 1976 layers."""

    def test_sea_level(self) -> None:
        """Sea level: 288.15 K, 101325 Pa, 1.225 kg/m^3, 340.29 m/s."""
        state = atmosphere(0.0)
        self.assertAlmostEqual(state.temperature, 288.15, places=12)
        self.assertAlmostEqual(state.pressure, 101325.0, places=9)
        self.assertAlmostEqual(state.density, 1.225, delta=1e-5)
        self.assertAlmostEqual(state.speed_of_sound, 340.294, delta=2e-3)
        self.assertFalse(state.extended)

    def test_tabulated_layer_bases(self) -> None:
        """Temperature, pressure and density at the geopotential layer bases.

        Reference values are the published standard-atmosphere table entries
        at 11, 20, 32 and 47 km geopotential altitude (rounded to the
        digits quoted), so the tolerance is the rounding of the table.
        """
        table = (
            (11000.0, 216.65, 22632.1, 0.36392),
            (20000.0, 216.65, 5474.89, 0.088035),
            (32000.0, 228.65, 868.02, 0.013225),
            (47000.0, 270.65, 110.906, 0.0014275),
        )
        for geopotential, temp, pressure, density in table:
            state = atmosphere(geometric_from_geopotential(geopotential))
            self.assertAlmostEqual(state.temperature, temp, delta=1e-9)
            self.assertAlmostEqual(state.pressure / pressure, 1.0, delta=5e-6)
            self.assertAlmostEqual(state.density / density, 1.0, delta=2e-4)

    def test_hydrostatic_balance_by_independent_integration(self) -> None:
        """RK4 of dp/dH = -p g0 M / (R* T(H)) reproduces the closed-form layers."""
        r_star, m_air = 8314.32, 28.9644

        def temperature(h_geop: float) -> float:
            """Piecewise-linear temperature profile of the 1976 model."""
            bases = (
                (0.0, 288.15, -0.0065),
                (11000.0, 216.65, 0.0),
                (20000.0, 216.65, 0.001),
                (32000.0, 228.65, 0.0028),
                (47000.0, 270.65, 0.0),
                (51000.0, 270.65, -0.0028),
                (71000.0, 214.65, -0.002),
            )
            base = bases[0]
            for row in bases:
                if h_geop >= row[0]:
                    base = row
            return base[1] + base[2] * (h_geop - base[0])

        def rhs(h_geop: float, y: list[float]) -> list[float]:
            """Hydrostatic equation in geopotential altitude."""
            return [-y[0] * STANDARD_GRAVITY * m_air / (r_star * temperature(h_geop))]

        # Integrate each layer separately from its base so that the lapse-rate
        # kink never falls inside a step.
        edges = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
        pressure = 101325.0
        for lower, upper in zip(edges, edges[1:], strict=False):
            steps = 200
            trajectory = integrate_fixed_step(
                rhs, lower, [pressure], step=(upper - lower) / steps, steps=steps
            )
            pressure = trajectory.final_state[0]
            closed = atmosphere(geometric_from_geopotential(upper)).pressure
            self.assertAlmostEqual(pressure / closed, 1.0, delta=1e-8)

    def test_continuity_across_layer_bases(self) -> None:
        """T, p and rho are continuous at every layer boundary."""
        for geopotential in (11e3, 20e3, 32e3, 47e3, 51e3, 71e3):
            below = atmosphere(geometric_from_geopotential(geopotential - 1e-3))
            above = atmosphere(geometric_from_geopotential(geopotential + 1e-3))
            self.assertAlmostEqual(below.temperature, above.temperature, delta=1e-5)
            self.assertAlmostEqual(below.pressure / above.pressure, 1.0, delta=1e-6)
            self.assertAlmostEqual(below.density / above.density, 1.0, delta=1e-6)

    def test_monotone_density_and_extension_flag(self) -> None:
        """Density falls with altitude; the extension starts above 86 km."""
        previous = atmosphere(0.0).density
        for altitude in range(2000, 200001, 2000):
            state = atmosphere(float(altitude))
            self.assertLess(state.density, previous)
            previous = state.density
            if abs(altitude - 86000) > 200:  # rounding of the 86 km boundary
                self.assertEqual(state.extended, altitude > 86000)
        self.assertAlmostEqual(atmosphere(86000.0).temperature, 186.946, delta=1e-3)

    def test_invalid_input(self) -> None:
        """Negative or non-numeric altitudes are rejected."""
        with self.assertRaises(ValueError):
            atmosphere(-1.0)
        with self.assertRaises(TypeError):
            atmosphere("0")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            atmosphere(math.nan)
        with self.assertRaises(ValueError):
            geometric_from_geopotential(-1.0)


class DragAndStageTests(unittest.TestCase):
    """Drag models and stage definitions."""

    def test_drag_models(self) -> None:
        """Constant, tabulated (linear) and scaled drag; no extrapolation."""
        self.assertEqual(DragModel.constant(0.4).at(7.0), 0.4)
        self.assertAlmostEqual(_DRAG.at(1.0), 0.4, places=12)  # midway 0.3..0.5
        self.assertAlmostEqual(_DRAG.at(3.5), 0.325, places=12)
        self.assertAlmostEqual(_DRAG.scaled(2.0).at(1.0), 0.8, places=12)
        with self.assertRaises(ValueError):
            _DRAG.at(60.0)
        with self.assertRaises(ValueError):
            DragModel.tabulated([(1.0, 0.3), (0.5, 0.3)])
        with self.assertRaises(ValueError):
            DragModel.constant(-0.1)
        with self.assertRaises(ValueError):
            _DRAG.scaled(0.0)

    def test_stage_thrust_and_isp_depend_on_ambient_pressure(self) -> None:
        """F = F_vac - p_a A_e; Isp = F / (mdot g0)."""
        stage = StageSpec(100.0, 900.0, 5.0e5, 300.0, 0.5, 1.0)
        self.assertAlmostEqual(stage.mass_flow, 5.0e5 / (300.0 * STANDARD_GRAVITY))
        self.assertAlmostEqual(stage.burn_time, 900.0 / stage.mass_flow)
        self.assertAlmostEqual(stage.thrust(0.0), 5.0e5)
        self.assertAlmostEqual(stage.thrust(101325.0), 5.0e5 - 0.5 * 101325.0)
        self.assertAlmostEqual(stage.specific_impulse(0.0), 300.0, places=10)
        self.assertLess(stage.specific_impulse(101325.0), 300.0)

    def test_stage_from_engine_uses_vacuum_performance(self) -> None:
        """A stage built from ideal-rocket performance reproduces Isp_vac and F_vac."""
        state = DEMONSTRATION_MIXTURE_TABLE.at(2.8)
        performance = ideal_rocket_performance(
            gamma=state.gamma,
            molecular_weight=state.molecular_weight,
            chamber_temperature=state.chamber_temperature,
            chamber_pressure=7.0e6,
            throat_area=0.25 * math.pi * 0.1**2,
            expansion_ratio=40.0,
            ambient_pressure=101325.0,
        )
        stage = stage_from_engine(
            performance,
            exit_area=2.0 * 0.25 * math.pi * 0.1**2 * 40.0,
            dry_mass=200.0,
            propellant_mass=2000.0,
            reference_area=0.5,
            engines=2,
        )
        self.assertAlmostEqual(stage.vacuum_thrust, 2.0 * performance.vacuum_thrust)
        self.assertAlmostEqual(stage.vacuum_isp, performance.vacuum_specific_impulse)
        # Mass flow of two engines is twice the engine mass flow.
        self.assertAlmostEqual(
            stage.mass_flow / (2.0 * performance.mass_flow), 1.0, delta=1e-12
        )
        # Sea-level Isp of the stage matches the single engine's.
        self.assertAlmostEqual(
            stage.specific_impulse(101325.0) / performance.specific_impulse,
            1.0,
            delta=1e-12,
        )
        with self.assertRaises(TypeError):
            stage_from_engine(
                "engine",  # type: ignore[arg-type]
                exit_area=1.0,
                dry_mass=1.0,
                propellant_mass=1.0,
                reference_area=1.0,
            )

    def test_invalid_stage_vehicle_guidance(self) -> None:
        """Out-of-range data are rejected."""
        with self.assertRaises(ValueError):
            StageSpec(100.0, 900.0, 5.0e4, 300.0, 1.0, 1.0)  # F_vac < p0 A_e
        with self.assertRaises(ValueError):
            StageSpec(-1.0, 900.0, 5.0e5, 300.0, 0.5, 1.0)
        with self.assertRaises(TypeError):
            StageSpec("a", 900.0, 5.0e5, 300.0, 0.5, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VehicleSpec((), 1.0)
        with self.assertRaises(TypeError):
            VehicleSpec((1,), 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GuidanceSpec(-1.0, 0.1)
        with self.assertRaises(ValueError):
            GuidanceSpec(5.0, 2.0)
        with self.assertRaises(ValueError):
            InsertionWindow(0.0, 5.0e5, 1.0e5)


class VacuumBurnTests(unittest.TestCase):
    """Vertical vacuum burn against the Tsiolkovsky closed form."""

    @staticmethod
    def _stage() -> StageSpec:
        """Single stage, mass ratio 10, thrust-to-weight 4."""
        return StageSpec(1000.0, 9000.0, 4.0e5, 300.0, 0.0, 1.0)

    def test_matches_closed_form_in_uniform_gravity(self) -> None:
        """Speed, altitude and gravity loss equal the closed form (RK4, h = 0.1 s)."""
        stage = self._stage()
        result = simulate_ascent(
            VehicleSpec((stage,), 0.0),
            GuidanceSpec.vertical(),
            drag=DragModel.constant(0.0),
            max_step=0.1,
            atmosphere_on=False,
            uniform_gravity=True,
        )
        self.assertEqual(result.status, "burnout")
        speed, altitude = vertical_burn_closed_form(
            300.0, 10000.0, stage.mass_flow, stage.burn_time
        )
        self.assertAlmostEqual(result.speed / speed, 1.0, delta=1e-8)
        self.assertAlmostEqual(result.altitude / altitude, 1.0, delta=1e-8)
        self.assertAlmostEqual(result.time, stage.burn_time, places=10)
        self.assertAlmostEqual(result.mass, 1000.0, places=7)
        # Gravity loss is exactly g * t_b; the thrust increment is Tsiolkovsky.
        self.assertAlmostEqual(
            result.gravity_loss / (STANDARD_GRAVITY * stage.burn_time), 1.0, delta=1e-10
        )
        self.assertAlmostEqual(
            result.thrust_delta_v / rocket_equation_delta_v(300.0, 10000.0, 1000.0),
            1.0,
            delta=1e-9,
        )
        self.assertAlmostEqual(result.drag_loss, 0.0, places=12)
        # Radial burnout has no Keplerian orbit.
        self.assertIsNone(result.elements)

    def test_loss_budget_identity_with_inverse_square_gravity(self) -> None:
        """Speed = thrust_dv - gravity_loss - drag_loss, and less gravity loss aloft."""
        stage = self._stage()
        vehicle = VehicleSpec((stage,), 0.0)
        flat = simulate_ascent(
            vehicle,
            GuidanceSpec.vertical(),
            drag=DragModel.constant(0.0),
            max_step=0.5,
            atmosphere_on=False,
            uniform_gravity=True,
        )
        real = simulate_ascent(
            vehicle,
            GuidanceSpec.vertical(),
            drag=DragModel.constant(0.0),
            max_step=0.5,
            atmosphere_on=False,
        )
        self.assertAlmostEqual(
            real.thrust_delta_v - real.gravity_loss - real.drag_loss,
            real.speed,
            delta=1e-8,
        )
        # g(r) < g0 above the ground, so less gravity loss and a higher speed;
        # the altitude here (~150 km) bounds the effect to about 5 percent.
        self.assertLess(real.gravity_loss, flat.gravity_loss)
        self.assertGreater(real.speed, flat.speed)
        self.assertLess((real.speed - flat.speed) / flat.speed, 0.05)
        with_air = simulate_ascent(
            vehicle,
            GuidanceSpec.vertical(),
            drag=DragModel.constant(0.3),
            max_step=0.5,
        )
        self.assertGreater(with_air.drag_loss, 0.0)
        self.assertAlmostEqual(
            with_air.thrust_delta_v - with_air.gravity_loss - with_air.drag_loss,
            with_air.speed,
            delta=1e-8,
        )
        # Atmospheric pressure reduces thrust only if the exit area is positive.
        self.assertLess(with_air.speed, real.speed)

    def test_rocket_equation_helpers(self) -> None:
        """Isp g0 ln(m0/mf) and closed-form input checks."""
        self.assertAlmostEqual(
            rocket_equation_delta_v(300.0, 10.0, 5.0),
            300.0 * STANDARD_GRAVITY * math.log(2.0),
            places=12,
        )
        with self.assertRaises(ValueError):
            rocket_equation_delta_v(300.0, 5.0, 10.0)
        with self.assertRaises(ValueError):
            vertical_burn_closed_form(300.0, 10.0, 1.0, 10.0)  # exhausts the mass
        speed, altitude = vertical_burn_closed_form(300.0, 100.0, 1.0, 0.0)
        self.assertEqual((speed, altitude), (0.0, 0.0))


class OrbitTests(unittest.TestCase):
    """Keplerian elements and ballistic coasting."""

    def test_circular_state(self) -> None:
        """Speed sqrt(mu/r) at gamma = 0 gives e = 0 and perigee = apogee = altitude."""
        altitude = 400e3
        radius = EARTH_RADIUS + altitude
        speed = math.sqrt(EARTH_MU / radius)
        elements = orbital_elements(altitude, speed, 0.0)
        self.assertLess(elements.eccentricity, 1e-7)
        self.assertAlmostEqual(elements.semi_major_axis, radius, delta=1e-3)
        self.assertAlmostEqual(elements.perigee_altitude, altitude, delta=1e-3)
        self.assertAlmostEqual(elements.apogee_altitude, altitude, delta=1e-3)
        self.assertAlmostEqual(
            elements.specific_energy, -EARTH_MU / (2.0 * radius), delta=1e-6
        )
        self.assertTrue(elements.bound)

    def test_elliptic_perigee_state(self) -> None:
        """A perigee state (gamma = 0) with v = sqrt(mu (1+e) / r_p) has that e."""
        r_p = EARTH_RADIUS + 300e3
        ecc = 0.2
        speed = math.sqrt(EARTH_MU * (1.0 + ecc) / r_p)
        elements = orbital_elements(300e3, speed, 0.0)
        self.assertAlmostEqual(elements.eccentricity, ecc, places=12)
        self.assertAlmostEqual(elements.perigee_altitude, 300e3, delta=1e-4)
        self.assertAlmostEqual(
            elements.apogee_radius, r_p * (1.0 + ecc) / (1.0 - ecc), delta=1e-3
        )
        # The same orbit seen at the apogee (smaller speed, gamma = 0).
        r_a = r_p * (1.0 + ecc) / (1.0 - ecc)
        v_a = math.sqrt(EARTH_MU * (1.0 - ecc) / r_a)
        again = orbital_elements(r_a - EARTH_RADIUS, v_a, 0.0)
        self.assertAlmostEqual(again.eccentricity, ecc, places=12)
        self.assertAlmostEqual(again.perigee_radius, r_p, delta=1e-3)

    def test_unbound_and_degenerate_states(self) -> None:
        """Escape speed is unbound with infinite apogee; radial states are rejected."""
        radius = EARTH_RADIUS + 200e3
        escape = math.sqrt(2.0 * EARTH_MU / radius)
        elements = orbital_elements(200e3, 1.2 * escape, 0.1)
        self.assertFalse(elements.bound)
        self.assertEqual(elements.apogee_altitude, math.inf)
        self.assertLess(elements.semi_major_axis, 0.0)
        self.assertGreater(elements.eccentricity, 1.0)
        with self.assertRaises(ValueError):
            orbital_elements(200e3, 5000.0, math.pi / 2)
        with self.assertRaises(ValueError):
            orbital_elements(200e3, 0.0, 0.0)
        with self.assertRaises(TypeError):
            orbital_elements("a", 1.0, 0.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(
            specific_energy(0.0, 0.0), -EARTH_MU / EARTH_RADIUS, delta=1e-6
        )

    def test_energy_and_momentum_conserved_in_coast(self) -> None:
        """A 40-minute ballistic arc conserves energy and angular momentum."""
        altitude, speed, gamma = 250e3, 8100.0, 0.05
        before = orbital_elements(altitude, speed, gamma)
        h_end, v_end, g_end, _ = coast_arc(
            altitude, speed, gamma, 2400.0, max_step=10.0
        )
        after = orbital_elements(h_end, v_end, g_end)
        self.assertLess(abs(after.specific_energy / before.specific_energy - 1.0), 1e-9)
        self.assertLess(
            abs(after.angular_momentum / before.angular_momentum - 1.0), 1e-9
        )
        self.assertAlmostEqual(after.eccentricity, before.eccentricity, places=9)

    def test_full_period_returns_to_start(self) -> None:
        """After one Kepler period 2 pi sqrt(a^3/mu) the state repeats."""
        altitude, speed, gamma = 250e3, 8100.0, 0.05
        elements = orbital_elements(altitude, speed, gamma)
        period = 2.0 * math.pi * math.sqrt(elements.semi_major_axis**3 / EARTH_MU)
        h_end, v_end, g_end, angle = coast_arc(
            altitude, speed, gamma, period, max_step=2.0
        )
        self.assertAlmostEqual(h_end, altitude, delta=0.5)
        self.assertAlmostEqual(v_end, speed, delta=1e-3)
        self.assertAlmostEqual(g_end, gamma, delta=1e-6)
        self.assertAlmostEqual(angle, 2.0 * math.pi, delta=1e-6)

    def test_coast_validation(self) -> None:
        """Invalid coast arguments are rejected."""
        with self.assertRaises(ValueError):
            coast_arc(250e3, 8100.0, 0.0, -1.0)
        with self.assertRaises(ValueError):
            coast_arc(250e3, 0.0, 0.0, 10.0)


class AscentTests(unittest.TestCase):
    """Two-stage gravity-turn ascent."""

    def test_nominal_ascent(self) -> None:
        """Burnout on a bound orbit inside the illustrative window."""
        vehicle = _vehicle()
        result = simulate_ascent(
            vehicle, _GUIDANCE, drag=_DRAG, max_step=2.0, window=_WINDOW, record=True
        )
        self.assertEqual(result.status, "burnout")
        elements = result.elements
        assert elements is not None
        self.assertTrue(elements.bound)
        self.assertTrue(result.insertion_success)
        self.assertGreater(elements.perigee_altitude, 100e3)
        self.assertLess(elements.perigee_altitude, 200e3)
        self.assertGreater(elements.apogee_altitude, elements.perigee_altitude)
        # Burnout time = both burn times plus the 2 s coast; mass is dry + payload.
        burn = sum(s.burn_time for s in vehicle.stages) + 2.0
        self.assertAlmostEqual(result.time, burn, places=9)
        self.assertAlmostEqual(result.mass, 500.0 + 800.0, places=6)
        # Recorded grid is monotone in time and mass never increases.
        times = [p[0] for p in result.trajectory]
        masses = [p[4] for p in result.trajectory]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:], strict=False)))
        self.assertTrue(
            all(b <= a + 1e-9 for a, b in zip(masses, masses[1:], strict=False))
        )
        self.assertGreater(result.max_dynamic_pressure, 1e4)
        self.assertLess(result.max_dynamic_pressure, 2e5)
        # Loss budget closes along the integration.
        self.assertAlmostEqual(
            result.thrust_delta_v - result.gravity_loss - result.drag_loss,
            result.speed,
            delta=1e-6,
        )
        self.assertGreater(result.gravity_loss, 500.0)

    def test_step_refinement_accuracy(self) -> None:
        """Perigee errors against the 0.25 s result shrink with the step.

        Measured offline for this vehicle: about 38 m at 4 s, 7.6 m at 2 s
        and 0.8 m at 1 s (second-order-or-better decay; the drag table and
        the atmosphere layer kinks limit the formal RK4 order).
        """
        fine = simulate_ascent(_vehicle(), _GUIDANCE, drag=_DRAG, max_step=0.25)
        assert fine.elements is not None
        errors = []
        for step, bound in ((4.0, 60.0), (2.0, 15.0), (1.0, 3.0)):
            coarse = simulate_ascent(_vehicle(), _GUIDANCE, drag=_DRAG, max_step=step)
            assert coarse.elements is not None
            error = abs(
                coarse.elements.perigee_altitude - fine.elements.perigee_altitude
            )
            self.assertLess(error, bound)
            errors.append(error)
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_energy_gain_equals_work_of_net_force(self) -> None:
        """Without drag, specific energy changes only through thrust power."""
        vehicle = VehicleSpec((StageSpec(1000.0, 9000.0, 4.0e5, 300.0, 0.0, 1.0),), 0.0)
        result = simulate_ascent(
            vehicle,
            GuidanceSpec(10.0, 0.3),
            drag=DragModel.constant(0.0),
            max_step=0.5,
            atmosphere_on=False,
        )
        self.assertEqual(result.status, "burnout")
        # dE/dt = (F/m) v along the path, so E_final - E_0 = int (F/m) v dt >= 0
        # and it exceeds the kinetic energy of the same speed in free fall.
        energy0 = specific_energy(0.0, 0.0)
        gain = specific_energy(result.altitude, result.speed) - energy0
        self.assertGreater(gain, 0.0)
        self.assertGreater(result.speed, 0.0)

    def test_impact_is_located_and_reported(self) -> None:
        """A near-horizontal kick at low speed falls back to the ground."""
        result = simulate_ascent(
            _vehicle(), GuidanceSpec(8.0, 1.4), drag=_DRAG, max_step=1.0
        )
        self.assertEqual(result.status, "impact")
        self.assertLess(abs(result.altitude), 1e-5)
        self.assertGreater(result.time, 8.0)
        self.assertIsNone(result.elements)
        self.assertIsNone(result.insertion_success)

    def test_no_liftoff(self) -> None:
        """Thrust-to-weight below one is reported, not integrated."""
        weak = StageSpec(4000.0, 42000.0, 4.0e5, 300.0, 0.1, 3.0)
        result = simulate_ascent(
            VehicleSpec((weak,), 500.0), _GUIDANCE, drag=_DRAG, max_step=2.0
        )
        self.assertEqual(result.status, "no_liftoff")
        self.assertEqual(result.steps, 0)

    def test_drag_table_range_failure_is_reported(self) -> None:
        """A drag table that stops at Mach 2 fails numerically beyond it."""
        short = DragModel.tabulated([(0.0, 0.3), (2.0, 0.3)])
        result = simulate_ascent(_vehicle(), _GUIDANCE, drag=short, max_step=2.0)
        self.assertEqual(result.status, "numerical_failure")
        self.assertIn("outside the drag table", result.stop_reason)

    def test_invalid_arguments(self) -> None:
        """Types, ranges and kick-speed requirement are validated."""
        with self.assertRaises(TypeError):
            simulate_ascent("v", _GUIDANCE, drag=_DRAG)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            simulate_ascent(_vehicle(), _GUIDANCE, drag=0.3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_ascent(_vehicle(), _GUIDANCE, drag=_DRAG, max_step=0.0)
        with self.assertRaises(ValueError):
            simulate_ascent(_vehicle(), _GUIDANCE, drag=_DRAG, density_scale=-1.0)
        with self.assertRaises(ValueError):
            simulate_ascent(_vehicle(), GuidanceSpec(0.0, 0.3), drag=_DRAG)

    def test_density_scale_and_drag_scale_act_on_drag_only(self) -> None:
        """Doubling density doubles drag loss to first order; vacuum has none."""
        base = simulate_ascent(_vehicle(), _GUIDANCE, drag=_DRAG, max_step=2.0)
        dense = simulate_ascent(
            _vehicle(), _GUIDANCE, drag=_DRAG, max_step=2.0, density_scale=1.2
        )
        heavy = simulate_ascent(
            _vehicle(), _GUIDANCE, drag=_DRAG.scaled(1.2), max_step=2.0
        )
        self.assertGreater(dense.drag_loss, base.drag_loss)
        self.assertAlmostEqual(
            dense.drag_loss, heavy.drag_loss, delta=0.05 * base.drag_loss
        )
        vacuum = simulate_ascent(
            _vehicle(), _GUIDANCE, drag=_DRAG, max_step=2.0, atmosphere_on=False
        )
        self.assertEqual(vacuum.drag_loss, 0.0)
        self.assertEqual(vacuum.max_dynamic_pressure, 0.0)


class _Linear:
    """Linear forward map y = 3 a + 4 b used to check the RSS budget."""

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Return the linear response."""
        return {"y": 3.0 * parameters["a"] + 4.0 * parameters["b"]}


class _Quadratic:
    """Quadratic forward map y = a^2 with zero slope at the mean."""

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Return the quadratic response."""
        return {"y": parameters["a"] ** 2 + 0.0 * parameters["b"]}


def _toy(forward: object) -> ProbabilisticModel:
    """Two-input toy model with standard-normal inputs."""
    return ProbabilisticModel(
        JointInput.independent(
            RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))
        ),
        forward,  # type: ignore[arg-type]
        (QuantityOfInterest("y", "response"),),
    )


class ModelTests(unittest.TestCase):
    """The ProbabilisticModel wrapper."""

    def test_nominal_evaluation_matches_direct_simulation(self) -> None:
        """Nominal inputs reproduce simulate_ascent at the same step."""
        model = _model()
        names = model.inputs.names
        self.assertEqual(
            names,
            (
                "thrust_scale",
                "isp_scale",
                "propellant_scale",
                "dry_mass_scale",
                "drag_scale",
                "density_scale",
                "pitch_kick_angle",
                "pitch_kick_time",
            ),
        )
        out = model.evaluate(model.nominal())
        direct = simulate_ascent(
            _vehicle(), _GUIDANCE, drag=_DRAG, max_step=4.0, window=_WINDOW
        )
        assert direct.elements is not None
        self.assertAlmostEqual(
            out["perigee_altitude"], direct.elements.perigee_altitude
        )
        self.assertAlmostEqual(out["apogee_altitude"], direct.elements.apogee_altitude)
        self.assertAlmostEqual(out["burnout_speed"], direct.speed)
        self.assertEqual(out["insertion_success"], 1.0)
        self.assertIn("gravity_loss", model.quantity_names)

    def test_stage_selection_and_scales(self) -> None:
        """Perturbing only stage 2 changes less than perturbing both."""
        both = _model(thrust_scale=Normal(1.0, 0.002))
        second = _model(perturbed_stages=(1,), thrust_scale=Normal(1.0, 0.002))
        point = list(both.nominal())
        point[0] = 1.02
        d_both = (
            both.evaluate(point)["perigee_altitude"]
            - both.evaluate(both.nominal())["perigee_altitude"]
        )
        d_second = (
            second.evaluate(point)["perigee_altitude"]
            - second.evaluate(second.nominal())["perigee_altitude"]
        )
        self.assertNotEqual(d_both, d_second)
        self.assertGreater(abs(d_both), 1e3)

    def test_construction_errors(self) -> None:
        """Bad configuration is rejected."""
        with self.assertRaises(ValueError):
            ascent_model(_vehicle(), _GUIDANCE, _DRAG)  # nothing uncertain
        with self.assertRaises(TypeError):
            ascent_model(
                _vehicle(),
                _GUIDANCE,
                _DRAG,
                thrust_scale=True,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ascent_model(
                _vehicle(),
                _GUIDANCE,
                _DRAG,
                thrust_scale=Normal(1.0, 0.1),
                perturbed_stages=(5,),
            )
        with self.assertRaises(TypeError):
            ascent_model("v", _GUIDANCE, _DRAG)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ascent_model(
                _vehicle(),
                GuidanceSpec.vertical(),
                _DRAG,
                thrust_scale=Normal(1.0, 0.1),
            )

    def test_failed_ascent_is_a_failed_run(self) -> None:
        """A kick that sends the vehicle into the ground fails; skip drops it."""
        model = _model()
        point = list(model.nominal())
        point[model.inputs.names.index("pitch_kick_angle")] = 1.4
        with self.assertRaises(RuntimeError):
            model.evaluate(point)
        skipping = _model(on_failure="skip")
        with self.assertRaises(Exception) as caught:  # noqa: B017
            skipping.evaluate(point)
        self.assertIn("impact", str(caught.exception.__cause__))
        # The failed attempt was counted separately.
        self.assertEqual(skipping.failed_evaluations, 1)

    def test_sobol_ranking_puts_the_kick_first(self) -> None:
        """First-order Sobol indices of perigee: the kick angle ranks first."""
        result = sobol_analysis(_model(), "perigee_altitude", 32, seed=1)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        total = {
            n: i.estimate
            for n, i in zip(result.input_names, result.total_effect, strict=True)
        }
        self.assertEqual(max(first, key=lambda k: first[k]), "pitch_kick_angle")
        self.assertGreater(first["pitch_kick_angle"], 0.25)
        self.assertGreater(total["thrust_scale"], 0.08)
        # Specific-impulse and density dispersions are negligible here.
        self.assertLess(total["isp_scale"], 0.15)
        self.assertLess(total["density_scale"], 0.15)


class DispersionTests(unittest.TestCase):
    """Intervals, ellipse, RSS budget and rare-event estimates."""

    def test_three_sigma_intervals(self) -> None:
        """The 99.73% normal interval is mean -/+ 3 sigma; empirical is inside it."""
        results = dispersion_intervals(
            _model(), ["perigee_altitude", "apogee_altitude"], 200, seed=1
        )
        for interval in results:
            self.assertEqual(interval.samples, 200)
            lo, hi = interval.normal_interval
            self.assertAlmostEqual(
                0.5 * (hi - lo),
                3.0 * interval.standard_deviation,
                delta=2e-3 * interval.standard_deviation,
            )
            self.assertAlmostEqual(0.5 * (hi + lo), interval.mean, delta=1e-6)
            e_lo, e_hi = interval.empirical_interval
            self.assertLess(e_lo, interval.mean)
            self.assertGreater(e_hi, interval.mean)
        perigee = results[0]
        # Mean and spread agree with the independent small-dispersion expectation
        # (about 8 km standard deviation, measured offline for this vehicle).
        self.assertGreater(perigee.mean, 120e3)
        self.assertLess(perigee.mean, 160e3)
        self.assertGreater(perigee.standard_deviation, 3e3)
        self.assertLess(perigee.standard_deviation, 20e3)

    def test_ellipse_closed_form(self) -> None:
        """Exact axes for a sample with known covariance; radius 3.439 at 99.73%."""
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        ys = [x + e for x, e in zip(xs, [0.5, -0.5, 0.0, 0.5, -0.5] * 2, strict=True)]
        ellipse = dispersion_ellipse(xs, ys)
        self.assertAlmostEqual(
            ellipse.radius, math.sqrt(-2.0 * math.log(0.0027)), places=10
        )
        self.assertAlmostEqual(ellipse.radius, 3.4393, places=3)
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs) / (n - 1)
        syy = sum((y - my) ** 2 for y in ys) / (n - 1)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (n - 1)
        spread = math.sqrt(0.25 * (sxx - syy) ** 2 + sxy**2)
        lam1 = 0.5 * (sxx + syy) + spread
        lam2 = 0.5 * (sxx + syy) - spread
        self.assertAlmostEqual(
            ellipse.semi_axes[0], ellipse.radius * math.sqrt(lam1), places=10
        )
        self.assertAlmostEqual(
            ellipse.semi_axes[1], ellipse.radius * math.sqrt(lam2), places=10
        )
        self.assertAlmostEqual(
            ellipse.correlation, sxy / math.sqrt(sxx * syy), places=12
        )
        self.assertEqual(ellipse.empirical_coverage, 1.0)
        # Perfectly correlated data have a singular covariance.
        with self.assertRaises(ValueError):
            dispersion_ellipse([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        with self.assertRaises(ValueError):
            dispersion_ellipse([1.0, 2.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            dispersion_ellipse(xs, ys, coverage=1.0)

    def test_ellipse_of_perigee_and_burnout_altitude(self) -> None:
        """Perigee and burnout altitude are anti-correlated (steeper: higher, lower)."""
        model = _model()
        points = model.inputs.sample(200, seed=4)
        perigee, altitude = [], []
        for point in points:
            out = model.evaluate(point)
            perigee.append(out["perigee_altitude"])
            altitude.append(out["burnout_altitude"])
        ellipse = dispersion_ellipse(perigee, altitude)
        self.assertLess(ellipse.correlation, -0.5)
        self.assertGreaterEqual(ellipse.empirical_coverage, 0.97)
        self.assertGreater(ellipse.semi_axes[0], ellipse.semi_axes[1])

    def test_rss_budget_exact_for_linear_response(self) -> None:
        """Linear 3a + 4b with unit normals: RSS = 5 = the true standard deviation."""
        budget = dispersion_budget(_toy(_Linear()), "y", 600, seed=2)
        self.assertAlmostEqual(budget.rss, 5.0, places=6)
        self.assertAlmostEqual(budget.derivatives[0], 3.0, places=8)
        self.assertAlmostEqual(budget.derivatives[1], 4.0, places=8)
        self.assertAlmostEqual(budget.monte_carlo_std / 5.0, 1.0, delta=0.08)
        self.assertAlmostEqual(budget.nonlinearity, 0.0, places=8)

    def test_rss_budget_fails_for_zero_slope_response(self) -> None:
        """A response a^2 has zero slope at the mean: RSS = 0, true spread sqrt(2)."""
        budget = dispersion_budget(_toy(_Quadratic()), "y", 800, seed=3)
        self.assertLess(budget.rss, 1e-6)
        self.assertGreater(budget.monte_carlo_std, 1.0)
        self.assertAlmostEqual(budget.monte_carlo_std / math.sqrt(2.0), 1.0, delta=0.2)
        self.assertGreater(budget.mean_shift, 0.5)  # E[a^2] = 1, nominal 0
        self.assertAlmostEqual(budget.nonlinearity, 1.0, places=8)

    def test_budget_of_ascent_perigee_small_and_large_dispersions(self) -> None:
        """RSS tracks Monte Carlo for small dispersions and loses it for larger ones."""
        small = dispersion_budget(_model(1.0), "perigee_altitude", 200, seed=2)
        self.assertGreater(small.ratio, 0.85)
        self.assertLess(small.ratio, 1.15)
        self.assertEqual(small.input_names[0], "thrust_scale")
        # Kick angle and thrust dominate the linear budget.
        order = sorted(range(8), key=lambda i: -small.contributions[i])
        self.assertIn(
            small.input_names[order[0]], ("pitch_kick_angle", "pitch_kick_time")
        )
        large = dispersion_budget(_model(2.0), "perigee_altitude", 200, seed=2)
        # Doubling every sigma: the response is no longer linear over the
        # range, RSS (which only doubles) underestimates the Monte Carlo spread
        # and the mean moves away from the nominal output.
        self.assertLess(large.ratio, 0.9)
        self.assertGreater(abs(large.mean_shift), abs(small.mean_shift))
        self.assertGreater(large.nonlinearity, 0.9)

    def test_budget_validation(self) -> None:
        """Unknown quantity, dependent inputs and bad steps are rejected."""
        with self.assertRaises(ValueError):
            dispersion_budget(_model(), "nothing", 10)
        with self.assertRaises(TypeError):
            dispersion_budget("model", "y", 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dispersion_budget(_model(), "perigee_altitude", 10, relative_step=0.0)
        with self.assertRaises(ValueError):
            dispersion_intervals(_model(), [], 10)
        with self.assertRaises(ValueError):
            dispersion_intervals(_model(), ["perigee_altitude"], 1)
        with self.assertRaises(ValueError):
            dispersion_intervals(_model(), ["perigee_altitude"], 10, coverage=1.5)

    def test_failures_stop_a_dispersion_study_unless_policy_skips(self) -> None:
        """Wide dispersions include failed ascents; 'raise' stops, 'skip' drops."""
        wide = _model(8.0)
        with self.assertRaises(RuntimeError):
            dispersion_intervals(wide, ["perigee_altitude"], 200, seed=1)
        skipping = _model(8.0, on_failure="skip")
        results = dispersion_intervals(skipping, ["perigee_altitude"], 200, seed=1)
        self.assertLess(results[0].samples, 200)
        self.assertGreater(results[0].samples, 20)

    def test_probability_of_low_perigee_form_subset_and_mc(self) -> None:
        """P(perigee < 130 km) by FORM, subset simulation and a small MC budget.

        Reference (offline, 3000 runs of the same model): 0.060 with standard
        error 0.004. The 400-run MC, FORM (first order) and subset simulation
        (150 samples per level) must agree within the stated bands.
        """
        state = ModelLimitState(_model(), "perigee_altitude", 130e3, "below")
        mc = model_monte_carlo(state, 400, seed=3)
        self.assertGreater(mc.probability, 0.02)
        self.assertLess(mc.probability, 0.1)
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.03)
        subset = subset_simulation(state, samples_per_level=150, seed=5)
        self.assertTrue(subset.converged)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)
        # The kick angle, thrust and kick time carry the failure.
        weights = form.importance_factors
        self.assertGreater(weights["pitch_kick_angle"], 0.2)
        self.assertLess(weights["isp_scale"], 0.05)

    def test_rare_perigee_event_needs_more_than_a_small_mc_budget(self) -> None:
        """P(perigee < 112 km) is about 0.005: 200 MC runs cannot resolve it."""
        state = ModelLimitState(_model(), "perigee_altitude", 112e3, "below")
        mc = model_monte_carlo(state, 200, seed=3)
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertGreater(form.probability, 0.001)
        self.assertLess(form.probability, 0.012)
        # Standard error of the small-budget estimate is at least half its size
        # (or the estimate is zero): the budget is insufficient.
        self.assertTrue(
            mc.probability == 0.0 or mc.standard_error >= 0.4 * mc.probability
        )
        subset = subset_simulation(state, samples_per_level=150, seed=5)
        self.assertTrue(subset.converged)
        self.assertLess(
            abs(subset.probability - form.probability),
            3.0 * subset.coefficient_of_variation_upper * subset.probability + 1e-12,
        )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowDispersionTests(unittest.TestCase):
    """Larger studies, run only with PROBSTAT_SLOW=1."""

    def test_empirical_three_sigma_coverage(self) -> None:
        """About 99.7% of 3000 perigee draws lie inside the normal 3-sigma interval."""
        model = _model()
        interval = dispersion_intervals(model, ["perigee_altitude"], 3000, seed=21)[0]
        lo, hi = interval.normal_interval
        values = [
            model.evaluate(p)["perigee_altitude"]
            for p in model.inputs.sample(3000, seed=22)
        ]
        inside = sum(lo <= v <= hi for v in values) / len(values)
        self.assertGreater(inside, 0.985)

    def test_sobol_ranking_stable_with_more_samples(self) -> None:
        """With N = 256 the kick angle still ranks first and isp stays negligible."""
        result = sobol_analysis(_model(), "perigee_altitude", 256, seed=9)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        self.assertEqual(max(first, key=lambda k: first[k]), "pitch_kick_angle")
        self.assertLess(first["isp_scale"], 0.1)


if __name__ == "__main__":
    unittest.main()
