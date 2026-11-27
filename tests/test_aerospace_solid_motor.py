"""Tests of the lumped solid-rocket-motor internal-ballistics model.

Order: closed forms of the ballistics and of the grain geometries, the ideal
nozzle against independent formulas, the quasi-steady and transient motor
against closed-form and independent-quadrature oracles, failure reporting, and
finally the probabilistic model (Monte Carlo dispersion, Sobol ranking and a
burst probability by subset simulation against crude Monte Carlo).
"""

from __future__ import annotations

import math
import os
import statistics
import time
import unittest

from probstat.aerospace.solid_motor import (
    BatesGrain,
    EndBurningGrain,
    SolidMotorParameters,
    TubularGrain,
    burn_rate,
    burn_rate_coefficient,
    chamber_rt,
    equilibrium_pressure,
    exit_mach_number,
    exit_pressure_ratio,
    ideal_thrust_coefficient,
    klemmung,
    simulate_solid_motor,
    solid_motor_model,
)
from probstat.optimization import brent_root
from probstat.reliability import ModelLimitState, model_monte_carlo, subset_simulation
from probstat.sensitivity import sobol_analysis_all

G0 = 9.80665


def _motor(**overrides: float) -> SolidMotorParameters:
    """Return the synthetic demonstration motor (not a real motor)."""
    base = {
        "burn_rate_coefficient": 8.0e-3,
        "pressure_exponent": 0.35,
        "characteristic_velocity": 1550.0,
        "propellant_density": 1750.0,
        "throat_diameter": 0.024,
        "chamber_volume": 0.0050,
        "expansion_ratio": 8.0,
        "gamma": 1.22,
        "temperature_sensitivity": 2.5e-3,
        "reference_pressure": 6.0e6,
        "erosion_rate": 0.0,
        "ambient_pressure": 0.0,
    }
    base.update(overrides)
    return SolidMotorParameters(**base)  # type: ignore[arg-type]


def _bates() -> BatesGrain:
    """Return four BATES segments, 0.100 m outer, 0.030 m core, 0.150 m long."""
    return BatesGrain(0.100, 0.030, 0.150, 4)


def _end_burner() -> EndBurningGrain:
    """Return an end burner of radius 0.04 m and length 0.04 m."""
    return EndBurningGrain(0.04, 0.04)


def _end_motor(**overrides: float) -> SolidMotorParameters:
    """Return motor data matched to the end burner (Kn about 240, 5 s burn)."""
    base = {"throat_diameter": 0.0052, "chamber_volume": 4.0e-4}
    base.update(overrides)
    return _motor(**base)


def _simpson(function, lower: float, upper: float, panels: int = 2000) -> float:  # type: ignore[no-untyped-def]
    """Composite Simpson rule used as an independent quadrature oracle."""
    h = (upper - lower) / panels
    total = function(lower) + function(upper)
    for index in range(1, panels):
        total += (4.0 if index % 2 else 2.0) * function(lower + index * h)
    return total * h / 3.0


class BallisticsClosedFormTests(unittest.TestCase):
    """Saint-Robert law, equilibrium pressure and Klemmung."""

    def test_equilibrium_pressure_closed_form_and_mass_balance(self) -> None:
        """P = (a rho c* Kn / p_ref^n)^(1/(1-n)) and generation = discharge."""
        a, n, c_star, rho, p_ref, kn = 8.0e-3, 0.35, 1550.0, 1750.0, 6.0e6, 250.0
        pressure = equilibrium_pressure(a, n, c_star, rho, kn, reference_pressure=p_ref)
        expected = (a * rho * c_star * kn / p_ref**n) ** (1.0 / (1.0 - n))
        self.assertAlmostEqual(pressure / expected, 1.0, places=13)
        # rho_p A_b r(p) = p A_t / c*  with A_t = 1, A_b = Kn
        generation = rho * kn * burn_rate(pressure, a, n, reference_pressure=p_ref)
        discharge = pressure / c_star
        self.assertAlmostEqual(generation / discharge, 1.0, places=12)

    def test_log_slope_in_klemmung_is_one_over_one_minus_n(self) -> None:
        """D ln p / d ln Kn = 1 / (1 - n), and p is strictly increasing in Kn."""
        n = 0.4
        kns = [100.0, 200.0, 400.0, 800.0]
        values = [equilibrium_pressure(5e-3, n, 1500.0, 1700.0, kn) for kn in kns]
        self.assertEqual(values, sorted(values))
        for low, high in zip(values, values[1:], strict=False):
            self.assertAlmostEqual(math.log(high / low) / math.log(2.0), 1 / (1 - n))

    def test_temperature_sensitivity(self) -> None:
        """a(T) = a0 exp(sigma_p dT); the pressure scales by exp(sigma dT/(1-n))."""
        sigma, d_t, n = 2.5e-3, 30.0, 0.35
        hot = burn_rate_coefficient(8.0e-3, sigma, 323.15, 293.15)
        self.assertAlmostEqual(hot, 8.0e-3 * math.exp(sigma * d_t), places=15)
        base = equilibrium_pressure(8.0e-3, n, 1550.0, 1750.0, 250.0)
        warm = equilibrium_pressure(hot, n, 1550.0, 1750.0, 250.0)
        self.assertAlmostEqual(warm / base, math.exp(sigma * d_t / (1 - n)), places=12)

    def test_klemmung_and_chamber_rt(self) -> None:
        """Kn = A_b / A_t; R T_c = (c* Gamma)^2 with Gamma from gamma."""
        self.assertEqual(klemmung(0.1, 0.0004), 250.0)
        gamma = 1.4
        big_gamma = math.sqrt(gamma) * (2 / (gamma + 1)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        )
        self.assertAlmostEqual(big_gamma, 0.6847, places=4)  # air table value
        self.assertAlmostEqual(chamber_rt(1000.0, gamma), (1000.0 * big_gamma) ** 2)

    def test_exponent_at_or_above_one_is_refused(self) -> None:
        """N -> 1: no equilibrium exists, so every entry point refuses it."""
        for bad in (1.0, 1.0 + 1e-12, 1.2, -0.1, math.nan):
            with self.assertRaises(ValueError, msg=f"n={bad}"):
                equilibrium_pressure(8e-3, bad, 1550.0, 1750.0, 250.0)
            with self.assertRaises(ValueError):
                burn_rate(1e6, 8e-3, bad)
            with self.assertRaises(ValueError):
                _motor(pressure_exponent=bad)
        # just below one the form still evaluates (huge, not silent nonsense)
        near = equilibrium_pressure(8e-3, 0.9, 1550.0, 1750.0, 250.0)
        self.assertTrue(math.isfinite(near) and near > 1e9)

    def test_invalid_inputs_are_rejected(self) -> None:
        """Type and range errors raise instead of being repaired."""
        with self.assertRaises(TypeError):
            klemmung("0.1", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            klemmung(0.0, 1.0)
        with self.assertRaises(ValueError):
            burn_rate_coefficient(8e-3, -1e-3, 300.0, 293.15)
        with self.assertRaises(ValueError):
            _motor(gamma=1.0)
        with self.assertRaises(ValueError):
            _motor(expansion_ratio=0.5)
        with self.assertRaises(ValueError):
            _motor(erosion_rate=-1e-6)


class GrainGeometryTests(unittest.TestCase):
    """Exact burning areas, volumes, burn-out and classification."""

    def test_bates_area_at_zero_web_and_at_burnout(self) -> None:
        """A_b(0) and A_b(w) of four BATES segments, by hand."""
        grain = _bates()
        d_out, d_core, length, count = 0.100, 0.030, 0.150, 4
        a0 = count * (
            math.pi * d_core * length + 0.5 * math.pi * (d_out**2 - d_core**2)
        )
        self.assertAlmostEqual(grain.burning_area(0.0), a0, places=14)
        self.assertAlmostEqual(a0, 0.113725654, places=8)
        # radial web (0.035) is smaller than the half length (0.075): the bore
        # has reached the case, only the bore of length L - 2w remains
        self.assertAlmostEqual(grain.web, 0.035, places=15)
        at_end = count * math.pi * d_out * (length - 2 * 0.035)
        self.assertAlmostEqual(grain.burning_area(grain.web), at_end, places=14)
        self.assertEqual(grain.burning_area(grain.web * 1.0000001), 0.0)
        self.assertAlmostEqual(grain.propellant_volume(0.0), 0.0042882740, places=9)
        self.assertEqual(grain.propellant_volume(grain.web), 0.0)

    def test_bates_axial_limited_web(self) -> None:
        """A short segment burns out axially at e = L / 2."""
        grain = BatesGrain(0.100, 0.040, 0.040, 1)
        self.assertAlmostEqual(grain.web, 0.020, places=15)
        # at e = L/2 the bore has zero length; only the two end annuli remain
        expected = 0.5 * math.pi * (0.100**2 - (0.040 + 0.040) ** 2)
        self.assertAlmostEqual(grain.burning_area(0.020), expected, places=14)
        self.assertEqual(grain.propellant_volume(0.020), 0.0)

    def test_volume_derivative_is_minus_area(self) -> None:
        """dV_p / de = -A_b for every grain, by central differences."""
        grains = [
            _bates(),
            _end_burner(),
            TubularGrain(0.02, 0.06, 0.3),
            TubularGrain(0.02, 0.06, 0.3, outer_burning=True),
        ]
        for grain in grains:
            for fraction in (0.1, 0.5, 0.9):
                e, h = fraction * grain.web, 1e-7
                slope = (
                    grain.propellant_volume(e + h) - grain.propellant_volume(e - h)
                ) / (2 * h)
                self.assertAlmostEqual(
                    -slope / grain.burning_area(e), 1.0, places=6, msg=str(grain)
                )

    def test_tubular_and_end_burner_formulas(self) -> None:
        """Bore burner A_b = 2 pi (r_i + e) L; end burner A_b = pi R^2 constant."""
        tube = TubularGrain(0.02, 0.06, 0.3)
        self.assertAlmostEqual(tube.web, 0.04)
        self.assertAlmostEqual(tube.burning_area(0.01), 2 * math.pi * 0.03 * 0.3)
        self.assertAlmostEqual(
            tube.propellant_volume(0.0), math.pi * 0.3 * (0.06**2 - 0.02**2)
        )
        both = TubularGrain(0.02, 0.06, 0.3, outer_burning=True)
        self.assertAlmostEqual(both.web, 0.02)
        self.assertAlmostEqual(both.burning_area(0.0), both.burning_area(0.015))
        self.assertAlmostEqual(both.burning_area(0.0), 2 * math.pi * 0.3 * 0.08)
        end = _end_burner()
        self.assertAlmostEqual(end.web, 0.04)
        self.assertAlmostEqual(end.burning_area(0.03), math.pi * 0.04**2)
        self.assertAlmostEqual(end.propellant_volume(0.01), math.pi * 0.04**2 * 0.03)
        self.assertEqual(end.burning_area(0.05), 0.0)

    def test_classification(self) -> None:
        """Neutral, progressive, regressive and the mixed BATES history."""
        self.assertEqual(_end_burner().classification(), "neutral")
        self.assertEqual(
            TubularGrain(0.02, 0.06, 0.3, outer_burning=True).classification(),
            "neutral",
        )
        self.assertEqual(TubularGrain(0.02, 0.06, 0.3).classification(), "progressive")
        self.assertEqual(_bates().classification(), "progressive-regressive")
        self.assertEqual(
            BatesGrain(0.100, 0.040, 0.040, 1).classification(), "regressive"
        )

    def test_invalid_grains_and_arguments(self) -> None:
        """Bad dimensions, negative web and bad classifier settings raise."""
        with self.assertRaises(ValueError):
            BatesGrain(0.03, 0.05, 0.1)
        with self.assertRaises(TypeError):
            BatesGrain(0.1, 0.03, 0.1, 1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TubularGrain(0.06, 0.02, 0.3)
        with self.assertRaises(ValueError):
            _bates().burning_area(-1e-9)
        with self.assertRaises(ValueError):
            _bates().classification(samples=2)
        with self.assertRaises(ValueError):
            _bates().classification(tolerance=1.5)


class NozzleTests(unittest.TestCase):
    """Ideal thrust coefficient against independent formulas."""

    @staticmethod
    def _area_ratio(mach: float, gamma: float) -> float:
        """Return A/A* from the area-Mach relation (independent form)."""
        t = 1.0 + 0.5 * (gamma - 1.0) * mach**2
        return (1.0 / mach) * ((2.0 / (gamma + 1.0)) * t) ** (
            (gamma + 1.0) / (2.0 * (gamma - 1.0))
        )

    def test_exit_mach_inverts_the_area_mach_relation(self) -> None:
        """A(M_e) reproduces eps, and M_e > 1 grows with eps."""
        previous = 1.0
        for eps in (1.0, 2.0, 8.0, 40.0, 400.0):
            mach = exit_mach_number(1.22, eps)
            self.assertAlmostEqual(self._area_ratio(mach, 1.22) / eps, 1.0, places=11)
            self.assertGreaterEqual(mach, previous)
            previous = mach
        self.assertEqual(exit_mach_number(1.3, 1.0), 1.0)

    def test_sonic_exit_closed_form(self) -> None:
        """For eps = 1: C_F = Gamma sqrt(2 g/(g+1)) + (2/(g+1))^(g/(g-1)) in vacuum."""
        for gamma in (1.2, 1.4):
            big_gamma = math.sqrt(gamma) * (2 / (gamma + 1)) ** (
                (gamma + 1) / (2 * (gamma - 1))
            )
            expected = big_gamma * math.sqrt(2 * gamma / (gamma + 1)) + (
                2 / (gamma + 1)
            ) ** (gamma / (gamma - 1))
            value = ideal_thrust_coefficient(gamma, 1.0, 5e6, 0.0)
            self.assertAlmostEqual(value, expected, places=12)
        self.assertAlmostEqual(ideal_thrust_coefficient(1.4, 1.0, 5e6, 0.0), 1.2679, 3)

    def test_infinite_expansion_limit(self) -> None:
        """C_F(eps -> inf, vacuum) -> Gamma sqrt(2 gamma / (gamma - 1))."""
        gamma = 1.4
        big_gamma = math.sqrt(gamma) * (2 / (gamma + 1)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        )
        limit = big_gamma * math.sqrt(2 * gamma / (gamma - 1))
        value = ideal_thrust_coefficient(gamma, 1e8, 5e6, 0.0)
        self.assertLess(value, limit)
        self.assertAlmostEqual(value / limit, 1.0, delta=2e-3)

    def test_optimum_expansion_at_matched_pressure(self) -> None:
        """For fixed p_a / p_c the C_F maximum is where p_e = p_a."""
        gamma, ratio = 1.22, 0.01
        root = brent_root(
            lambda eps: exit_pressure_ratio(gamma, eps) - ratio,
            1.0,
            1e4,
            tolerance=1e-10,
        )
        best = root.root
        pc, pa = 5e6, 5e6 * ratio
        peak = ideal_thrust_coefficient(gamma, best, pc, pa)
        self.assertGreater(peak, ideal_thrust_coefficient(gamma, 0.8 * best, pc, pa))
        self.assertGreater(peak, ideal_thrust_coefficient(gamma, 1.25 * best, pc, pa))
        # at matched pressure the pressure term vanishes
        pi_e = exit_pressure_ratio(gamma, best)
        self.assertAlmostEqual(pi_e, ratio, places=9)

    def test_invalid_nozzle_arguments(self) -> None:
        """Out-of-range gamma, area ratio and pressures raise."""
        with self.assertRaises(ValueError):
            exit_mach_number(1.0, 4.0)
        with self.assertRaises(ValueError):
            exit_pressure_ratio(1.2, 0.9)
        with self.assertRaises(ValueError):
            ideal_thrust_coefficient(1.2, 4.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            ideal_thrust_coefficient(1.2, 4.0, 1e6, -1.0)


class QuasiSteadyMotorTests(unittest.TestCase):
    """Quasi-steady motor against closed forms and independent quadrature."""

    def test_end_burner_closed_form(self) -> None:
        """Constant pressure, burn time L/r, I = C_F p A_t t_b, F_avg = C_F p A_t."""
        grain, motor = _end_burner(), _end_motor()
        result = simulate_solid_motor(grain, motor)
        a_t = motor.throat_area()
        pressure = equilibrium_pressure(
            motor.effective_coefficient,
            motor.pressure_exponent,
            motor.characteristic_velocity,
            motor.propellant_density,
            grain.burning_area(0.0) / a_t,
            reference_pressure=motor.reference_pressure,
        )
        rate = burn_rate(
            pressure,
            motor.effective_coefficient,
            motor.pressure_exponent,
            reference_pressure=motor.reference_pressure,
        )
        cf = ideal_thrust_coefficient(motor.gamma, motor.expansion_ratio, pressure)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "web_burnout")
        self.assertAlmostEqual(result.meop / pressure, 1.0, places=12)
        self.assertAlmostEqual(min(result.pressures) / pressure, 1.0, places=12)
        self.assertAlmostEqual(result.burn_time / (grain.web / rate), 1.0, places=9)
        self.assertAlmostEqual(
            result.total_impulse / (cf * pressure * a_t * result.burn_time),
            1.0,
            places=9,
        )
        self.assertAlmostEqual(result.average_thrust / (cf * pressure * a_t), 1.0, 9)
        self.assertAlmostEqual(result.webs_burned[-1], grain.web, places=12)
        self.assertEqual(result.action_time, result.burn_time)

    def test_end_burner_with_throat_erosion_closed_form(self) -> None:
        """Burn-out time with a linearly eroding throat from the exact integral.

        With A_t = pi R(t)^2, R = R_0 + eps t: de/dt = K pi^-m R^(-2m),
        m = n/(1-n), K = a p_ref^-n (a rho c* A_b p_ref^-n ... )^m; integrate and
        invert for e = L.
        """
        grain = _end_burner()
        motor = _end_motor(erosion_rate=2.0e-5)
        n = motor.pressure_exponent
        m = n / (1.0 - n)
        a = motor.effective_coefficient
        cc = a * motor.propellant_density * motor.characteristic_velocity
        cc *= grain.burning_area(0.0) / motor.reference_pressure**n
        k = a * motor.reference_pressure ** (-n) * cc**m * math.pi ** (-m)
        r0, eps = 0.5 * motor.throat_diameter, motor.erosion_rate
        power = 1.0 - 2.0 * m
        total = r0**power + grain.web * eps * power / k
        expected = (total ** (1.0 / power) - r0) / eps
        result = simulate_solid_motor(grain, motor, steps=400)
        self.assertAlmostEqual(result.burn_time / expected, 1.0, places=7)
        # erosion lowers the pressure: the peak is the initial equilibrium
        self.assertAlmostEqual(result.meop, result.pressures[0], places=6)
        self.assertGreater(result.pressures[0], result.pressures[-1] * 1.0001)

    def test_impulse_is_the_integral_of_thrust(self) -> None:
        """I equals the independent integral of F / r over the web, and C_F c* m_p."""
        grain, motor = _bates(), _motor()
        result = simulate_solid_motor(grain, motor, steps=400)
        a_t = motor.throat_area()

        def pressure_at(e: float) -> float:
            return motor.equilibrium_pressure(grain.burning_area(e))

        def rate_at(e: float) -> float:
            return burn_rate(
                pressure_at(e),
                motor.effective_coefficient,
                motor.pressure_exponent,
                reference_pressure=motor.reference_pressure,
            )

        def thrust_at(e: float) -> float:
            p = pressure_at(e)
            return a_t * p * ideal_thrust_coefficient(motor.gamma, 8.0, p)

        impulse = _simpson(lambda e: thrust_at(e) / rate_at(e), 0.0, grain.web)
        burn = _simpson(lambda e: 1.0 / rate_at(e), 0.0, grain.web)
        self.assertAlmostEqual(result.total_impulse / impulse, 1.0, delta=2e-5)
        self.assertAlmostEqual(result.burn_time / burn, 1.0, delta=2e-6)
        # trace consistency: the reported impulse is the trapezoid area of the trace
        area = sum(
            0.5 * (t1 - t0) * (f0 + f1)
            for t0, t1, f0, f1 in zip(
                result.times,
                result.times[1:],
                result.thrusts,
                result.thrusts[1:],
                strict=False,
            )
        )
        self.assertAlmostEqual(result.total_impulse, area, places=9)
        # I = integral C_F p A_t dt = C_F c* m_p in vacuum, where C_F is constant
        mass = 1750.0 * grain.propellant_volume(0.0)
        self.assertAlmostEqual(result.propellant_mass / mass, 1.0, places=12)
        cf = ideal_thrust_coefficient(motor.gamma, 8.0, 5e6)
        self.assertAlmostEqual(
            result.total_impulse / (cf * 1550.0 * mass), 1.0, delta=2e-5
        )
        self.assertAlmostEqual(
            result.specific_impulse, result.total_impulse / (mass * G0), places=9
        )

    def test_ambient_pressure_subtracts_exit_pressure_force(self) -> None:
        """With p_a > 0 the impulse drops by exactly p_a A_e t_b (end burner)."""
        grain = _end_burner()
        vacuum = simulate_solid_motor(grain, _end_motor())
        ground = simulate_solid_motor(grain, _end_motor(ambient_pressure=101325.0))
        motor = _end_motor()
        loss = 101325.0 * motor.exit_area * vacuum.burn_time
        self.assertAlmostEqual(
            (vacuum.total_impulse - ground.total_impulse) / loss, 1.0, places=9
        )

    def test_monotone_in_klemmung(self) -> None:
        """A smaller throat (larger Kn) gives a higher MEOP and a shorter burn."""
        grain = _bates()
        meops, burns, kns = [], [], []
        for diameter in (0.030, 0.026, 0.022, 0.018):
            motor = _motor(throat_diameter=diameter)
            result = simulate_solid_motor(grain, motor)
            meops.append(result.meop)
            burns.append(result.burn_time)
            kns.append(klemmung(grain.burning_area(0.0), motor.throat_area()))
        self.assertEqual(kns, sorted(kns))
        self.assertEqual(meops, sorted(meops))
        self.assertEqual(burns, sorted(burns, reverse=True))

    def test_step_refinement(self) -> None:
        """Halving the step changes the BATES impulse by less than 1e-4."""
        grain, motor = _bates(), _motor(erosion_rate=5e-5)
        coarse = simulate_solid_motor(grain, motor, steps=50)
        fine = simulate_solid_motor(grain, motor, steps=200)
        self.assertAlmostEqual(
            coarse.total_impulse / fine.total_impulse, 1.0, delta=1e-3
        )
        self.assertAlmostEqual(coarse.burn_time / fine.burn_time, 1.0, delta=1e-5)

    def test_step_limit_is_reported_not_hidden(self) -> None:
        """An exhausted step budget returns converged=False with a stop reason."""
        result = simulate_solid_motor(_bates(), _motor(), steps=100, max_steps=20)
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "step_limit")
        self.assertLess(result.webs_burned[-1], _bates().web)

    def test_invalid_simulation_arguments(self) -> None:
        """Bad mode, chamber, step counts and types raise."""
        with self.assertRaises(ValueError):
            simulate_solid_motor(_bates(), _motor(), mode="steady")
        with self.assertRaises(ValueError):
            simulate_solid_motor(_bates(), _motor(chamber_volume=0.004))
        with self.assertRaises(ValueError):
            simulate_solid_motor(_bates(), _motor(), steps=5)
        with self.assertRaises(ValueError):
            simulate_solid_motor(
                _bates(), _motor(), mode="transient", tail_fraction=1.0
            )
        with self.assertRaises(TypeError):
            simulate_solid_motor("grain", _motor())  # type: ignore[arg-type]


class TransientMotorTests(unittest.TestCase):
    """Gas mass balance ODE: steady state, tail-off and agreement with quasi-steady."""

    def test_steady_state_equals_closed_form_without_dilatation(self) -> None:
        """The mass balance plateau is the classical equilibrium pressure."""
        grain, motor = _end_burner(), _end_motor()
        result = simulate_solid_motor(
            grain, motor, mode="transient", dilatation=False, steps=100
        )
        expected = motor.equilibrium_pressure(grain.burning_area(0.0))
        plateau = [
            p
            for t, p in zip(result.times, result.pressures, strict=True)
            if 0.25 * result.burn_time < t < result.burn_time
        ]
        self.assertGreater(len(plateau), 10)
        for p in plateau:
            self.assertAlmostEqual(p / expected, 1.0, delta=2e-5)

    def test_steady_state_with_dilatation_shifts_by_gas_density_ratio(self) -> None:
        """Keeping the volume-growth term lowers p by about p / (rho_p R T_c (1-n))."""
        grain, motor = _end_burner(), _end_motor()
        result = simulate_solid_motor(grain, motor, mode="transient", steps=100)
        a_t = motor.throat_area()
        rt = chamber_rt(motor.characteristic_velocity, motor.gamma)
        area = grain.burning_area(0.0)

        def residual(p: float) -> float:
            rate = burn_rate(
                p,
                motor.effective_coefficient,
                motor.pressure_exponent,
                reference_pressure=motor.reference_pressure,
            )
            return (
                motor.propellant_density
                * area
                * rate
                * (1.0 - p / (motor.propellant_density * rt))
                - p * a_t / motor.characteristic_velocity
            )

        root = brent_root(residual, 1e5, 1e8, tolerance=1e-6).root
        classical = motor.equilibrium_pressure(area)
        mid = result.pressures[len(result.pressures) // 3]
        self.assertAlmostEqual(mid / root, 1.0, delta=2e-5)
        shift = 1.0 - root / classical
        estimate = classical / (motor.propellant_density * rt) / (1 - 0.35)
        self.assertAlmostEqual(shift / estimate, 1.0, delta=0.15)
        self.assertGreater(shift, 0.0)

    def test_tail_off_is_exponential_with_the_emptying_time(self) -> None:
        """After burn-out ln p falls with slope -R T_c A_t / (V c*)."""
        grain, motor = _bates(), _motor()
        result = simulate_solid_motor(grain, motor, mode="transient", steps=100)
        self.assertGreater(result.action_time, result.burn_time)
        rt = chamber_rt(motor.characteristic_velocity, motor.gamma)
        rate = (
            rt
            * motor.throat_area()
            / (motor.chamber_volume * motor.characteristic_velocity)
        )
        tail = [
            (t, p)
            for t, p in zip(result.times, result.pressures, strict=True)
            if t >= result.burn_time
        ]
        self.assertGreater(len(tail), 4)
        for (t0, p0), (t1, p1) in zip(tail, tail[1:], strict=False):
            self.assertAlmostEqual(
                math.log(p0 / p1) / (t1 - t0) / rate, 1.0, delta=1e-3
            )
        self.assertAlmostEqual(
            tail[-1][1] / result.meop, 0.05, delta=0.005
        )  # cut-off at tail_fraction * peak

    def test_agrees_with_quasi_steady(self) -> None:
        """Without dilatation the transient run matches quasi-steady to O(tau / t_b)."""
        grain, motor = _bates(), _motor(erosion_rate=5e-5)
        quasi = simulate_solid_motor(grain, motor, steps=200)
        full = simulate_solid_motor(
            grain, motor, mode="transient", steps=200, dilatation=False
        )
        self.assertTrue(full.converged)
        self.assertAlmostEqual(max(full.webs_burned), grain.web, places=12)
        self.assertAlmostEqual(full.meop / quasi.meop, 1.0, delta=1e-3)
        # the transient impulse adds the tail-off F_b tau (1 - f) beyond burn-out
        cut = full.times.index(full.burn_time)
        burn_phase = sum(
            0.5 * (t1 - t0) * (f0 + f1)
            for t0, t1, f0, f1 in zip(
                full.times[: cut + 1],
                full.times[1 : cut + 1],
                full.thrusts[: cut + 1],
                full.thrusts[1 : cut + 1],
                strict=False,
            )
        )
        self.assertAlmostEqual(burn_phase / quasi.total_impulse, 1.0, delta=2e-3)
        rt = chamber_rt(motor.characteristic_velocity, motor.gamma)
        a_t_end = motor.throat_area(full.burn_time)
        tau = motor.chamber_volume * motor.characteristic_velocity / (rt * a_t_end)
        tail = full.total_impulse - burn_phase
        remaining = 0.05 * full.meop / full.pressures[cut]
        estimate = full.thrusts[cut] * tau * (1.0 - remaining)
        self.assertAlmostEqual(tail / estimate, 1.0, delta=1.5e-2)
        self.assertAlmostEqual(full.burn_time / quasi.burn_time, 1.0, delta=5e-3)

    def test_dilatation_lowers_the_peak_by_the_gas_density_ratio(self) -> None:
        """Keeping -(p/V) dV/dt lowers MEOP by p / (rho_p R T_c (1 - n)) (0.5 %)."""
        grain, motor = _bates(), _motor(erosion_rate=5e-5)
        quasi = simulate_solid_motor(grain, motor, steps=200)
        full = simulate_solid_motor(grain, motor, mode="transient", steps=200)
        rt = chamber_rt(motor.characteristic_velocity, motor.gamma)
        shift = quasi.meop / (motor.propellant_density * rt) / (1 - 0.35)
        self.assertAlmostEqual(full.meop / quasi.meop, 1.0 - shift, delta=1e-3)
        self.assertLess(full.meop, quasi.meop)

    def test_startup_pressure_rises_monotonically(self) -> None:
        """From ignition the pressure rises to the plateau without overshoot."""
        result = simulate_solid_motor(_end_burner(), _end_motor(), mode="transient")
        reach = next(
            k for k, p in enumerate(result.pressures) if p >= (1 - 1e-3) * result.meop
        )
        rising = list(result.pressures[: reach + 1])
        self.assertEqual(rising, sorted(rising))
        self.assertAlmostEqual(result.pressures[0], 101325.0)
        at_burnout = result.pressures[result.times.index(result.burn_time)]
        self.assertAlmostEqual(at_burnout / result.meop, 1.0, places=9)


class ProbabilisticModelTests(unittest.TestCase):
    """The motor as a ProbabilisticModel: dispersion, Sobol, burst probability."""

    NOMINAL = _motor(erosion_rate=5e-5)

    def _model(self, burst: float = 7.4e6):  # type: ignore[no-untyped-def]
        return solid_motor_model(_bates(), self.NOMINAL, burst, steps=40)

    def test_model_structure_and_agreement_with_the_simulator(self) -> None:
        """Nine inputs, four QoIs; a mapping call equals a direct simulation."""
        model = self._model()
        self.assertEqual(
            model.inputs.names,
            (
                "a0",
                "n",
                "c_star",
                "rho_p",
                "sigma_p",
                "T_grain",
                "d_throat",
                "erosion_rate",
                "p_burst",
            ),
        )
        self.assertEqual(
            tuple(q.name for q in model.quantities),
            ("meop", "total_impulse", "burn_time", "burst_margin"),
        )
        point = {
            "a0": 8.0e-3,
            "n": 0.35,
            "c_star": 1550.0,
            "rho_p": 1750.0,
            "sigma_p": 2.5e-3,
            "T_grain": 293.15,
            "d_throat": 0.024,
            "erosion_rate": 5e-5,
            "p_burst": 7.4e6,
        }
        out = model.evaluate(point)
        direct = simulate_solid_motor(_bates(), self.NOMINAL, steps=40)
        self.assertAlmostEqual(out["meop"], direct.meop, places=6)
        self.assertAlmostEqual(out["total_impulse"], direct.total_impulse, places=6)
        self.assertAlmostEqual(out["burn_time"], direct.burn_time, places=12)
        self.assertAlmostEqual(out["burst_margin"], 7.4e6 - direct.meop, places=6)

    def test_nominal_without_uncertain_fields_is_refused(self) -> None:
        """Uncertain inputs with relative spreads need a positive nominal."""
        with self.assertRaises(ValueError):
            solid_motor_model(_bates(), _motor(erosion_rate=0.0), 7e6)
        with self.assertRaises(ValueError):
            solid_motor_model(_bates(), _motor(temperature_sensitivity=0.0), 7e6)
        with self.assertRaises(ValueError):
            solid_motor_model(_bates(), self.NOMINAL, -1.0)
        with self.assertRaises(ValueError):
            solid_motor_model(_bates(), self.NOMINAL, 7e6, erosion_spread=1.0)

    def test_monte_carlo_dispersion_of_total_impulse(self) -> None:
        """The impulse CoV is sqrt(cov_c*^2 + cov_rho^2) = 1.41 % to first order.

        I = C_F c* m_p with m_p = rho_p V_p, so only c* and rho_p move it
        materially. The sample standard deviation of N draws has relative
        standard error 1/sqrt(2 (N - 1)); the band is five of those plus 5 %
        for the small C_F and erosion contributions.
        """
        model = self._model()
        count = 300
        draws = model.inputs.sample(count, seed=11)
        impulses = [model.evaluate(x)["total_impulse"] for x in draws]
        mean, sd = statistics.fmean(impulses), statistics.stdev(impulses)
        expected = math.hypot(0.01, 0.01)
        band = 5.0 / math.sqrt(2.0 * (count - 1)) + 0.05
        self.assertAlmostEqual(sd / mean / expected, 1.0, delta=band)
        nominal = simulate_solid_motor(_bates(), self.NOMINAL, steps=40).total_impulse
        self.assertAlmostEqual(
            mean / nominal, 1.0, delta=5.0 * sd / mean / math.sqrt(count) + 2e-3
        )

    def test_sobol_ranking(self) -> None:
        """Impulse is driven by rho_p and c*; burn time by the burn-rate inputs."""
        model = self._model()
        result = sobol_analysis_all(
            model, 64, quantities=("total_impulse", "burn_time"), design="sobol", seed=5
        )
        impulse = result["total_impulse"].as_mapping()
        top_two = sorted(impulse, key=lambda k: impulse[k][1], reverse=True)[:2]
        self.assertEqual(set(top_two), {"rho_p", "c_star"})
        self.assertLess(impulse["a0"][1], 0.15)
        burn = result["burn_time"].as_mapping()
        ranked = sorted(burn, key=lambda k: burn[k][1], reverse=True)
        self.assertEqual(set(ranked[:2]), {"T_grain", "a0"})
        self.assertLess(burn["rho_p"][1], 0.15)
        self.assertEqual(result.evaluations, 64 * (model.dimension + 2))

    def test_burst_probability_subset_simulation_against_crude_monte_carlo(
        self,
    ) -> None:
        """P(MEOP > burst) by subset simulation matches crude MC within its error."""
        model = self._model(7.8e6)
        limit = ModelLimitState(model, "burst_margin", 0.0, failure="below")
        crude = model_monte_carlo(limit, 1500, seed=21)
        subset = subset_simulation(limit, samples_per_level=300, seed=22)
        self.assertTrue(subset.converged)
        self.assertGreater(crude.probability, 0.01)
        self.assertGreaterEqual(subset.levels, 2)
        self.assertLess(subset.evaluations, crude.evaluations)
        spread = math.hypot(
            crude.standard_error,
            subset.coefficient_of_variation * crude.probability,
        )
        self.assertLess(abs(subset.probability - crude.probability), 4.0 * spread)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowBurstProbabilityTests(unittest.TestCase):
    """Rare burst event: subset simulation needs far fewer runs than crude MC."""

    def test_rare_burst_probability(self) -> None:
        """A p ~ 1e-3 event: subset simulation vs a 12000-run crude reference."""
        model = solid_motor_model(_bates(), _motor(erosion_rate=5e-5), 8.7e6, steps=40)
        limit = ModelLimitState(model, "burst_margin", 0.0, failure="below")
        start = time.perf_counter()
        crude = model_monte_carlo(limit, 12000, seed=31)
        crude_seconds = time.perf_counter() - start
        start = time.perf_counter()
        subset = subset_simulation(limit, samples_per_level=500, seed=32)
        subset_seconds = time.perf_counter() - start
        print(
            f"\ncrude p={crude.probability:.5f} se={crude.standard_error:.5f} "
            f"evals={crude.evaluations} t={crude_seconds:.1f}s; "
            f"subset p={subset.probability:.5f} "
            f"cov={subset.coefficient_of_variation:.3f} "
            f"evals={subset.evaluations} t={subset_seconds:.1f}s"
        )
        self.assertLess(subset.evaluations, crude.evaluations)
        spread = math.hypot(
            crude.standard_error, subset.coefficient_of_variation * crude.probability
        )
        self.assertLess(
            abs(subset.probability - crude.probability), 4.0 * spread + 1e-4
        )


if __name__ == "__main__":
    unittest.main()
