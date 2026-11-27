"""Tests for the ideal-rocket liquid-engine model and its UQ wiring."""

from __future__ import annotations

import math
import unittest

from probstat.aerospace.liquid_engine import (
    DEMONSTRATION_MIXTURE_TABLE,
    G0,
    UNIVERSAL_GAS_CONSTANT,
    MixtureRatioTable,
    PumpSpec,
    area_ratio_from_mach,
    characteristic_velocity,
    divergence_factor,
    exit_mach_from_area_ratio,
    exit_pressure_ratio,
    expansion_ratio_from_pressure_ratio,
    ideal_rocket_performance,
    limiting_thrust_coefficient,
    liquid_engine_model,
    power_balance,
    pump_power,
    separation_margin,
    thrust_coefficient,
    turbine_specific_work,
)
from probstat.distributions import Normal, TruncatedNormal, Uniform
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis


def _bisect_mach(g: float, eps: float) -> float:
    """Independent supersonic area-Mach root by plain bisection."""
    lo, hi = 1.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_ratio_from_mach(g, mid) < eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class AreaMachTests(unittest.TestCase):
    """Closed forms of the isentropic relations for gamma = 1.4."""

    def test_area_ratio_closed_values(self) -> None:
        """A/A* is 1 at M=1, 1.6875 at M=2, 25 at M=5 and 1.3398 at M=0.5."""
        self.assertAlmostEqual(area_ratio_from_mach(1.4, 1.0), 1.0, places=14)
        self.assertAlmostEqual(area_ratio_from_mach(1.4, 2.0), 1.6875, places=12)
        self.assertAlmostEqual(area_ratio_from_mach(1.4, 5.0), 25.0, places=11)
        self.assertAlmostEqual(
            area_ratio_from_mach(1.4, 0.5), 2.0 * 0.875**3, places=12
        )
        self.assertAlmostEqual(area_ratio_from_mach(1.4, 3.0), 4.2345679012, places=9)

    def test_root_recovers_mach(self) -> None:
        """Solving A/A* = 1.6875, 4.2345679..., 25 returns M = 2, 3, 5."""
        for eps, mach in ((1.6875, 2.0), (4.234567901234568, 3.0), (25.0, 5.0)):
            solution = exit_mach_from_area_ratio(1.4, eps)
            self.assertTrue(solution.converged)
            self.assertAlmostEqual(solution.mach, mach, places=9)
            self.assertLess(abs(solution.residual), 1e-12)

    def test_matches_independent_bisection_across_gamma(self) -> None:
        """Newton agrees with bisection for gamma and epsilon sweeps."""
        for g in (1.05, 1.15, 1.25, 1.4, 1.67):
            for eps in (1.0001, 1.5, 10.0, 80.0, 400.0, 1e5):
                solution = exit_mach_from_area_ratio(g, eps)
                self.assertTrue(solution.converged, (g, eps))
                self.assertAlmostEqual(solution.mach, _bisect_mach(g, eps), places=7)
                self.assertGreater(solution.mach, 1.0)

    def test_unit_expansion_is_sonic_and_exact(self) -> None:
        """Epsilon = 1 returns M = 1 without iterating."""
        solution = exit_mach_from_area_ratio(1.3, 1.0)
        self.assertEqual((solution.mach, solution.iterations), (1.0, 0))
        self.assertEqual(solution.stop_reason, "exact")

    def test_failure_status_is_reported_not_hidden(self) -> None:
        """One iteration cannot reach 1e-13, and the result says so."""
        solution = exit_mach_from_area_ratio(1.2, 50.0, max_iterations=1)
        self.assertFalse(solution.converged)
        self.assertEqual(solution.stop_reason, "max_iterations")

    def test_pressure_ratio_closed_values(self) -> None:
        """p/p0 is 0.52828 at M=1 and 1.8^-3.5 at M=2 for gamma 1.4."""
        self.assertAlmostEqual(exit_pressure_ratio(1.4, 1.0), 1.2**-3.5, places=14)
        self.assertAlmostEqual(exit_pressure_ratio(1.4, 2.0), 1.8**-3.5, places=14)
        self.assertAlmostEqual(exit_pressure_ratio(1.4, 2.0), 0.12780, places=5)
        self.assertEqual(exit_pressure_ratio(1.4, 0.0), 1.0)

    def test_pressure_ratio_inverse(self) -> None:
        """expansion_ratio_from_pressure_ratio inverts the Mach route."""
        for g in (1.15, 1.4):
            for eps in (3.0, 40.0, 250.0):
                mach = exit_mach_from_area_ratio(g, eps).mach
                ratio = exit_pressure_ratio(g, mach)
                self.assertAlmostEqual(
                    expansion_ratio_from_pressure_ratio(g, ratio) / eps, 1.0, places=8
                )
        self.assertEqual(expansion_ratio_from_pressure_ratio(1.2, 1.0), 1.0)

    def test_invalid_input(self) -> None:
        """Domain violations raise TypeError or ValueError."""
        with self.assertRaises(ValueError):
            exit_mach_from_area_ratio(1.4, 0.9)
        with self.assertRaises(ValueError):
            exit_mach_from_area_ratio(1.0, 5.0)
        with self.assertRaises(ValueError):
            exit_mach_from_area_ratio(1.4, math.nan)
        with self.assertRaises(TypeError):
            exit_mach_from_area_ratio(1.4, "5")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            exit_mach_from_area_ratio(1.4, 5.0, max_iterations=2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            area_ratio_from_mach(1.4, 0.0)
        with self.assertRaises(ValueError):
            expansion_ratio_from_pressure_ratio(1.4, 1.5)


class CharacteristicVelocityTests(unittest.TestCase):
    """c* against the choked-flow mass-flow formula."""

    def test_matches_choked_mass_flow(self) -> None:
        """c* = p A / mdot with mdot from the sonic-throat formula."""
        g, weight, temperature = 1.22, 21.0, 3300.0
        gas = UNIVERSAL_GAS_CONSTANT / weight
        pc, area = 5.0e6, 0.02
        mdot = (
            area
            * pc
            * math.sqrt(g / (gas * temperature))
            * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        )
        self.assertAlmostEqual(
            characteristic_velocity(g, weight, temperature) / (pc * area / mdot),
            1.0,
            places=12,
        )

    def test_scaling_and_range(self) -> None:
        """c* scales as sqrt(T/M) and is in the physical few-km/s range."""
        base = characteristic_velocity(1.2, 16.0, 3000.0)
        self.assertAlmostEqual(
            characteristic_velocity(1.2, 16.0, 12000.0) / base, 2.0, places=12
        )
        self.assertAlmostEqual(
            characteristic_velocity(1.2, 4.0, 3000.0) / base, 2.0, places=12
        )
        self.assertTrue(1000.0 < base < 3000.0)


class ThrustCoefficientTests(unittest.TestCase):
    """CF limits, optimum expansion and monotonicity."""

    def test_limiting_coefficient_closed_form(self) -> None:
        """The infinite-expansion limit has the stated closed form."""
        for g in (1.1, 1.2, 1.4):
            expected = math.sqrt(
                2.0 * g * g / (g - 1.0) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
            )
            self.assertAlmostEqual(limiting_thrust_coefficient(g), expected, places=14)
        self.assertAlmostEqual(limiting_thrust_coefficient(1.2), 2.2466, places=4)

    def test_vacuum_cf_increases_toward_the_limit(self) -> None:
        """Vacuum CF rises with epsilon and stays below the limit."""
        g = 1.2
        limit = limiting_thrust_coefficient(g)
        previous = 0.0
        for eps in (2.0, 10.0, 50.0, 400.0, 1e4, 1e8, 1e14):
            cf = thrust_coefficient(g, eps, 5e6, 0.0)
            self.assertGreater(cf, previous)
            self.assertLess(cf, limit)
            previous = cf
        self.assertLess(limit - previous, 5e-3)

    def test_optimum_expansion_equalizes_pressures(self) -> None:
        """CF is maximal where p_e = p_a (ideal expansion)."""
        g, pc, pa = 1.2, 7.0e6, 101325.0
        eps_opt = expansion_ratio_from_pressure_ratio(g, pa / pc)
        best = thrust_coefficient(g, eps_opt, pc, pa)
        for factor in (0.7, 0.9, 1.1, 1.4):
            self.assertLess(thrust_coefficient(g, eps_opt * factor, pc, pa), best)

    def test_pressure_term_and_ambient_monotone(self) -> None:
        """CF falls linearly with p_a at slope -epsilon/p_c."""
        g, eps, pc = 1.25, 30.0, 6.0e6
        cf0 = thrust_coefficient(g, eps, pc, 0.0)
        cf1 = thrust_coefficient(g, eps, pc, 1.0e5)
        cf2 = thrust_coefficient(g, eps, pc, 2.0e5)
        self.assertAlmostEqual(cf0 - cf1, 1.0e5 * eps / pc, places=12)
        self.assertAlmostEqual(cf1 - cf2, 1.0e5 * eps / pc, places=12)

    def test_divergence_factor(self) -> None:
        """Lambda = (1 + cos a)/2 and multiplies only the momentum term."""
        self.assertEqual(divergence_factor(0.0), 1.0)
        alpha = math.radians(15.0)
        self.assertAlmostEqual(divergence_factor(alpha), 0.5 * (1 + math.cos(alpha)))
        self.assertAlmostEqual(divergence_factor(alpha), 0.98296291, places=8)
        g, eps, pc, pa = 1.2, 20.0, 5e6, 1e5
        lam = divergence_factor(alpha)
        full = thrust_coefficient(g, eps, pc, pa)
        part = thrust_coefficient(g, eps, pc, pa, divergence=lam)
        ratio = exit_pressure_ratio(g, exit_mach_from_area_ratio(g, eps).mach)
        momentum = full - (ratio - pa / pc) * eps
        self.assertAlmostEqual(full - part, (1.0 - lam) * momentum, places=12)
        with self.assertRaises(ValueError):
            divergence_factor(0.5 * math.pi)
        with self.assertRaises(ValueError):
            divergence_factor(-0.1)


class PerformanceTests(unittest.TestCase):
    """Consistency, corrections, and a worked example from the formulas."""

    PARAMS = {
        "gamma": 1.25,
        "molecular_weight": 21.0,
        "chamber_temperature": 3500.0,
        "chamber_pressure": 6.0e6,
        "throat_area": 0.25 * math.pi * 0.1**2,
        "expansion_ratio": 40.0,
    }

    def test_worked_example_by_momentum_balance(self) -> None:
        """Thrust equals mdot*ve + (p_e - p_a) A_e from independent formulas."""
        p = self.PARAMS
        g, weight, tc = p["gamma"], p["molecular_weight"], p["chamber_temperature"]
        pc, at, eps = p["chamber_pressure"], p["throat_area"], p["expansion_ratio"]
        pa = 101325.0
        gas = UNIVERSAL_GAS_CONSTANT / weight
        mach = _bisect_mach(g, eps)
        ratio = (1.0 + 0.5 * (g - 1.0) * mach**2) ** (-g / (g - 1.0))
        pe = ratio * pc
        ve = math.sqrt(
            2.0 * g / (g - 1.0) * gas * tc * (1.0 - ratio ** ((g - 1.0) / g))
        )
        mdot = (
            at
            * pc
            * math.sqrt(g / (gas * tc))
            * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        )
        thrust = mdot * ve + (pe - pa) * eps * at
        result = ideal_rocket_performance(**p, ambient_pressure=pa)
        self.assertAlmostEqual(result.thrust / thrust, 1.0, places=8)
        self.assertAlmostEqual(result.mass_flow / mdot, 1.0, places=11)
        self.assertAlmostEqual(result.exit_pressure / pe, 1.0, places=8)
        self.assertAlmostEqual(result.specific_impulse, thrust / (mdot * G0), places=5)
        vac = mdot * ve + pe * eps * at
        self.assertAlmostEqual(result.vacuum_thrust / vac, 1.0, places=8)

    def test_isp_cf_cstar_consistency(self) -> None:
        """Isp = CF c*/g0 and F = mdot Isp g0 = CF p_c A_t."""
        result = ideal_rocket_performance(**self.PARAMS, ambient_pressure=1.0e5)
        self.assertAlmostEqual(
            result.specific_impulse,
            result.thrust_coefficient * result.characteristic_velocity / G0,
            places=9,
        )
        self.assertAlmostEqual(
            result.thrust,
            result.mass_flow * result.specific_impulse * G0,
            delta=1e-6 * result.thrust,
        )
        self.assertAlmostEqual(
            result.thrust,
            result.thrust_coefficient
            * self.PARAMS["chamber_pressure"]
            * self.PARAMS["throat_area"],
            delta=1e-9 * result.thrust,
        )

    def test_sea_level_below_vacuum(self) -> None:
        """Sea-level Isp is below vacuum Isp; vacuum call equals vacuum field."""
        sea = ideal_rocket_performance(**self.PARAMS, ambient_pressure=101325.0)
        vac = ideal_rocket_performance(**self.PARAMS, ambient_pressure=0.0)
        self.assertLess(sea.specific_impulse, sea.vacuum_specific_impulse)
        self.assertAlmostEqual(
            vac.specific_impulse, sea.vacuum_specific_impulse, places=9
        )
        self.assertAlmostEqual(vac.thrust, sea.vacuum_thrust, delta=1e-9 * vac.thrust)

    def test_combustion_efficiency_scaling(self) -> None:
        """eta_c* scales Isp and c* linearly, mass flow inversely, F not at all."""
        one = ideal_rocket_performance(**self.PARAMS, ambient_pressure=1e5)
        low = ideal_rocket_performance(
            **self.PARAMS, ambient_pressure=1e5, combustion_efficiency=0.95
        )
        self.assertAlmostEqual(low.specific_impulse / one.specific_impulse, 0.95)
        self.assertAlmostEqual(low.mass_flow / one.mass_flow, 1.0 / 0.95)
        self.assertAlmostEqual(low.thrust / one.thrust, 1.0)
        self.assertAlmostEqual(
            low.characteristic_velocity, 0.95 * low.ideal_characteristic_velocity
        )

    def test_divergence_loss_reduces_isp(self) -> None:
        """A 15 degree cone loses momentum thrust by lambda only."""
        base = ideal_rocket_performance(**self.PARAMS, ambient_pressure=0.0)
        cone = ideal_rocket_performance(
            **self.PARAMS, ambient_pressure=0.0, divergence_half_angle=math.radians(15)
        )
        self.assertLess(cone.vacuum_specific_impulse, base.vacuum_specific_impulse)
        self.assertGreater(
            cone.vacuum_specific_impulse, 0.97 * base.vacuum_specific_impulse
        )
        self.assertAlmostEqual(cone.divergence_factor, 0.98296291, places=8)

    def test_monotonicity_in_expansion_ratio(self) -> None:
        """Vacuum Isp increases with epsilon; exit pressure decreases."""
        isp, pe = [], []
        for eps in (5.0, 15.0, 40.0, 100.0):
            params = dict(self.PARAMS, expansion_ratio=eps)
            r = ideal_rocket_performance(**params)
            isp.append(r.vacuum_specific_impulse)
            pe.append(r.exit_pressure)
        self.assertEqual(isp, sorted(isp))
        self.assertEqual(pe, sorted(pe, reverse=True))

    def test_chamber_pressure_scaling(self) -> None:
        """Thrust is linear in p_c in vacuum; vacuum Isp does not depend on p_c."""
        a = ideal_rocket_performance(**self.PARAMS)
        b = ideal_rocket_performance(**dict(self.PARAMS, chamber_pressure=1.2e7))
        self.assertAlmostEqual(b.vacuum_thrust / a.vacuum_thrust, 2.0, places=12)
        self.assertAlmostEqual(
            b.vacuum_specific_impulse / a.vacuum_specific_impulse, 1.0, places=12
        )

    def test_summerfield_flag(self) -> None:
        """Separation is flagged exactly when p_e < 0.4 p_a."""
        self.assertAlmostEqual(separation_margin(30e3, 101325.0), 30e3 - 0.4 * 101325.0)
        self.assertEqual(separation_margin(5e3, 0.0), 5e3)
        over = ideal_rocket_performance(**self.PARAMS, ambient_pressure=101325.0)
        self.assertTrue(over.separation_predicted)
        self.assertLess(over.separation_margin, 0.0)
        pe = over.exit_pressure
        for pa, expected in ((pe / 0.4 * 0.999, False), (pe / 0.4 * 1.001, True)):
            r = ideal_rocket_performance(**self.PARAMS, ambient_pressure=pa)
            self.assertEqual(r.separation_predicted, expected)
        vac = ideal_rocket_performance(**self.PARAMS)
        self.assertFalse(vac.separation_predicted)

    def test_failure_cases(self) -> None:
        """Bad inputs raise rather than producing numbers."""
        bad = (
            {"gamma": 1.0},
            {"molecular_weight": -1.0},
            {"chamber_temperature": 0.0},
            {"chamber_pressure": math.inf},
            {"throat_area": -1e-3},
            {"expansion_ratio": 0.5},
            {"ambient_pressure": -1.0},
            {"combustion_efficiency": 1.2},
            {"combustion_efficiency": 0.0},
            {"divergence_half_angle": 2.0},
        )
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ideal_rocket_performance(**dict(self.PARAMS, **change))
        with self.assertRaises(TypeError):
            ideal_rocket_performance(**dict(self.PARAMS, gamma=True))  # type: ignore[arg-type]


class MixtureTableTests(unittest.TestCase):
    """Interpolation of the O/F table."""

    def _table(self, method: str = "linear") -> MixtureRatioTable:
        """Return a small table with a temperature peak."""
        return MixtureRatioTable(
            (1.0, 2.0, 4.0),
            (2000.0, 3000.0, 2800.0),
            (1.3, 1.2, 1.1),
            (10.0, 14.0, 22.0),
            method=method,  # type: ignore[arg-type]
        )

    def test_linear_nodes_and_midpoints(self) -> None:
        """Linear interpolation hits nodes and midpoints exactly."""
        table = self._table()
        state = table.at(2.0)
        self.assertEqual(
            (state.chamber_temperature, state.gamma, state.molecular_weight),
            (3000.0, 1.2, 14.0),
        )
        mid = table.at(3.0)
        self.assertAlmostEqual(mid.chamber_temperature, 2900.0)
        self.assertAlmostEqual(mid.gamma, 1.15)
        self.assertAlmostEqual(mid.molecular_weight, 18.0)

    def test_pchip_nodes_exact_and_no_overshoot(self) -> None:
        """PCHIP passes through nodes and never exceeds the data range."""
        table = self._table("pchip")
        for of, tc in ((1.0, 2000.0), (2.0, 3000.0), (4.0, 2800.0)):
            self.assertAlmostEqual(table.at(of).chamber_temperature, tc, places=9)
        grid = [1.0 + 3.0 * k / 300 for k in range(301)]
        values = [table.at(of).chamber_temperature for of in grid]
        self.assertLessEqual(max(values), 3000.0 + 1e-9)
        self.assertGreaterEqual(min(values), 2000.0 - 1e-9)
        molecular = [table.at(of).molecular_weight for of in grid]
        self.assertEqual(molecular, sorted(molecular))

    def test_pchip_reproduces_linear_data(self) -> None:
        """Collinear data are reproduced exactly by PCHIP."""
        table = MixtureRatioTable(
            (1.0, 2.0, 3.5, 5.0),
            (2000.0, 2100.0, 2250.0, 2400.0),
            (1.2, 1.2, 1.2, 1.2),
            (10.0, 10.0, 10.0, 10.0),
        )
        self.assertAlmostEqual(table.at(2.7).chamber_temperature, 2170.0, places=9)

    def test_out_of_range_and_invalid_tables(self) -> None:
        """No extrapolation; malformed tables are rejected."""
        table = self._table()
        for of in (0.99, 4.01):
            with self.assertRaises(ValueError):
                table.at(of)
        with self.assertRaises(ValueError):
            MixtureRatioTable((1.0, 1.0), (1.0, 2.0), (1.2, 1.2), (10.0, 10.0))
        with self.assertRaises(ValueError):
            MixtureRatioTable((1.0, 2.0), (1.0,), (1.2, 1.2), (10.0, 10.0))
        with self.assertRaises(ValueError):
            MixtureRatioTable((1.0, 2.0), (1.0, 2.0), (1.0, 1.2), (10.0, 10.0))
        with self.assertRaises(ValueError):
            MixtureRatioTable((1.0,), (1.0,), (1.2,), (10.0,))
        with self.assertRaises(ValueError):
            MixtureRatioTable(
                (1.0, 2.0),
                (1.0, 2.0),
                (1.2, 1.2),
                (10.0, 10.0),
                method="spline",  # type: ignore[arg-type]
            )

    def test_demonstration_table_is_marked_illustrative(self) -> None:
        """The bundled table says it is synthetic and not CEA output."""
        label = DEMONSTRATION_MIXTURE_TABLE.label
        self.assertIn("ILLUSTRATIVE", label)
        self.assertIn("Not CEA", label)
        low, high = DEMONSTRATION_MIXTURE_TABLE.bounds
        self.assertEqual((low, high), (1.0, 4.0))
        self.assertGreater(DEMONSTRATION_MIXTURE_TABLE.at(2.8).gamma, 1.0)


class PowerBalanceTests(unittest.TestCase):
    """Turbine power = pump power, simplified."""

    def test_hand_computed_balance(self) -> None:
        """Power and flow fraction match the closed formulas."""
        pump = PumpSpec(
            mass_flow=100.0, density=1000.0, pressure_rise=1.0e7, efficiency=0.7
        )
        self.assertAlmostEqual(pump_power(pump), 100.0 * 1.0e7 / (1000.0 * 0.7))
        work = turbine_specific_work(
            specific_heat=3000.0,
            inlet_temperature=900.0,
            pressure_ratio=20.0,
            gamma=1.2,
            efficiency=0.6,
        )
        self.assertAlmostEqual(
            work, 0.6 * 3000.0 * 900.0 * (1.0 - 20.0 ** (-0.2 / 1.2))
        )
        balance = power_balance(
            [pump, pump],
            400.0,
            specific_heat=3000.0,
            inlet_temperature=900.0,
            pressure_ratio=20.0,
            gamma=1.2,
            turbine_efficiency=0.6,
        )
        self.assertAlmostEqual(balance.pump_power, 2.0 * pump_power(pump))
        self.assertAlmostEqual(balance.turbine_mass_flow * work, balance.pump_power)
        self.assertAlmostEqual(balance.flow_fraction, balance.turbine_mass_flow / 400.0)
        self.assertTrue(balance.feasible)

    def test_monotone_in_pump_rise_and_infeasible_flag(self) -> None:
        """Flow fraction grows with pump pressure rise; fraction > 1 is flagged."""
        fractions = []
        for rise in (5e6, 1e7, 2e7):
            pump = PumpSpec(100.0, 1000.0, rise, 0.7)
            fractions.append(
                power_balance(
                    [pump],
                    400.0,
                    specific_heat=3000.0,
                    inlet_temperature=900.0,
                    pressure_ratio=20.0,
                    gamma=1.2,
                    turbine_efficiency=0.6,
                ).flow_fraction
            )
        self.assertEqual(fractions, sorted(fractions))
        heavy = power_balance(
            [PumpSpec(100.0, 1000.0, 1e9, 0.7)],
            10.0,
            specific_heat=3000.0,
            inlet_temperature=900.0,
            pressure_ratio=20.0,
            gamma=1.2,
            turbine_efficiency=0.6,
        )
        self.assertFalse(heavy.feasible)

    def test_failure_cases(self) -> None:
        """Invalid inputs raise."""
        with self.assertRaises(ValueError):
            power_balance(
                [],
                10.0,
                specific_heat=1.0,
                inlet_temperature=1.0,
                pressure_ratio=2.0,
                gamma=1.2,
                turbine_efficiency=0.5,
            )
        with self.assertRaises(TypeError):
            pump_power((1.0, 2.0, 3.0, 0.5))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pump_power(PumpSpec(1.0, 1000.0, 1e6, 1.5))
        with self.assertRaises(ValueError):
            turbine_specific_work(
                specific_heat=3000.0,
                inlet_temperature=900.0,
                pressure_ratio=1.0,
                gamma=1.2,
                efficiency=0.6,
            )


def _uq_model(**overrides: object):  # noqa: ANN202
    """Build the small UQ test model (illustrative synthetic engine)."""
    config: dict[str, object] = {
        "chamber_pressure": Normal(7.0e6, 2.0e5),
        "mixture_ratio": Uniform(2.2, 3.4),
        "throat_diameter": Normal(0.1, 0.0005),
        "expansion_ratio": Normal(40.0, 1.0),
        "combustion_efficiency": TruncatedNormal(0.97, 0.01, upper=1.0),
    }
    config.update(overrides)
    return liquid_engine_model(**config)  # type: ignore[arg-type]


class ModelTests(unittest.TestCase):
    """The ProbabilisticModel and its UQ workflows."""

    def test_nominal_evaluation_matches_direct_call(self) -> None:
        """The model reproduces ideal_rocket_performance at the same point."""
        model = liquid_engine_model(
            chamber_pressure=Normal(7e6, 1e5),
            mixture_ratio=2.8,
            throat_diameter=0.1,
            expansion_ratio=40.0,
        )
        self.assertEqual(model.inputs.names, ("chamber_pressure",))
        out = model.evaluate((7.0e6,))
        state = DEMONSTRATION_MIXTURE_TABLE.at(2.8)
        direct = ideal_rocket_performance(
            gamma=state.gamma,
            molecular_weight=state.molecular_weight,
            chamber_temperature=state.chamber_temperature,
            chamber_pressure=7.0e6,
            throat_area=0.25 * math.pi * 0.1**2,
            expansion_ratio=40.0,
            ambient_pressure=101325.0,
        )
        self.assertAlmostEqual(out["thrust"], direct.thrust, places=6)
        self.assertAlmostEqual(out["specific_impulse"], direct.specific_impulse)
        self.assertAlmostEqual(out["mass_flow"], direct.mass_flow)
        self.assertSetEqual(
            set(model.quantity_names),
            {
                "thrust",
                "specific_impulse",
                "vacuum_specific_impulse",
                "mass_flow",
                "exit_pressure",
                "separation_margin",
                "characteristic_velocity",
            },
        )

    def test_construction_errors(self) -> None:
        """Bad configuration is rejected."""
        with self.assertRaises(ValueError):
            liquid_engine_model(
                chamber_pressure=7e6,
                mixture_ratio=2.8,
                throat_diameter=0.1,
                expansion_ratio=40.0,
            )
        with self.assertRaises(TypeError):
            liquid_engine_model(
                chamber_pressure=True,
                mixture_ratio=Uniform(2.0, 3.0),  # type: ignore[arg-type]
                throat_diameter=0.1,
                expansion_ratio=40.0,
            )
        with self.assertRaises(TypeError):
            liquid_engine_model(
                chamber_pressure=7e6,
                mixture_ratio=Uniform(2.0, 3.0),
                throat_diameter=0.1,
                expansion_ratio=40.0,
                table="demo",  # type: ignore[arg-type]
            )

    def test_mixture_ratio_outside_table_fails_the_run(self) -> None:
        """An O/F draw outside the table is a run failure, not an extrapolation."""
        model = _uq_model(mixture_ratio=Uniform(0.5, 3.0))
        with self.assertRaises(Exception) as caught:  # noqa: B017
            model.evaluate((7.0e6, 0.5, 0.97, 0.1, 40.0))
        self.assertIn("outside the table range", str(caught.exception))

    def test_sobol_indices_match_physics(self) -> None:
        """Vacuum Isp is exactly independent of p_c and d_t; O/F dominates."""
        result = sobol_analysis(_uq_model(), "vacuum_specific_impulse", 256, seed=1)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        total = {
            n: i.estimate
            for n, i in zip(result.input_names, result.total_effect, strict=True)
        }
        self.assertLess(abs(total["chamber_pressure"]), 1e-9)
        self.assertLess(abs(total["throat_diameter"]), 1e-9)
        self.assertGreater(first["mixture_ratio"], 0.7)
        self.assertGreater(first["combustion_efficiency"], 0.05)
        self.assertAlmostEqual(sum(first.values()), 1.0, delta=0.05)
        thrust = sobol_analysis(_uq_model(), "thrust", 256, seed=1)
        total_thrust = {
            n: i.estimate
            for n, i in zip(thrust.input_names, thrust.total_effect, strict=True)
        }
        self.assertGreater(total_thrust["chamber_pressure"], 0.1)
        self.assertGreater(total_thrust["throat_diameter"], 0.01)

    def test_form_and_subset_simulation_agree_with_monte_carlo(self) -> None:
        """P(vacuum Isp <= 378 s) by FORM, subset simulation and crude MC.

        The 378 s requirement is a synthetic threshold for the illustrative
        demonstration table. Band: three combined standard errors for
        subset simulation (its upper coefficient of variation), 0.02 absolute
        for the first-order FORM approximation.
        """
        state = ModelLimitState(_uq_model(), "vacuum_specific_impulse", 378.0, "below")
        mc = model_monte_carlo(state, 3000, seed=3)
        self.assertGreater(mc.probability, 0.02)
        self.assertLess(mc.probability, 0.1)
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.02)
        subset = subset_simulation(state, samples_per_level=400, seed=5)
        self.assertTrue(subset.converged)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)
        # O/F and eta_c* carry the failure; p_c and d_t do not enter.
        self.assertLess(form.importance_factors["chamber_pressure"], 1e-6)
        self.assertGreater(form.importance_factors["mixture_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
