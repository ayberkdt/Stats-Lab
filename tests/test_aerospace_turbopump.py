"""Tests for the turbopump suction and cavitation-risk model.

All numbers are synthetic (a LOX-class liquid with an invented
Clausius-Clapeyron law); none is data from a real pump, engine or vehicle.
Numbers quoted as measured were produced by the stated seeds.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.aerospace.liquid_engine import G0, PumpSpec, pump_power
from probstat.aerospace.turbopump import (
    DEMONSTRATION_VAPOUR_MODEL,
    VapourPressureModel,
    clausius_clapeyron_model,
    line_loss_head,
    minimum_tank_pressure_for_probability,
    npsh_available,
    npsh_required_from_suction_specific_speed,
    npsh_required_from_thoma,
    pump_operating_point,
    suction_margin,
    suction_specific_speed,
    suction_specific_speed_dimensional,
    suction_specific_speed_from_dimensional,
    turbopump_cavitation_model,
)
from probstat.distributions import LogNormal, Normal
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis

RHO = 1140.0
Q = 0.1
DIAMETER = 0.2
HEIGHT = 8.0

MEANS = {
    "tank_pressure": 3.3e5,
    "temperature": 90.0,
    "axial_acceleration": 20.0,
    "loss_coefficient": 3.0,
    "suction_specific_speed": 3.0,
    "pump_speed": 600.0,
}


def _ln(mean: float, cov: float) -> LogNormal:
    sigma = math.sqrt(math.log1p(cov * cov))
    return LogNormal(math.log(mean) - 0.5 * sigma * sigma, sigma)


def _model(pressure_mean: float = 3.3e5, **override):
    args = {
        "tank_pressure": _ln(pressure_mean, 0.03),
        "temperature": Normal(90.0, 1.5),
        "axial_acceleration": Normal(20.0, 2.0),
        "loss_coefficient": _ln(3.0, 0.2),
        "suction_specific_speed": _ln(3.0, 0.1),
        "pump_speed": Normal(600.0, 12.0),
        "density": RHO,
        "volumetric_flow": Q,
        "column_height": HEIGHT,
        "line_diameter": DIAMETER,
    }
    args.update(override)
    return turbopump_cavitation_model(**args)


class VapourAndBudgetTests(unittest.TestCase):
    """Vapour law, line loss and the suction head budget."""

    def test_clausius_clapeyron_reference_and_slope(self) -> None:
        """The law returns p_ref at T_ref and has d ln p / dT = B / T^2."""
        law = DEMONSTRATION_VAPOUR_MODEL
        self.assertAlmostEqual(law.pressure(90.0) / 1.0e5, 1.0, places=12)
        h = 1e-4
        upper = math.log(law.pressure(90.0 + h))
        lower = math.log(law.pressure(90.0 - h))
        self.assertAlmostEqual((upper - lower) / (2 * h), 820.0 / 90.0**2, places=7)

    def test_antoine_form_value(self) -> None:
        """exp(A - B / (T + C)) evaluated by hand."""
        law = VapourPressureModel(a=20.0, b=700.0, c=-5.0, t_min=60.0, t_max=120.0)
        self.assertAlmostEqual(law.pressure(80.0), math.exp(20.0 - 700.0 / 75.0), 9)

    def test_non_physical_temperature_rejected(self) -> None:
        """Outside the validity interval, non-finite or non-real: error."""
        law = DEMONSTRATION_VAPOUR_MODEL
        for bad in (-5.0, 0.0, 69.9, 130.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                law.pressure(bad)
        with self.assertRaises(TypeError):
            law.pressure("90")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VapourPressureModel(a=1.0, b=1.0, c=-80.0, t_min=70.0, t_max=90.0)
        with self.assertRaises(ValueError):
            clausius_clapeyron_model(
                reference_pressure=1e5,
                reference_temperature=90.0,
                latent_heat_over_gas_constant=-1.0,
                t_min=70.0,
                t_max=130.0,
            )

    def test_darcy_weisbach_hand_case(self) -> None:
        """D = 0.1 m, v = 2 m/s, f = 0.02, L = 5 m, K = 1."""
        flow = 2.0 * math.pi * 0.1**2 / 4.0
        loss = line_loss_head(
            volumetric_flow=flow,
            diameter=0.1,
            length=5.0,
            friction_factor=0.02,
            minor_loss_coefficient=1.0,
        )
        self.assertAlmostEqual(loss, (0.02 * 50.0 + 1.0) * 4.0 / (2.0 * G0), places=12)

    def test_npsh_closed_form(self) -> None:
        """(p - pv) / (rho g) + h a / g - loss, with a = g0 on the pad."""
        budget = npsh_available(
            tank_pressure=3.0e5,
            density=1000.0,
            vapour_pressure=1.0e5,
            column_height=10.0,
            axial_acceleration=G0,
            loss_head=0.5,
        )
        expected = 2.0e5 / (1000.0 * G0) + 10.0 - 0.5
        self.assertAlmostEqual(budget.npsh_available, expected, places=12)
        self.assertEqual(budget.flag, "ok")
        self.assertAlmostEqual(budget.column_head, 10.0, places=12)

    def test_acceleration_head_is_linear(self) -> None:
        """Tripling the proper acceleration triples the column head."""
        one = npsh_available(
            tank_pressure=2e5,
            density=1000.0,
            vapour_pressure=1e5,
            column_height=4.0,
            axial_acceleration=G0,
            loss_head=0.0,
        )
        three = npsh_available(
            tank_pressure=2e5,
            density=1000.0,
            vapour_pressure=1e5,
            column_height=4.0,
            axial_acceleration=3 * G0,
            loss_head=0.0,
        )
        zero = npsh_available(
            tank_pressure=2e5,
            density=1000.0,
            vapour_pressure=1e5,
            column_height=4.0,
            axial_acceleration=0.0,
            loss_head=0.0,
        )
        self.assertAlmostEqual(three.column_head, 3 * one.column_head, places=12)
        self.assertAlmostEqual(
            zero.npsh_available, one.pressure_head - one.vapour_head, places=12
        )

    def test_negative_npsh_is_flagged(self) -> None:
        """Inlet total head below the vapour head: negative value and flag."""
        budget = npsh_available(
            tank_pressure=1.0e5,
            density=1000.0,
            vapour_pressure=2.0e5,
            column_height=1.0,
            axial_acceleration=G0,
            loss_head=0.0,
        )
        self.assertLess(budget.npsh_available, 0.0)
        self.assertEqual(budget.flag, "inlet_at_or_below_vapour_pressure")
        ratio, head = suction_margin(budget.npsh_available, 5.0)
        self.assertLess(ratio, 0.0)
        self.assertAlmostEqual(head, budget.npsh_available - 5.0)

    def test_validation(self) -> None:
        """Bad types and ranges are rejected."""
        with self.assertRaises(ValueError):
            line_loss_head(
                volumetric_flow=1.0, diameter=0.0, length=1.0, friction_factor=0.01
            )
        with self.assertRaises(ValueError):
            npsh_available(
                tank_pressure=-1.0,
                density=1.0,
                vapour_pressure=1.0,
                column_height=0.0,
                axial_acceleration=0.0,
                loss_head=0.0,
            )
        with self.assertRaises(TypeError):
            npsh_available(
                tank_pressure=True,
                density=1.0,
                vapour_pressure=1.0,
                column_height=0.0,
                axial_acceleration=0.0,
                loss_head=0.0,
            )


class RequiredHeadTests(unittest.TestCase):
    """Suction specific speed, Thoma number and unit consistency."""

    def test_round_trip(self) -> None:
        """S* of the inverted NPSH_r equals the input S*."""
        npsh = npsh_required_from_suction_specific_speed(
            speed=600.0, volumetric_flow=0.1, s_star=3.0
        )
        recovered = suction_specific_speed(
            speed=600.0, volumetric_flow=0.1, npsh_required=npsh
        )
        self.assertAlmostEqual(recovered, 3.0, places=12)

    def test_dimensional_variant_matches_rpm_definition(self) -> None:
        """S = N sqrt(Q) / NPSH^(3/4) with N in rpm equals the converted S*."""
        speed, flow, s_star = 600.0, 0.1, 3.0
        npsh = npsh_required_from_suction_specific_speed(
            speed=speed, volumetric_flow=flow, s_star=s_star
        )
        rpm = speed * 60.0 / (2.0 * math.pi)
        direct = rpm * math.sqrt(flow) / npsh**0.75
        self.assertAlmostEqual(
            direct / suction_specific_speed_dimensional(s_star), 1.0, places=12
        )
        self.assertAlmostEqual(
            suction_specific_speed_from_dimensional(direct), s_star, places=12
        )

    def test_scaled_units(self) -> None:
        """S* is unchanged when Q is given in L/s and N in rpm and converted."""
        speed_rpm, flow_ls, npsh = 5000.0, 80.0, 12.0
        a = suction_specific_speed(
            speed=speed_rpm * 2.0 * math.pi / 60.0,
            volumetric_flow=flow_ls / 1000.0,
            npsh_required=npsh,
        )
        b = suction_specific_speed_from_dimensional(
            speed_rpm * math.sqrt(flow_ls / 1000.0) / npsh**0.75
        )
        self.assertAlmostEqual(a / b, 1.0, places=12)

    def test_similarity_exponents(self) -> None:
        """NPSH_r scales as w^(4/3) and Q^(2/3) at fixed S*."""
        base = npsh_required_from_suction_specific_speed(
            speed=300.0, volumetric_flow=0.05, s_star=2.5
        )
        fast = npsh_required_from_suction_specific_speed(
            speed=600.0, volumetric_flow=0.05, s_star=2.5
        )
        big = npsh_required_from_suction_specific_speed(
            speed=300.0, volumetric_flow=0.2, s_star=2.5
        )
        self.assertAlmostEqual(fast / base, 2.0 ** (4.0 / 3.0), places=12)
        self.assertAlmostEqual(big / base, 4.0 ** (2.0 / 3.0), places=12)

    def test_thoma_and_margin(self) -> None:
        """NPSH_r = sigma H and the margin pair."""
        self.assertAlmostEqual(
            npsh_required_from_thoma(thoma_number=0.1, pump_head=200.0), 20.0
        )
        ratio, head = suction_margin(30.0, 20.0)
        self.assertAlmostEqual(ratio, 1.5)
        self.assertAlmostEqual(head, 10.0)
        with self.assertRaises(ValueError):
            suction_margin(1.0, 0.0)
        with self.assertRaises(ValueError):
            npsh_required_from_suction_specific_speed(
                speed=0.0, volumetric_flow=0.1, s_star=3.0
            )


class OperatingPointTests(unittest.TestCase):
    """Pump curve against system curve."""

    COEFFS = (250.0, -400.0, -9000.0)

    def test_matches_independent_quadratic_solve(self) -> None:
        """Positive root of (h2 - k) Q^2 + h1 Q + h0 - hs = 0."""
        h0, h1, h2 = self.COEFFS
        hs, k = 40.0, 3000.0
        point = pump_operating_point(
            pump_coefficients=self.COEFFS,
            static_lift=hs,
            system_coefficient=k,
            density=RHO,
            efficiency=0.7,
        )
        a, b, c = h2 - k, h1, h0 - hs
        root = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
        self.assertTrue(point.converged)
        self.assertAlmostEqual(point.volumetric_flow, root, places=11)
        self.assertAlmostEqual(point.head, h0 + h1 * root + h2 * root**2, places=8)
        self.assertAlmostEqual(point.head, hs + k * root**2, places=8)

    def test_power_matches_liquid_engine_and_hydraulic_formula(self) -> None:
        """Shaft power equals rho g Q H / eta and pump_power of the PumpSpec."""
        point = pump_operating_point(
            pump_coefficients=self.COEFFS,
            static_lift=40.0,
            system_coefficient=3000.0,
            density=RHO,
            efficiency=0.7,
        )
        hydraulic = RHO * G0 * point.volumetric_flow * point.head / 0.7
        self.assertAlmostEqual(point.shaft_power / hydraulic, 1.0, places=12)
        spec = PumpSpec(
            mass_flow=RHO * point.volumetric_flow,
            density=RHO,
            pressure_rise=RHO * G0 * point.head,
            efficiency=0.7,
        )
        self.assertAlmostEqual(point.shaft_power / pump_power(spec), 1.0, places=12)

    def test_pure_resistance_limit(self) -> None:
        """With h1 = hs = 0 the root is Q = sqrt(h0 / (k - h2))."""
        point = pump_operating_point(
            pump_coefficients=(100.0, 0.0, -2000.0),
            static_lift=0.0,
            system_coefficient=3000.0,
            density=RHO,
            efficiency=1.0,
        )
        self.assertAlmostEqual(point.volumetric_flow, math.sqrt(100.0 / 5000.0), 11)

    def test_invalid_curves(self) -> None:
        """No intersection, rising pump curve and bad efficiency are rejected."""
        with self.assertRaises(ValueError):
            pump_operating_point(
                pump_coefficients=self.COEFFS,
                static_lift=300.0,
                system_coefficient=0.0,
                density=RHO,
                efficiency=0.7,
            )
        with self.assertRaises(ValueError):
            pump_operating_point(
                pump_coefficients=(250.0, 10.0, -9000.0),
                static_lift=0.0,
                system_coefficient=0.0,
                density=RHO,
                efficiency=0.7,
            )
        with self.assertRaises(ValueError):
            pump_operating_point(
                pump_coefficients=self.COEFFS,
                static_lift=0.0,
                system_coefficient=0.0,
                density=RHO,
                efficiency=1.5,
            )


class ModelTests(unittest.TestCase):
    """Probabilistic model: reliability agreement, Sobol ranking, design."""

    def test_deterministic_evaluation_matches_functions(self) -> None:
        """The model output at the means equals the standalone functions."""
        out = _model().evaluate(MEANS)
        loss = 3.0 * (4 * Q / (math.pi * DIAMETER**2)) ** 2 / (2 * G0)
        budget = npsh_available(
            tank_pressure=3.3e5,
            density=RHO,
            vapour_pressure=DEMONSTRATION_VAPOUR_MODEL.pressure(90.0),
            column_height=HEIGHT,
            axial_acceleration=20.0,
            loss_head=loss,
        )
        required = npsh_required_from_suction_specific_speed(
            speed=600.0, volumetric_flow=Q, s_star=3.0
        )
        self.assertAlmostEqual(out["npsh_available"], budget.npsh_available, 10)
        self.assertAlmostEqual(out["npsh_required"], required, 10)
        self.assertAlmostEqual(out["margin"], budget.npsh_available / required, 12)

    def test_form_subset_and_monte_carlo_agree(self) -> None:
        """P(margin < 1) at 3.3 bar mean tank pressure.

        Measured: FORM 0.0199, MC 0.02045 (SE 0.0010, 20000 draws, seed 1),
        subset simulation 0.0368 (c.o.v. upper bound 0.25, seed 2).
        """
        state = ModelLimitState(_model(), "margin", 1.0, "below")
        mc = model_monte_carlo(state, 20000, seed=1)
        form = form_hlrf(state)
        subset = subset_simulation(state, samples_per_level=500, seed=2)
        self.assertTrue(form.converged and subset.converged)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.004)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)

    def test_sobol_ranking_measured(self) -> None:
        """Margin variance is dominated by the S* scatter, not by temperature.

        Measured (size 256, seed 1), total effects of the margin: S* 0.78,
        acceleration 0.09, temperature 0.07, tank pressure 0.03, pump speed
        0.03, loss coefficient 0.003. For NPSH_a alone: acceleration 0.50,
        temperature 0.35, tank pressure 0.15. The ranking depends on the
        chosen input spreads and is not a general statement.
        """
        result = sobol_analysis(_model(), "margin", 128, seed=1)
        total = dict(
            zip(
                result.input_names,
                (i.estimate for i in result.total_effect),
                strict=True,
            )
        )
        ranked = sorted(total, key=lambda k: total[k], reverse=True)
        self.assertEqual(ranked[0], "suction_specific_speed")
        self.assertGreater(total["suction_specific_speed"], 0.5)
        self.assertLess(total["loss_coefficient"], 0.05)
        avail = sobol_analysis(_model(), "npsh_available", 128, seed=1)
        eff = dict(
            zip(
                avail.input_names,
                (i.estimate for i in avail.total_effect),
                strict=True,
            )
        )
        self.assertGreater(eff["temperature"], eff["tank_pressure"])
        self.assertLess(abs(eff["suction_specific_speed"]), 0.02)
        self.assertLess(abs(eff["pump_speed"]), 0.02)

    def test_probability_decreases_with_tank_pressure(self) -> None:
        """The FORM probability is monotone in the mean tank pressure."""
        probs = [
            form_hlrf(ModelLimitState(_model(p), "margin", 1.0, "below")).probability
            for p in (3.0e5, 3.3e5, 3.6e5, 3.9e5)
        ]
        self.assertTrue(all(a > b for a, b in zip(probs, probs[1:], strict=False)))

    def test_design_pressure_root(self) -> None:
        """The sized pressure attains the target; Monte Carlo agrees."""
        kwargs = {
            "temperature": Normal(90.0, 1.5),
            "axial_acceleration": Normal(20.0, 2.0),
            "loss_coefficient": _ln(3.0, 0.2),
            "suction_specific_speed": _ln(3.0, 0.1),
            "pump_speed": Normal(600.0, 12.0),
            "density": RHO,
            "volumetric_flow": Q,
            "column_height": HEIGHT,
            "line_diameter": DIAMETER,
            "pressure_cov": 0.03,
            "lower": 2.5e5,
            "upper": 5.0e5,
        }
        design = minimum_tank_pressure_for_probability(
            target_probability=0.01, **kwargs
        )
        self.assertTrue(design.converged)
        self.assertAlmostEqual(design.probability, 0.01, delta=1e-6)
        stricter = minimum_tank_pressure_for_probability(
            target_probability=0.001, **kwargs
        )
        self.assertGreater(stricter.tank_pressure, design.tank_pressure)
        state = ModelLimitState(_model(design.tank_pressure), "margin", 1.0, "below")
        mc = model_monte_carlo(state, 20000, seed=5)
        self.assertLess(abs(mc.probability - 0.01), 4.0 * mc.standard_error + 0.003)
        narrow = {**kwargs, "lower": 4.5e5, "upper": 5.0e5}
        with self.assertRaises(ValueError):
            minimum_tank_pressure_for_probability(target_probability=0.01, **narrow)

    def test_failure_cases(self) -> None:
        """Out-of-range temperature draws and bad arguments are flagged."""
        with self.assertRaises(ValueError):
            _model(temperature=200.0)
        with self.assertRaises(ValueError):
            turbopump_cavitation_model(
                tank_pressure=3e5,
                temperature=90.0,
                axial_acceleration=20.0,
                loss_coefficient=3.0,
                suction_specific_speed=3.0,
                pump_speed=600.0,
                density=RHO,
                volumetric_flow=Q,
                column_height=HEIGHT,
                line_diameter=DIAMETER,
            )
        with self.assertRaises(TypeError):
            _model(pump_speed=True)
        with self.assertRaises(ValueError):
            _model().evaluate({**MEANS, "temperature": 200.0})
        low = _model().evaluate(
            {**MEANS, "tank_pressure": 5.0e4, "temperature": 100.0}
            | {"axial_acceleration": 0.0}
        )
        self.assertLess(low["npsh_available"], 0.0)
        self.assertLess(low["margin"], 0.0)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowTests(unittest.TestCase):
    """Larger Monte Carlo cross-check of the FORM probability."""

    def test_form_against_large_monte_carlo(self) -> None:
        """FORM probability within 3 SE + 0.002 of 400000-draw Monte Carlo."""
        state = ModelLimitState(_model(), "margin", 1.0, "below")
        mc = model_monte_carlo(state, 400000, seed=21)
        form = form_hlrf(state)
        self.assertLess(
            abs(form.probability - mc.probability), 3 * mc.standard_error + 0.002
        )


if __name__ == "__main__":
    unittest.main()
