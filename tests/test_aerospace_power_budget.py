"""Tests for the power-subsystem energy-balance reliability.

Order: closed-form eclipse geometry (beta 0, the critical beta, an independent
brute-force shadow scan, Kepler scaling), the array and battery relations, a
hand energy balance (zero eclipse, and a case built to have zero margin), the
lifetime against its closed form, monotonicity, the probabilistic model, the
reliability methods (an exactly linear case against the normal CDF, and the
series system with measured FORM adequacy and the component bounds), the
binomial string mixture, a measured Sobol ranking, the monotone root of the
required beginning-of-life power, and failure cases.

Every number is synthetic and illustrative. The orbit, degradation rates, fade
rates, efficiencies, loads and capacities are invented for the tests; none is a
datasheet value, a published coefficient or a statement about a real
spacecraft.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pickle
import unittest

from probstat.aerospace import power_budget as pb
from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS
from probstat.distributions import LogNormal, Normal, Uniform
from probstat.sensitivity import sobol_analysis

DESIGN = pb.demonstration_design()
# Depth-of-discharge mode active: tighter limit than the demonstration design.
DESIGN_DOD = dataclasses.replace(DESIGN, dod_max=0.30)
EOL = 10.0

NOMINAL = {
    "bol_power": 540.0,
    "degradation_rate": 0.025,
    "fade_rate": 5.0e-6,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "load_factor": 1.0,
    "failed_strings": 0.0,
}


def normal_cdf(x: float) -> float:
    """Return the standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def phi_lifetime_model(
    design, p0, *, fade=None, sigma=0.03, lifetime=False, **overrides
):
    """Return the model whose margins are measured in the reliability tests."""
    given = {
        "bol_power": LogNormal(math.log(p0), sigma),
        "degradation_rate": Uniform(0.015, 0.035),
        "fade_rate": fade if fade is not None else Uniform(2.0e-6, 8.0e-6),
        "charge_efficiency": Uniform(0.90, 0.98),
        "discharge_efficiency": Uniform(0.90, 0.98),
        "load_factor": LogNormal(0.0, 0.04),
        "failed_strings": Uniform(0.0, 2.0),
    }
    given.update(overrides)
    return pb.power_budget_model(design, EOL, lifetime=lifetime, **given)


def shadow_scan_fraction(altitude, beta, points=200000):
    """Independent eclipse fraction: scan the orbit, test the shadow cylinder."""
    r = EARTH_RADIUS + altitude
    sun = (1.0, 0.0, 0.0)
    # Orbit plane contains y and the vector (cos beta, 0, sin beta)-perpendicular:
    # e1 is the projection of the Sun on the plane, e2 completes the frame.
    e1 = (math.cos(beta), 0.0, -math.sin(beta))
    e2 = (0.0, 1.0, 0.0)
    count = 0
    for k in range(points):
        th = 2.0 * math.pi * (k + 0.5) / points
        pos = tuple(
            r * (math.cos(th) * a + math.sin(th) * b)
            for a, b in zip(e1, e2, strict=True)
        )
        along = sum(p * s for p, s in zip(pos, sun, strict=True))
        perp2 = sum(p * p for p in pos) - along * along
        if along < 0.0 and perp2 < EARTH_RADIUS**2:
            count += 1
    return count / points


class GeometryTests(unittest.TestCase):
    """Eclipse fraction and critical beta against closed forms."""

    def test_beta_zero_closed_form(self) -> None:
        """At beta 0 the fraction is arcsin(R/r)/pi, and arccos(sqrt(1-rho^2))/pi."""
        for h in (400.0e3, 600.0e3, 2000.0e3):
            rho = EARTH_RADIUS / (EARTH_RADIUS + h)
            expected = math.asin(rho) / math.pi
            self.assertAlmostEqual(pb.eclipse_fraction(h, 0.0), expected, delta=1e-14)
            self.assertAlmostEqual(
                pb.eclipse_fraction(h, 0.0),
                math.acos(math.sqrt(1.0 - rho * rho)) / math.pi,
                delta=1e-12,
            )

    def test_critical_beta(self) -> None:
        """Zero eclipse at and above beta* = arcsin(R/r); positive just below."""
        h = 600.0e3
        crit = pb.critical_beta(h)
        self.assertAlmostEqual(
            math.sin(crit), EARTH_RADIUS / (EARTH_RADIUS + h), delta=1e-14
        )
        self.assertEqual(pb.eclipse_fraction(h, crit + 1.0e-9), 0.0)
        self.assertEqual(pb.eclipse_fraction(h, crit + 0.2), 0.0)
        self.assertEqual(pb.eclipse_fraction(h, 0.5 * math.pi), 0.0)
        just = pb.eclipse_fraction(h, crit - 1.0e-6)
        self.assertGreater(just, 0.0)
        # Square-root onset: f ~ sqrt(2 eps tan(beta*) ...) is small but not tiny.
        self.assertLess(just, 5.0e-3)
        self.assertGreater(pb.eclipse_fraction(h, crit - 0.05), just)

    def test_monotone_and_symmetric_in_beta(self) -> None:
        """The fraction decreases in |beta| and is even in beta."""
        h = 800.0e3
        grid = [0.02 * k for k in range(0, 78)]
        values = [pb.eclipse_fraction(h, b) for b in grid]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:], strict=False)))
        self.assertEqual(pb.eclipse_fraction(h, -0.4), pb.eclipse_fraction(h, 0.4))

    def test_against_brute_force_shadow_scan(self) -> None:
        """Independent geometry: scan the orbit and test the shadow cylinder."""
        h = 600.0e3
        for beta in (0.0, 0.5, 1.0, 1.15):
            scan = shadow_scan_fraction(h, beta, points=40000)
            self.assertAlmostEqual(pb.eclipse_fraction(h, beta), scan, delta=5.0e-5)

    def test_period(self) -> None:
        """Kepler: T = 2 pi r / v_circ and T ~ r^(3/2)."""
        h1, h2 = 500.0e3, 1500.0e3
        r1 = EARTH_RADIUS + h1
        self.assertAlmostEqual(
            pb.orbital_period(h1),
            2.0 * math.pi * r1 / math.sqrt(EARTH_MU / r1),
            delta=1e-8,
        )
        ratio = pb.orbital_period(h2) / pb.orbital_period(h1)
        self.assertAlmostEqual(ratio, ((EARTH_RADIUS + h2) / r1) ** 1.5, delta=1e-12)
        self.assertAlmostEqual(
            pb.eclipse_duration(h1, 0.0),
            pb.eclipse_fraction(h1, 0.0) * pb.orbital_period(h1),
            delta=1e-9,
        )


class ComponentTests(unittest.TestCase):
    """Array power and battery capacity."""

    def test_array_power_loss_chain(self) -> None:
        """P = P0 eta cos(theta) (1 - kT dT) (1-d)^t (1 - nf/N), by hand."""
        d = DESIGN
        expected = (
            400.0
            * 0.95
            * math.cos(math.radians(10.0))
            * (1.0 - 0.002 * 40.0)
            * 0.97**4.0
            * (1.0 - 2.0 / 20.0)
        )
        got = pb.array_power(d, 400.0, 4.0, degradation_rate=0.03, failed_strings=2.0)
        self.assertAlmostEqual(got, expected, delta=1e-10)
        linear = dataclasses.replace(d, degradation_law="linear")
        self.assertAlmostEqual(
            pb.array_power(linear, 400.0, 4.0, degradation_rate=0.03),
            400.0 * 0.95 * math.cos(math.radians(10.0)) * 0.92 * (1.0 - 0.12),
            delta=1e-10,
        )

    def test_battery_fade_laws(self) -> None:
        """Linear and square-root laws at a known cycle count."""
        d = DESIGN
        n = 3.0 * d.orbits_per_year
        self.assertAlmostEqual(
            pb.battery_capacity(d, 3.0, fade_rate=2.0e-6),
            500.0 * (1.0 - 2.0e-6 * n),
            delta=1e-10,
        )
        s = dataclasses.replace(d, fade_law="sqrt")
        self.assertAlmostEqual(
            pb.battery_capacity(s, 3.0, fade_rate=1.0e-3),
            500.0 * (1.0 - 1.0e-3 * math.sqrt(n)),
            delta=1e-10,
        )
        self.assertEqual(pb.battery_capacity(d, 0.0, fade_rate=1.0e-3), 500.0)


class BalanceTests(unittest.TestCase):
    """Energy balance of one orbit by hand, lifetime and monotonicity."""

    def test_no_eclipse_hand_case(self) -> None:
        """Beta above beta*: margin = (P - L_s) T / 3600 and DoD = 0."""
        d = dataclasses.replace(
            DESIGN,
            beta=1.4,
            bus_efficiency=1.0,
            incidence_angle=0.0,
            temperature_coefficient=0.0,
        )
        s = pb.power_budget(
            d,
            0.0,
            bol_power=260.0,
            degradation_rate=0.02,
            fade_rate=1e-6,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
        )
        period = pb.orbital_period(d.altitude)
        self.assertEqual(s.eclipse_seconds, 0.0)
        self.assertAlmostEqual(s.margin_wh, 60.0 * period / 3600.0, delta=1e-9)
        self.assertAlmostEqual(s.power_margin_w, 60.0, delta=1e-9)
        self.assertAlmostEqual(s.power_margin_ratio, 60.0 / 200.0, delta=1e-12)
        self.assertEqual(s.dod, 0.0)
        self.assertEqual(s.combined_margin, s.power_margin_ratio)

    def test_zero_margin_by_construction(self) -> None:
        """With unit efficiencies, P = (L_s T_s + L_e T_e) / T_s gives margin 0."""
        d = dataclasses.replace(
            DESIGN, bus_efficiency=1.0, incidence_angle=0.0, temperature_coefficient=0.0
        )
        t_e = d.eclipse_seconds
        t_s = d.period - t_e
        p0 = (200.0 * t_s + 150.0 * t_e) / t_s
        s = pb.power_budget(
            d,
            0.0,
            bol_power=p0,
            degradation_rate=0.02,
            fade_rate=1e-6,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
        )
        self.assertAlmostEqual(s.margin_wh, 0.0, delta=1e-9)
        self.assertAlmostEqual(s.dod, 150.0 * t_e / 3600.0 / 500.0, delta=1e-12)
        self.assertAlmostEqual(s.dod_margin_ratio, 1.0 - s.dod / 0.4, delta=1e-12)

    def test_efficiency_enters_the_recharge_only(self) -> None:
        """E_req = L_s T_s + E_e / (eta_d eta_c); DoD = E_e / (C eta_d)."""
        d = DESIGN
        s = pb.power_budget(
            d,
            0.0,
            bol_power=500.0,
            degradation_rate=0.02,
            fade_rate=1e-6,
            charge_efficiency=0.8,
            discharge_efficiency=0.9,
        )
        t_e, t_s = d.eclipse_seconds, d.period - d.eclipse_seconds
        e_e = 150.0 * t_e / 3600.0
        self.assertAlmostEqual(
            s.required_wh, 200.0 * t_s / 3600.0 + e_e / (0.9 * 0.8), delta=1e-9
        )
        self.assertAlmostEqual(s.dod, e_e / (500.0 * 0.9), delta=1e-12)

    def test_lifetime_closed_form(self) -> None:
        """Compound degradation, no battery limit: t = ln(P0 K / req) / -ln(1-d)."""
        d = DESIGN
        kwargs = {
            "degradation_rate": 0.03,
            "fade_rate": 0.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        }
        s0 = pb.power_budget(d, 0.0, bol_power=500.0, **kwargs)
        expected = math.log(s0.available_wh / s0.required_wh) / -math.log(0.97)
        life = pb.power_lifetime(d, 40.0, bol_power=500.0, **kwargs)
        self.assertFalse(life.survived)
        self.assertAlmostEqual(life.years, expected, delta=1e-7)
        after = pb.power_budget(d, life.years + 1e-3, bol_power=500.0, **kwargs)
        self.assertLess(after.combined_margin, 0.0)
        self.assertAlmostEqual(
            pb.power_budget(d, life.years, bol_power=500.0, **kwargs).combined_margin,
            0.0,
            delta=1e-7,
        )

    def test_lifetime_censoring_and_immediate_failure(self) -> None:
        """A large array survives the horizon; a small one fails at t = 0."""
        kwargs = {
            "degradation_rate": 0.02,
            "fade_rate": 1.0e-6,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        }
        big = pb.power_lifetime(DESIGN, 8.0, bol_power=2000.0, **kwargs)
        self.assertTrue(big.survived)
        self.assertEqual(big.years, 8.0)
        small = pb.power_lifetime(DESIGN, 8.0, bol_power=100.0, **kwargs)
        self.assertEqual((small.years, small.survived), (0.0, False))

    def test_dod_limited_lifetime(self) -> None:
        """With a huge array the battery limit alone ends the mission."""
        kwargs = {
            "degradation_rate": 0.0,
            "fade_rate": 8.0e-6,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        }
        life = pb.power_lifetime(DESIGN_DOD, 30.0, bol_power=5000.0, **kwargs)
        s = pb.power_budget(DESIGN_DOD, life.years, bol_power=5000.0, **kwargs)
        self.assertFalse(life.survived)
        self.assertAlmostEqual(s.dod, 0.30, delta=1e-6)
        self.assertAlmostEqual(s.dod_margin_ratio, s.combined_margin, delta=1e-12)

    def test_monotonicity(self) -> None:
        """Margins decrease in time, load, degradation, fade; increase in power."""

        def margin(attr: str = "combined_margin", **change: float) -> float:
            args = dict(NOMINAL)
            years = change.pop("years", EOL)
            args.update(change)
            return float(getattr(pb.power_budget(DESIGN_DOD, years, **args), attr))

        base = margin()
        self.assertGreater(margin(years=5.0), base)
        self.assertGreater(
            margin("power_margin_ratio", bol_power=600.0), margin("power_margin_ratio")
        )
        self.assertLess(margin(load_factor=1.05), base)
        self.assertLess(
            margin("power_margin_ratio", load_factor=1.05), margin("power_margin_ratio")
        )
        self.assertLess(
            margin("power_margin_ratio", degradation_rate=0.03),
            margin("power_margin_ratio"),
        )
        self.assertLess(
            margin("dod_margin_ratio", fade_rate=7.0e-6),
            margin("dod_margin_ratio", fade_rate=5.0e-6),
        )
        self.assertGreater(
            margin("power_margin_ratio", charge_efficiency=0.98),
            margin("power_margin_ratio"),
        )
        self.assertLess(
            margin("power_margin_ratio", failed_strings=1.0),
            margin("power_margin_ratio"),
        )
        years = [0.5 * k for k in range(0, 30)]
        values = [margin(years=t) for t in years]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:], strict=False)))
        lives = [
            pb.power_lifetime(DESIGN, 60.0, **{**NOMINAL, "bol_power": p}).years
            for p in (450.0, 500.0, 550.0, 600.0)
        ]
        self.assertTrue(all(a < b for a, b in zip(lives, lives[1:], strict=False)))


class ModelTests(unittest.TestCase):
    """The probabilistic model reproduces the deterministic balance."""

    def test_outputs_match_balance(self) -> None:
        """Model quantities equal power_budget and power_lifetime at one point."""
        model = phi_lifetime_model(DESIGN_DOD, 540.0, lifetime=True)
        out = model.evaluate(dict(NOMINAL))
        s = pb.power_budget(DESIGN_DOD, EOL, **NOMINAL)
        self.assertAlmostEqual(out["combined_margin"], s.combined_margin, delta=1e-14)
        self.assertAlmostEqual(out["eol_dod"], s.dod, delta=1e-14)
        self.assertAlmostEqual(out["eol_power_margin_w"], s.power_margin_w, delta=1e-9)
        life = pb.power_lifetime(DESIGN_DOD, 2.0 * EOL, **NOMINAL)
        self.assertAlmostEqual(out["lifetime_years"], life.years, delta=1e-12)
        self.assertAlmostEqual(
            out["combined_margin"],
            min(out["eol_power_margin_ratio"], out["eol_dod_margin_ratio"]),
            delta=1e-15,
        )
        pickle.loads(pickle.dumps(model.model))

    def test_failure_policy_for_exhausted_battery(self) -> None:
        """A fade rate that empties the battery is reported, not clipped."""
        model = phi_lifetime_model(DESIGN_DOD, 540.0)
        with self.assertRaises(ValueError):
            model.evaluate({**NOMINAL, "fade_rate": 1.0e-4})


class ReliabilityTests(unittest.TestCase):
    """FORM, subset simulation and Monte Carlo on the margins."""

    def test_linear_case_matches_normal_cdf(self) -> None:
        """Only BOL power random (normal): P_f = Phi((p_crit - mu) / sigma).

        The combined margin is linear in the BOL power on the failure side
        (the depth-of-discharge margin is a positive constant there), so FORM
        is exact and the critical power follows from the balance at BOL = 1 W.
        """
        mu, sigma = 450.0, 25.0
        fixed = {k: v for k, v in NOMINAL.items() if k != "bol_power"}
        model = pb.power_budget_model(DESIGN, EOL, bol_power=Normal(mu, sigma), **fixed)
        unit = pb.power_budget(DESIGN, EOL, **{**NOMINAL, "bol_power": 1.0})
        p_crit = unit.required_wh / unit.available_wh
        self.assertGreater(unit.dod_margin_ratio, 0.0)
        exact = normal_cdf((p_crit - mu) / sigma)
        self.assertGreater(exact, 0.05)
        self.assertLess(exact, 0.95)
        form = pb.power_failure_probability(model)
        self.assertTrue(form.converged)
        self.assertAlmostEqual(form.probability, exact, delta=1e-6)
        mc = pb.power_failure_probability(
            model, method="monte_carlo", samples=4000, seed=3
        )
        self.assertLess(abs(mc.probability - exact), 4.0 * mc.standard_error)
        sub = pb.power_failure_probability(model, method="subset", samples=400, seed=5)
        self.assertLess(
            abs(sub.probability - exact), 4.0 * max(sub.standard_error, 0.01)
        )

    def test_series_system_form_adequacy_measured(self) -> None:
        """Both modes contribute at BOL 500 W: FORM on the minimum is low.

        Measured in development (BOL 500 W nominal, 4000 samples): FORM 0.277,
        Monte Carlo 0.375 +- 0.008, subset (500 per level) 0.384, component
        FORM 0.277 (power) and 0.211 (DoD), so the series bounds are
        [0.277, 0.488] and contain the simulation value. FORM follows one
        branch of the minimum and underestimates by about 0.1 here.
        """
        model = phi_lifetime_model(DESIGN_DOD, 500.0)
        form = pb.power_failure_probability(model)
        mc = pb.power_failure_probability(
            model, method="monte_carlo", samples=3000, seed=4
        )
        sub = pb.power_failure_probability(model, method="subset", samples=400, seed=6)
        band = 4.0 * math.hypot(mc.standard_error, sub.standard_error)
        self.assertLess(abs(sub.probability - mc.probability), band)
        self.assertGreater(mc.probability, 0.30)
        self.assertLess(mc.probability, 0.45)
        # FORM on the minimum underestimates (one branch only) but is within 0.2.
        self.assertLess(form.probability, mc.probability)
        self.assertLess(mc.probability - form.probability, 0.2)
        bounds = pb.component_form_bounds(model)
        self.assertLessEqual(bounds.lower, mc.probability + 4.0 * mc.standard_error)
        self.assertGreaterEqual(bounds.upper, mc.probability - 4.0 * mc.standard_error)
        self.assertGreater(bounds.power, 0.15)
        self.assertGreater(bounds.dod, 0.15)

    def test_single_mode_dominated_case(self) -> None:
        """At BOL 580 W the DoD mode dominates and FORM is close to simulation.

        Measured (4000 samples, seed 0): FORM 0.211, Monte Carlo 0.202 +- 0.006,
        power-mode Monte Carlo 0.0045, DoD-mode Monte Carlo 0.199.
        """
        model = phi_lifetime_model(DESIGN_DOD, 580.0)
        form = pb.power_failure_probability(model)
        mc = pb.power_failure_probability(
            model, method="monte_carlo", samples=3000, seed=0
        )
        self.assertLess(abs(form.probability - mc.probability), 0.03)
        dod = pb.power_failure_probability(
            model, quantity="eol_dod_margin_ratio", method="monte_carlo", samples=3000
        )
        self.assertLess(abs(dod.probability - mc.probability), 0.02)


class StringFailureTests(unittest.TestCase):
    """Binomial failed strings."""

    def test_pmf(self) -> None:
        """The pmf sums to one, has mean N q and matches C(N, k) by hand."""
        pmf = pb.string_failure_pmf(20, 0.1)
        self.assertEqual(len(pmf), 21)
        self.assertAlmostEqual(sum(pmf), 1.0, delta=1e-12)
        self.assertAlmostEqual(sum(k * p for k, p in enumerate(pmf)), 2.0, delta=1e-12)
        self.assertAlmostEqual(pmf[2], 190.0 * 0.01 * 0.9**18, delta=1e-12)

    def test_mixture(self) -> None:
        """P(fail | k) = 1{k >= 3} gives the binomial upper tail."""
        pmf = pb.string_failure_pmf(20, 0.1)
        tail = 1.0 - sum(pmf[:3])
        mix = pb.binomial_mixture_probability(lambda k: 1.0 if k >= 3 else 0.0, 20, 0.1)
        self.assertAlmostEqual(mix.probability, tail, delta=1e-9)
        self.assertLessEqual(mix.neglected_mass, 1.0e-9)
        const = pb.binomial_mixture_probability(lambda k: 0.5, 20, 0.1, tail=1e-3)
        self.assertGreater(const.neglected_mass, 0.0)
        self.assertAlmostEqual(
            const.probability + 0.5 * const.neglected_mass, 0.5, delta=0.5e-3
        )

    def test_mixture_with_the_model(self) -> None:
        """Exact mixture with the model is within the neglected mass of a hand sum."""

        def conditional(k: int) -> float:
            model = pb.power_budget_model(
                DESIGN,
                EOL,
                bol_power=Normal(400.0, 20.0),
                degradation_rate=0.025,
                fade_rate=5.0e-6,
                failed_strings=float(k),
            )
            return pb.power_failure_probability(model).probability

        mix = pb.binomial_mixture_probability(conditional, 20, 0.05)
        self.assertGreater(mix.probability, 0.0)
        # More failures can only lower the array power: conditional P rises in k.
        values = [conditional(k) for k in (0, 1, 2, 3)]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:], strict=False)))
        pmf = pb.string_failure_pmf(20, 0.05)
        by_hand = sum(pmf[k] * conditional(k) for k in mix.counts)
        self.assertAlmostEqual(mix.probability, by_hand, delta=1e-12)


class SensitivityTests(unittest.TestCase):
    """Sobol ranking of each margin, measured."""

    def test_modes_have_different_leaders(self) -> None:
        """Degradation leads the power margin; the fade rate leads the DoD margin.

        Measured in development (size 256, seed 1, BOL 540 W), total effects:
        power ratio: degradation 0.50, load 0.22, BOL 0.13, failed strings
        0.13, others <= 0.01; DoD ratio: fade 0.89, load 0.08, discharge
        efficiency 0.03, others 0; combined margin: fade 0.71, degradation
        0.17, load 0.16.
        """
        model = phi_lifetime_model(DESIGN_DOD, 540.0)
        power = sobol_analysis(model, "eol_power_margin_ratio", 256, seed=1)
        dod = sobol_analysis(model, "eol_dod_margin_ratio", 256, seed=1)

        def total(result):
            return {
                n: i.estimate
                for n, i in zip(result.input_names, result.total_effect, strict=True)
            }

        tp, td = total(power), total(dod)
        self.assertEqual(max(tp, key=tp.__getitem__), "degradation_rate")
        self.assertGreater(tp["degradation_rate"], 0.3)
        self.assertLess(tp["fade_rate"], 0.05)
        self.assertEqual(max(td, key=td.__getitem__), "fade_rate")
        self.assertGreater(td["fade_rate"], 0.7)
        self.assertLess(td["bol_power"], 0.05)
        self.assertLess(td["degradation_rate"], 0.05)


class DesignTests(unittest.TestCase):
    """Required beginning-of-life power."""

    @staticmethod
    def build(p: float):
        """Power-mode model (battery limit slack) with nominal BOL power ``p``."""
        return phi_lifetime_model(
            DESIGN, p, fade=Uniform(1.0e-6, 3.0e-6), failed_strings=Uniform(0.0, 2.0)
        )

    def test_probability_is_non_increasing_in_power(self) -> None:
        """Common random numbers: MC probability is non-increasing in BOL power."""
        powers = [400.0 + 20.0 * k for k in range(10)]
        probs = [
            pb.power_failure_probability(
                self.build(p), method="monte_carlo", samples=500, seed=2
            ).probability
            for p in powers
        ]
        self.assertTrue(all(a >= b for a, b in zip(probs, probs[1:], strict=False)))
        self.assertGreater(probs[0], 0.9)
        self.assertLess(probs[-1], 0.1)

    def test_root_form_and_monte_carlo(self) -> None:
        """The root meets the target, its lower neighbour does not; MC agrees."""
        target = 0.05
        form = pb.required_bol_power(self.build, target, (440.0, 700.0))
        self.assertLessEqual(form.probability, target)
        self.assertLessEqual(form.bol_power - form.lower_power, 0.5)
        self.assertGreater(
            pb.power_failure_probability(self.build(form.lower_power)).probability,
            target,
        )
        mc = pb.required_bol_power(
            self.build,
            target,
            (440.0, 700.0),
            method="monte_carlo",
            samples=1500,
            seed=3,
        )
        self.assertLessEqual(mc.probability, target)
        # Both estimate the same quantile; MCSE of a 5 % tail at n = 1500 is 0.6 %,
        # about 12 W here (measured slope), so allow 30 W.
        self.assertLess(abs(mc.bol_power - form.bol_power), 30.0)

    def test_tighter_target_needs_a_larger_array(self) -> None:
        """The required power is monotone in the target probability."""
        values = [
            pb.required_bol_power(self.build, t, (440.0, 800.0)).bol_power
            for t in (0.2, 0.05, 0.01)
        ]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_battery_floor_has_no_root(self) -> None:
        """A target below the DoD-mode probability cannot be met by a larger array."""
        build = lambda p: phi_lifetime_model(DESIGN_DOD, p)  # noqa: E731
        with self.assertRaises(ValueError):
            pb.required_bol_power(build, 0.05, (500.0, 900.0))


class FailureCaseTests(unittest.TestCase):
    """Invalid input is rejected, not repaired."""

    def test_geometry(self) -> None:
        """Altitude and beta ranges."""
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                pb.eclipse_fraction(bad, 0.0)
        with self.assertRaises(ValueError):
            pb.eclipse_fraction(600.0e3, 1.6)
        with self.assertRaises(TypeError):
            pb.eclipse_fraction(True, 0.0)
        with self.assertRaises(TypeError):
            pb.orbital_period("600")

    def test_design(self) -> None:
        """Design validation."""
        for field, value in (
            ("dod_max", 1.5),
            ("bus_efficiency", 0.0),
            ("load_sunlit", 0.0),
            ("load_eclipse", -1.0),
            ("incidence_angle", 1.6),
            ("temperature_coefficient", -0.1),
            ("fade_law", "cubic"),
            ("degradation_law", "step"),
            ("battery_capacity_wh", 0.0),
        ):
            with self.assertRaises(ValueError, msg=field):
                dataclasses.replace(DESIGN, **{field: value})
        with self.assertRaises(TypeError):
            dataclasses.replace(DESIGN, total_strings=2.5)
        with self.assertRaises(ValueError):
            dataclasses.replace(DESIGN, total_strings=0)

    def test_budget(self) -> None:
        """Rates, efficiencies, exhausted components."""
        with self.assertRaises(ValueError):
            pb.array_power(DESIGN, 400.0, 1.0, degradation_rate=1.0)
        with self.assertRaises(ValueError):
            pb.array_power(DESIGN, 400.0, 1.0, degradation_rate=-0.1)
        with self.assertRaises(ValueError):
            pb.array_power(
                DESIGN, 400.0, 1.0, degradation_rate=0.1, failed_strings=20.0
            )
        linear = dataclasses.replace(DESIGN, degradation_law="linear")
        with self.assertRaises(ValueError):
            pb.array_power(linear, 400.0, 30.0, degradation_rate=0.04)
        with self.assertRaises(ValueError):
            pb.battery_capacity(DESIGN, 10.0, fade_rate=1.0e-4)
        with self.assertRaises(ValueError):
            pb.battery_capacity(DESIGN, 1.0, fade_rate=-1.0e-6)
        with self.assertRaises(ValueError):
            pb.power_budget(DESIGN, 1.0, **{**NOMINAL, "charge_efficiency": 1.2})
        with self.assertRaises(ValueError):
            pb.power_budget(DESIGN, -1.0, **NOMINAL)
        with self.assertRaises(TypeError):
            pb.power_budget("design", 1.0, **NOMINAL)
        # Exhausted battery inside the search counts as a negative margin.
        life = pb.power_lifetime(DESIGN, 20.0, **{**NOMINAL, "fade_rate": 5.0e-5})
        self.assertFalse(life.survived)
        self.assertLess(life.years, 20.0)

    def test_model_and_methods(self) -> None:
        """Model construction and method selection."""
        fixed = {k: v for k, v in NOMINAL.items()}
        with self.assertRaises(ValueError):
            pb.power_budget_model(DESIGN, EOL, **fixed)
        with self.assertRaises(TypeError):
            pb.power_budget_model(DESIGN, EOL, **{**fixed, "bol_power": True})
        with self.assertRaises(ValueError):
            pb.power_budget_model(DESIGN, 0.0, **{**fixed, "bol_power": Normal(1, 1)})
        model = phi_lifetime_model(DESIGN, 540.0)
        with self.assertRaises(ValueError):
            pb.power_failure_probability(model, method="gibbs")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pb.power_failure_probability(model, method="monte_carlo", samples=1)
        with self.assertRaises(TypeError):
            pb.power_failure_probability("model")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pb.required_bol_power(self_build, 0.05, (700.0, 440.0))
        with self.assertRaises(ValueError):
            pb.required_bol_power(self_build, 1.5, (440.0, 700.0))
        with self.assertRaises(ValueError):
            pb.required_bol_power(self_build, 0.05, (600.0, 700.0))
        with self.assertRaises(ValueError):
            pb.string_failure_pmf(10, 1.5)
        with self.assertRaises(TypeError):
            pb.string_failure_pmf(10.0, 0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pb.binomial_mixture_probability(lambda k: 2.0, 5, 0.1)


def self_build(p: float):
    """Return the power-mode model used by the failure-case tests."""
    return DesignTests.build(p)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowTests(unittest.TestCase):
    """Larger studies (slow tier)."""

    def test_series_agreement_large_sample(self) -> None:
        """Monte Carlo (20000) and subset (2000) agree at BOL 500 W."""
        model = phi_lifetime_model(DESIGN_DOD, 500.0)
        mc = pb.power_failure_probability(
            model, method="monte_carlo", samples=20000, seed=1
        )
        sub = pb.power_failure_probability(model, method="subset", samples=2000, seed=2)
        band = 4.0 * math.hypot(mc.standard_error, sub.standard_error)
        self.assertLess(abs(sub.probability - mc.probability), band)


if __name__ == "__main__":
    unittest.main()
