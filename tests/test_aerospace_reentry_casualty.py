"""Tests for ``probstat.aerospace.reentry_casualty``: demise, casualty area, risk.

Verification order: closed forms first (geometry, the casualty-area hand case,
the temperature-energy map, a constant-density vertical fall against its
analytic impact speed, the heat-load identity against
:func:`probstat.aerospace.entry.simulate_entry`), then an independent
quadrature of the energy balance, then physical monotonicity and the demise of
a thin plate against the survival of a heavy sphere, then the probabilistic
model (Sobol ranking, Monte Carlo, subset simulation and FORM, including the
documented failure of FORM on the step-valued casualty expectation) and the
design levers, and last the failure cases. Every object and number is
SYNTHETIC; nothing here is data of a real vehicle or object.

Tolerances are stated next to each assertion with their justification. The
studies of the slow tier run only with ``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import math
import os
import statistics
import unittest
from dataclasses import replace

from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS
from probstat.aerospace.entry import (
    SUTTON_GRAVES_EARTH_K,
    EntryVehicle,
    ExponentialAtmosphere,
    simulate_entry,
    sutton_graves_heat_flux,
)
from probstat.aerospace.reentry_casualty import (
    STEFAN_BOLTZMANN,
    CasualtyAssessment,
    Fragment,
    FragmentFate,
    Material,
    _temperature,
    assess_reentry_casualty,
    casualty_area,
    cylinder_fragment,
    demonstration_fragments,
    demonstration_materials,
    plate_fragment,
    reentry_casualty_model,
    simulate_fragment,
    sphere_fragment,
)
from probstat.distributions import LogNormal, Normal, Uniform
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis

_MATS = demonstration_materials()
_PERSON = 0.36  # m^2, synthetic person area
_ENERGY = 15.0  # J, synthetic energy threshold (to be checked)
_LIMIT = 1.0e-4  # synthetic casualty-expectation limit (to be checked)
_DENSITY = 40.0  # persons / km^2, synthetic


def _fates(**kw: float) -> dict[str, FragmentFate]:
    """Fly the demonstration fragments from 78 km at a shallow angle.

    Args:
        **kw: Keyword overrides of :func:`simulate_fragment`.

    Returns:
        Name to fate.

    """
    out: dict[str, FragmentFate] = {}
    for frag in demonstration_fragments():
        args: dict[str, float] = {"breakup_altitude": 78e3, "flight_path_angle": -0.01}
        args.update(kw)
        result = simulate_fragment(frag, **args)  # type: ignore[arg-type]
        assert isinstance(result, FragmentFate)
        out[frag.name] = result
    return out


def _three() -> tuple[Fragment, ...]:
    """Return the tank, panel and shaft of the demonstration set.

    Returns:
        The three fragments.

    """
    return tuple(
        f for f in demonstration_fragments() if f.name in ("tank", "panel", "shaft")
    )


def _small_model(fragments: tuple[Fragment, ...] | None = None, **kw: object):  # type: ignore[no-untyped-def]
    """Build the four-input model of the fast tier.

    Args:
        fragments: Fragments (default the tank, panel and shaft).
        **kw: Overrides of :func:`reentry_casualty_model` arguments.

    Returns:
        The model.

    """
    args: dict[str, object] = {
        "breakup_altitude": Normal(78e3, 4e3),
        "flight_path_angle": -0.01,
        "person_area": _PERSON,
        "population_density": _DENSITY,
        "energy_threshold": _ENERGY,
        "casualty_threshold": _LIMIT,
        "heating_scale": LogNormal(0.0, 0.3),
        "melt_temperature_scale": {"aluminium_like": Normal(1.0, 0.03)},
        "ballistic_scale": {"panel": LogNormal(0.0, 0.15)},
    }
    args.update(kw)
    return reentry_casualty_model(fragments or _three(), **args)  # type: ignore[arg-type]


def _full_model():  # type: ignore[no-untyped-def]
    """Build the nine-input model on three fragments (slow tier).

    Returns:
        The model.

    """
    return reentry_casualty_model(
        _three(),
        breakup_altitude=Normal(78e3, 4e3),
        flight_path_angle=Uniform(-0.02, -0.003),
        person_area=_PERSON,
        population_density=_DENSITY,
        energy_threshold=_ENERGY,
        casualty_threshold=_LIMIT,
        heating_scale=LogNormal(0.0, 0.3),
        melt_temperature_scale={
            "aluminium_like": Normal(1.0, 0.03),
            "titanium_like": Normal(1.0, 0.03),
        },
        fusion_heat_scale={"aluminium_like": Normal(1.0, 0.1)},
        emissivity_scale={"aluminium_like": Uniform(0.7, 1.3)},
        ballistic_scale={"panel": LogNormal(0.0, 0.15), "tank": LogNormal(0.0, 0.15)},
    )


class GeometryTests(unittest.TestCase):
    """Fragment geometry against closed forms and input validation."""

    def test_sphere_closed_forms(self) -> None:
        """S = pi d^2, A = S/4 = pi d^2/4, R_n = d/2, beta = m / (C_D A)."""
        frag = sphere_fragment(
            "s", _MATS["steel_like"], 10.0, 0.5, drag_coefficient=0.9
        )
        self.assertAlmostEqual(frag.surface_area, math.pi * 0.25, places=14)
        self.assertAlmostEqual(frag.reference_area, math.pi * 0.25 / 4, places=14)
        self.assertAlmostEqual(frag.nose_radius, 0.25, places=14)
        self.assertAlmostEqual(
            frag.ballistic_coefficient, 10.0 / (0.9 * math.pi * 0.0625), places=12
        )

    def test_cylinder_and_plate_surface_areas(self) -> None:
        """Hand-computed surface areas of a closed cylinder and a thin plate."""
        cyl = cylinder_fragment("c", _MATS["steel_like"], 2.0, 0.2, 1.0)
        self.assertAlmostEqual(
            cyl.surface_area, math.pi * 0.2 * 1.0 + 0.5 * math.pi * 0.04, places=14
        )
        plate = plate_fragment("p", _MATS["aluminium_like"], 1.0, 2.0, 1.0, 0.01)
        self.assertAlmostEqual(plate.surface_area, 4.0 + 2 * 0.01 * 3.0, places=14)

    def test_validation(self) -> None:
        """Bad materials, fragments and counts are rejected."""
        with self.assertRaises(ValueError):
            Material("m", 900.0, 900.0, 3e5, 1.2)
        with self.assertRaises(ValueError):
            Material("bad name", 900.0, 900.0, 3e5, 0.5)
        with self.assertRaises(ValueError):
            Material("m", -1.0, 900.0, 3e5, 0.5)
        with self.assertRaises(TypeError):
            Material("m", "x", 900.0, 3e5, 0.5)  # type: ignore[arg-type]
        mat = _MATS["steel_like"]
        with self.assertRaises(ValueError):
            sphere_fragment("s", mat, 1.0, 0.0)
        with self.assertRaises(ValueError):
            sphere_fragment("s", mat, 1.0, 0.3, count=0)
        with self.assertRaises(TypeError):
            sphere_fragment("s", mat, 1.0, 0.3, count=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Fragment("s", "cube", mat, 1.0, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Fragment("s", "sphere", "steel", 1.0, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            plate_fragment("p", mat, 1.0, 1.0, 1.0, -0.1)


class CasualtyAreaTests(unittest.TestCase):
    """The casualty-area formula and the temperature-energy map by hand."""

    def test_hand_case(self) -> None:
        """A_p = 0.36, A_f = 0.64: (0.6 + 0.8)^2 = 1.96 m^2."""
        self.assertAlmostEqual(casualty_area(0.64, 0.36), 1.96, places=12)
        self.assertAlmostEqual(casualty_area(0.0, 0.36), 0.36, places=14)
        # Larger fragments only increase the area.
        self.assertGreater(casualty_area(1.0, 0.36), casualty_area(0.64, 0.36))

    def test_area_validation(self) -> None:
        """Negative fragment area and non-positive person area are rejected."""
        with self.assertRaises(ValueError):
            casualty_area(-0.1, 0.36)
        with self.assertRaises(ValueError):
            casualty_area(0.1, 0.0)
        with self.assertRaises(TypeError):
            casualty_area("a", 0.36)  # type: ignore[arg-type]

    def test_temperature_energy_map(self) -> None:
        """Hand values on the three branches (m = 2, c = 500, T_m = 1000, L_f = 1e5)."""
        mat = Material("m", 500.0, 1000.0, 1e5, 0.5)
        t0 = 300.0
        sensible = 2.0 * 500.0 * 700.0  # 7.0e5 J
        fusion = 2.0 * 1e5  # 2.0e5 J
        self.assertAlmostEqual(_temperature(0.0, 2.0, mat, t0), 300.0, places=12)
        self.assertAlmostEqual(_temperature(3.5e5, 2.0, mat, t0), 650.0, places=10)
        self.assertAlmostEqual(_temperature(sensible + 1e5, 2.0, mat, t0), 1000.0)
        top = sensible + fusion
        self.assertAlmostEqual(_temperature(top + 1e5, 2.0, mat, t0), 1100.0, places=10)
        # Continuity at both branch points.
        eps = 1e-6
        self.assertAlmostEqual(
            _temperature(sensible - eps, 2.0, mat, t0),
            _temperature(sensible + eps, 2.0, mat, t0),
            places=6,
        )
        self.assertAlmostEqual(
            _temperature(top - eps, 2.0, mat, t0),
            _temperature(top + eps, 2.0, mat, t0),
            places=6,
        )


class TrajectoryTests(unittest.TestCase):
    """Dynamics against analytic and library references."""

    def test_vertical_fall_matches_the_analytic_impact_speed(self) -> None:
        """Constant density, vertical fall: V^2 = V_t^2 - (V_t^2 - V_0^2) e^(-2ks).

        With ``k = rho / (2 beta)`` the linear equation ``d(V^2)/ds = 2 g(s) -
        2 k V^2`` has the integrating-factor solution evaluated here by Simpson
        quadrature with the exact ``g = mu / r^2``. The scale height is 1e9 m,
        so the density varies by 2e-6 over the fall; the residual is that and
        the RK4 error: 1e-4 relative.
        """
        frag = sphere_fragment("s", _MATS["steel_like"], 40.0, 0.5)
        height, v0 = 3000.0, 80.0
        atm = ExponentialAtmosphere(1.225, 1e9)
        fate = simulate_fragment(
            frag,
            breakup_altitude=height,
            flight_path_angle=-0.5 * math.pi,
            breakup_speed=v0,
            atmosphere=atm,
        )
        assert isinstance(fate, FragmentFate)
        k = 1.225 / (2.0 * frag.ballistic_coefficient)
        # V^2(s) = V_0^2 e^(-2ks) + 2 int_0^s g(s') e^(-2k(s - s')) ds' with the
        # exact g = mu / (R + h_0 - s')^2, by composite Simpson (n = 2000).
        n = 2000
        width = height / n
        total = 0.0
        for i in range(n + 1):
            sp = i * width
            weight = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            gravity = EARTH_MU / (EARTH_RADIUS + height - sp) ** 2
            total += weight * gravity * math.exp(-2.0 * k * (height - sp))
        v2 = v0 * v0 * math.exp(-2.0 * k * height) + 2.0 * total * width / 3.0
        self.assertEqual(fate.status, "impact")
        self.assertAlmostEqual(fate.impact_speed / math.sqrt(v2), 1.0, delta=1e-4)
        self.assertAlmostEqual(
            fate.impact_energy, 0.5 * 40.0 * v2, delta=1e-3 * v2 * 20
        )

    def test_heat_load_matches_simulate_entry_and_the_energy_identity(self) -> None:
        """With negligible emissivity E = f_s A Q, and Q equals ``simulate_entry``.

        The fragment is heavy enough not to melt, so ``T`` stays in the sensible
        branch and ``E' = f_s A Q'`` to the radiation of ``epsilon = 1e-9``
        (relative 1e-6 at most). ``Q`` is compared with the heat load of
        :func:`simulate_entry` run with the same ``beta`` and ``R_n``; the two
        codes use different step sequences, so 2e-3 relative is allowed.
        """
        mat = Material("cold", 500.0, 3000.0, 3e5, 1e-9)
        frag = sphere_fragment("s", mat, 300.0, 0.6, heating_shape_factor=0.8)
        outcome = simulate_fragment(
            frag, breakup_altitude=78e3, flight_path_angle=-0.01, record=True
        )
        assert isinstance(outcome, tuple)
        fate, grid = outcome
        q_final = grid[-1][5]
        e_final = grid[-1][6]
        self.assertAlmostEqual(
            e_final, 0.8 * frag.reference_area * q_final, delta=1e-6 * e_final
        )
        reference = simulate_entry(
            EntryVehicle(frag.ballistic_coefficient, frag.nose_radius),
            altitude=78e3,
            speed=7500.0,
            flight_path_angle=-0.01,
            terminal_altitude=0.0,
        )
        self.assertEqual(reference.status, "landed")
        self.assertAlmostEqual(q_final / reference.heat_load, 1.0, delta=2e-3)
        self.assertAlmostEqual(grid[-1][2], 0.0, delta=1e-3)  # ground located
        self.assertFalse(fate.demised)

    def test_energy_balance_against_an_independent_quadrature(self) -> None:
        """E(t_end) equals the trapezoid of P_in - P_rad from the recorded nodes.

        ``P_in`` and ``P_rad`` are recomputed from the nodes with the public
        :func:`sutton_graves_heat_flux` and the Stefan-Boltzmann law (not with
        the module's right-hand side). The body (``T_m = 3000`` K) stays in
        the sensible branch and radiates noticeably. The trapezoid error with
        ``max_step = 0.25`` s is measured below; 5e-3 relative is allowed.
        """
        mat = Material("hot", 500.0, 3000.0, 3e5, 0.8)
        frag = sphere_fragment("s", mat, 3.0, 0.5)
        atm = ExponentialAtmosphere()
        outcome = simulate_fragment(
            frag,
            breakup_altitude=90e3,
            flight_path_angle=-0.02,
            atmosphere=atm,
            max_step=0.25,
            record=True,
        )
        assert isinstance(outcome, tuple)
        fate, grid = outcome
        k_eff = SUTTON_GRAVES_EARTH_K
        powers = []
        peak_t = 0.0
        for _t, _theta, h, v, _g, _q, energy in grid:
            rho = atm.density(h)
            p_in = frag.reference_area * sutton_graves_heat_flux(
                rho, v, frag.nose_radius, constant=k_eff
            )
            temp = 250.0 + energy / (frag.mass * mat.specific_heat)
            peak_t = max(peak_t, temp)
            p_rad = 0.8 * STEFAN_BOLTZMANN * frag.surface_area * temp**4
            powers.append(p_in - p_rad)
        integral = sum(
            0.5 * (powers[i] + powers[i + 1]) * (grid[i + 1][0] - grid[i][0])
            for i in range(len(grid) - 1)
        )
        e_final = grid[-1][6]
        self.assertLess(peak_t, 3000.0)
        self.assertGreater(peak_t, 700.0)  # radiation is not negligible
        self.assertAlmostEqual(integral / e_final, 1.0, delta=5e-3)
        self.assertFalse(fate.demised)

    def test_step_refinement(self) -> None:
        """Halving the step changes the margin by less than 1e-3 (RK4, smooth)."""
        coarse = _fates()
        fine = _fates(max_step=0.5, resolution=0.05)
        for name, fate in coarse.items():
            self.assertAlmostEqual(
                fate.demise_margin, fine[name].demise_margin, delta=1e-3
            )
            self.assertAlmostEqual(
                fate.impact_speed / fine[name].impact_speed, 1.0, delta=1e-3
            )


class DemiseTests(unittest.TestCase):
    """Survival and demise of extreme fragments, and monotonicity."""

    def test_huge_heat_capacity_survives_and_thin_plate_demises(self) -> None:
        """A 500 kg steel sphere survives; a 0.4 kg thin aluminium plate demises."""
        heavy = sphere_fragment("heavy", _MATS["steel_like"], 500.0, 0.6)
        thin = plate_fragment("thin", _MATS["aluminium_like"], 0.4, 1.0, 1.0, 1e-4)
        for frag, expect_demise in ((heavy, False), (thin, True)):
            fate = simulate_fragment(
                frag, breakup_altitude=78e3, flight_path_angle=-0.01
            )
            assert isinstance(fate, FragmentFate)
            self.assertEqual(fate.status, "impact")
            self.assertEqual(fate.demised, expect_demise)
            self.assertEqual(fate.survived, not expect_demise)
            self.assertEqual(fate.demise_margin < 0.0, expect_demise)
        heavy_fate = simulate_fragment(
            heavy, breakup_altitude=78e3, flight_path_angle=-0.01
        )
        assert isinstance(heavy_fate, FragmentFate)
        self.assertGreater(heavy_fate.demise_margin, 0.5)
        self.assertTrue(math.isnan(heavy_fate.demise_altitude))

    def test_demise_altitude_lies_in_the_flight_band(self) -> None:
        """A demised fragment reports a first-crossing altitude below break-up."""
        fates = _fates()
        panel = fates["panel"]
        self.assertTrue(panel.demised)
        self.assertLess(panel.demise_altitude, 78e3)
        self.assertGreater(panel.demise_altitude, 20e3)
        self.assertGreaterEqual(panel.peak_energy, panel.demise_energy)

    def test_higher_breakup_altitude_means_more_demise(self) -> None:
        """The margin decreases strictly with break-up altitude (longer heating)."""
        previous: dict[str, float] = {}
        for altitude in (60e3, 70e3, 80e3, 90e3, 110e3):
            fates = _fates(breakup_altitude=altitude)
            for name in ("tank", "panel", "shaft"):
                if name in previous:
                    self.assertLess(
                        fates[name].demise_margin, previous[name], msg=f"{name}"
                    )
                previous[name] = fates[name].demise_margin

    def test_more_heating_means_more_demise(self) -> None:
        """The margin decreases strictly with the heating scale."""
        previous: dict[str, float] = {}
        for scale in (0.5, 0.8, 1.0, 1.5, 2.5):
            fates = _fates(heating_scale=scale)
            for name in ("tank", "panel", "shaft"):
                if name in previous:
                    self.assertLess(fates[name].demise_margin, previous[name])
                previous[name] = fates[name].demise_margin
        # A large enough scale demises the tank.
        self.assertTrue(_fates(heating_scale=6.0)["tank"].demised)

    def test_margin_is_continuous_across_the_switch(self) -> None:
        """Near the survive/demise switch the margin moves by a small amount.

        The margin is the continuous quantity; the demise decision flips only
        where it crosses zero.
        """
        panel = next(f for f in demonstration_fragments() if f.name == "panel")
        scales = [0.40 + 0.01 * i for i in range(41)]
        margins = []
        for s in scales:
            fate = simulate_fragment(
                panel, breakup_altitude=78e3, flight_path_angle=-0.01, heating_scale=s
            )
            assert isinstance(fate, FragmentFate)
            margins.append(fate.demise_margin)
        steps = [abs(b - a) for a, b in zip(margins, margins[1:], strict=False)]
        self.assertLess(max(steps), 0.03)
        flips = sum(1 for a, b in zip(margins, margins[1:], strict=False) if a * b < 0)
        self.assertEqual(flips, 1)

    def test_impact_energy_uses_the_ground_speed(self) -> None:
        """E_k = m V_i^2 / 2 for every demonstration fragment."""
        for frag in demonstration_fragments():
            fate = simulate_fragment(
                frag, breakup_altitude=78e3, flight_path_angle=-0.01
            )
            assert isinstance(fate, FragmentFate)
            self.assertAlmostEqual(
                fate.impact_energy, 0.5 * frag.mass * fate.impact_speed**2, places=8
            )


class AssessmentTests(unittest.TestCase):
    """Casualty area and expectation of fragment sets."""

    def test_single_survivor_hand_case(self) -> None:
        """Three heavy steel spheres: A_c = 3 (sqrt(A_p) + sqrt(A_f))^2."""
        heavy = sphere_fragment("heavy", _MATS["steel_like"], 500.0, 0.6, count=3)
        result = assess_reentry_casualty(
            [heavy],
            breakup_altitude=78e3,
            flight_path_angle=-0.01,
            person_area=_PERSON,
            population_density=25.0,
            energy_threshold=_ENERGY,
        )
        area = 3.0 * (0.6 + math.sqrt(math.pi * 0.36 / 4.0)) ** 2
        self.assertEqual(result.survivors, 3)
        self.assertEqual(result.casualty_survivors, 3)
        self.assertAlmostEqual(result.total_casualty_area, area, places=12)
        self.assertAlmostEqual(result.casualty_expectation, area * 25.0e-6, places=15)
        self.assertEqual(result.dominant_fragment(), "heavy")
        self.assertAlmostEqual(sum(result.contributions().values()), 1.0)

    def test_energy_threshold_removes_light_survivors(self) -> None:
        """Survivors below the energy threshold add no casualty area."""
        base = {
            "breakup_altitude": 78e3,
            "flight_path_angle": -0.01,
            "person_area": _PERSON,
            "population_density": _DENSITY,
        }
        frags = demonstration_fragments()
        low = assess_reentry_casualty(frags, energy_threshold=1.0, **base)
        high = assess_reentry_casualty(frags, energy_threshold=1e9, **base)
        self.assertGreater(low.total_casualty_area, 0.0)
        self.assertEqual(high.total_casualty_area, 0.0)
        self.assertEqual(high.casualty_expectation, 0.0)
        self.assertEqual(high.casualty_survivors, 0)
        self.assertEqual(high.survivors, low.survivors)
        self.assertIsNone(high.dominant_fragment())
        self.assertTrue(all(v == 0.0 for v in high.contributions().values()))

    def test_demonstration_nominal_outcome(self) -> None:
        """Tank and shaft survive, panel and boxes demise (measured nominal)."""
        result = assess_reentry_casualty(
            demonstration_fragments(),
            breakup_altitude=78e3,
            flight_path_angle=-0.01,
            person_area=_PERSON,
            population_density=_DENSITY,
            energy_threshold=_ENERGY,
        )
        self.assertIsInstance(result, CasualtyAssessment)
        self.assertEqual(result.survivors, 2)
        self.assertEqual(result.dominant_fragment(), "tank")
        fates = {f.name: f for f in result.fates}
        self.assertTrue(fates["panel"].demised and fates["box"].demised)
        self.assertFalse(fates["tank"].demised or fates["shaft"].demised)

    def test_assessment_failures(self) -> None:
        """Duplicate names, empty sets, wrong types and timeouts are errors."""
        base = {
            "breakup_altitude": 78e3,
            "flight_path_angle": -0.01,
            "person_area": _PERSON,
            "population_density": _DENSITY,
            "energy_threshold": _ENERGY,
        }
        frags = demonstration_fragments()
        with self.assertRaises(ValueError):
            assess_reentry_casualty([frags[0], frags[0]], **base)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            assess_reentry_casualty([], **base)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            assess_reentry_casualty(["tank"], **base)  # type: ignore[list-item,arg-type]
        with self.assertRaises(TypeError):
            assess_reentry_casualty("tank", **base)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            assess_reentry_casualty(frags, **{**base, "population_density": -1.0})  # type: ignore[arg-type]
        with self.assertRaises(RuntimeError):
            assess_reentry_casualty(frags, max_time=5.0, **base)

    def test_flight_failures_are_reported_in_status(self) -> None:
        """Timeout and skip-out are statuses of the fate, never silent."""
        frag = sphere_fragment("s", _MATS["steel_like"], 500.0, 0.6)
        fate = simulate_fragment(
            frag, breakup_altitude=78e3, flight_path_angle=-0.01, max_time=5.0
        )
        assert isinstance(fate, FragmentFate)
        self.assertEqual(fate.status, "timeout")
        self.assertFalse(fate.survived)
        skip = simulate_fragment(
            frag,
            breakup_altitude=120e3,
            flight_path_angle=-1e-4,
            breakup_speed=8200.0,
        )
        assert isinstance(skip, FragmentFate)
        self.assertEqual(skip.status, "skip_out")
        self.assertFalse(skip.survived)

    def test_simulate_validation(self) -> None:
        """Invalid arguments raise."""
        frag = demonstration_fragments()[0]
        with self.assertRaises(TypeError):
            simulate_fragment("x", breakup_altitude=78e3, flight_path_angle=-0.01)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            simulate_fragment(frag, breakup_altitude=78e3, flight_path_angle=0.0)
        with self.assertRaises(ValueError):
            simulate_fragment(frag, breakup_altitude=-1.0, flight_path_angle=-0.01)
        with self.assertRaises(ValueError):
            simulate_fragment(
                frag,
                breakup_altitude=78e3,
                flight_path_angle=-0.01,
                initial_temperature=5000.0,
            )
        with self.assertRaises(ValueError):
            simulate_fragment(
                frag, breakup_altitude=78e3, flight_path_angle=-0.01, heating_scale=0.0
            )
        with self.assertRaises(TypeError):
            simulate_fragment(
                frag,
                breakup_altitude=78e3,
                flight_path_angle=-0.01,
                record=1,  # type: ignore[arg-type]
            )


class ModelTests(unittest.TestCase):
    """Structure and failure handling of the probabilistic model."""

    def test_inputs_outputs_and_determinism(self) -> None:
        """Input names, QoI names, and a repeated evaluation agree exactly."""
        model = _small_model()
        self.assertEqual(
            tuple(model.inputs.names),
            (
                "breakup_altitude",
                "heating_scale",
                "melt_temperature_scale_aluminium_like",
                "ballistic_scale_panel",
            ),
        )
        point = {
            n: m for n, m in zip(model.inputs.names, (78e3, 1.0, 1.0, 1.0), strict=True)
        }
        first = model.evaluate(point)
        second = model.evaluate(point)
        self.assertEqual(first, second)
        for name in ("tank", "panel", "shaft"):
            for prefix in ("demise_margin_", "impact_energy_", "casualty_area_"):
                self.assertIn(prefix + name, first)
        self.assertAlmostEqual(
            first["casualty_expectation"], first["total_casualty_area"] * 40e-6
        )
        self.assertAlmostEqual(
            first["casualty_margin"], _LIMIT - first["casualty_expectation"]
        )
        self.assertAlmostEqual(
            first["casualty_ratio"], first["casualty_expectation"] / _LIMIT
        )
        self.assertAlmostEqual(
            first["total_casualty_area"],
            sum(first["casualty_area_" + n] for n in ("tank", "panel", "shaft")),
        )

    def test_nominal_model_matches_direct_assessment(self) -> None:
        """At the nominal point the model equals :func:`assess_reentry_casualty`."""
        model = _small_model()
        out = model.evaluate(
            dict(zip(model.inputs.names, (78e3, 1.0, 1.0, 1.0), strict=True))
        )
        direct = assess_reentry_casualty(
            _three(),
            breakup_altitude=78e3,
            flight_path_angle=-0.01,
            person_area=_PERSON,
            population_density=_DENSITY,
            energy_threshold=_ENERGY,
        )
        self.assertAlmostEqual(out["total_casualty_area"], direct.total_casualty_area)
        self.assertEqual(out["survivors"], float(direct.survivors))

    def test_smooth_expectation_tracks_the_switch(self) -> None:
        """Smoothing keeps the nominal value within the logistic weights' band."""
        model = _small_model()
        out = model.evaluate(
            dict(zip(model.inputs.names, (78e3, 1.0, 1.0, 1.0), strict=True))
        )
        # Nominal margins: tank 0.46 and shaft 0.24 (logistic weights 0.9999 and
        # 0.992 at w = 0.05), panel -0.33 (weight 1e-3 on a small area): the
        # smooth value is within 0.5 % of E_c (measured ratio 0.9989).
        ratio = out["smooth_casualty_expectation"] / out["casualty_expectation"]
        self.assertAlmostEqual(ratio, 1.0, delta=5e-3)

    def test_scale_inputs_expand_per_material_and_fragment(self) -> None:
        """A single distribution gives one input per material, fixed numbers none."""
        model = reentry_casualty_model(
            _three(),
            breakup_altitude=78e3,
            flight_path_angle=-0.01,
            person_area=_PERSON,
            population_density=_DENSITY,
            energy_threshold=_ENERGY,
            casualty_threshold=_LIMIT,
            fusion_heat_scale=Normal(1.0, 0.05),
            ballistic_scale=1.0,
        )
        self.assertEqual(
            sorted(model.inputs.names),
            [
                "fusion_heat_scale_aluminium_like",
                "fusion_heat_scale_steel_like",
                "fusion_heat_scale_titanium_like",
            ],
        )

    def test_model_validation(self) -> None:
        """No random input, unknown names and bad limits are rejected."""
        base = {
            "breakup_altitude": 78e3,
            "flight_path_angle": -0.01,
            "person_area": _PERSON,
            "population_density": _DENSITY,
            "energy_threshold": _ENERGY,
            "casualty_threshold": _LIMIT,
        }
        with self.assertRaises(ValueError):
            reentry_casualty_model(_three(), **base)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            reentry_casualty_model(
                _three(),
                **{**base, "ballistic_scale": {"nope": Normal(1.0, 0.1)}},  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _small_model(casualty_threshold=0.0)
        with self.assertRaises(ValueError):
            _small_model(smoothing_width=0.0)
        with self.assertRaises(TypeError):
            _small_model(breakup_altitude=True)
        with self.assertRaises(TypeError):
            reentry_casualty_model("tank", **base)  # type: ignore[arg-type]

    def test_out_of_domain_emissivity_is_a_model_failure(self) -> None:
        """A scaled emissivity above 1 (0.3 * 3.7 = 1.11) is a model failure."""
        ok_point = {
            "breakup_altitude": 78e3,
            "heating_scale": 1.0,
            "melt_temperature_scale_aluminium_like": 1.0,
            "ballistic_scale_panel": 1.0,
        }
        fine = _small_model(emissivity_scale={"aluminium_like": Uniform(0.5, 2.0)})
        point = {**ok_point, "emissivity_scale_aluminium_like": 1.5}
        self.assertIn("casualty_expectation", fine.evaluate(point))
        bad = _small_model(emissivity_scale={"aluminium_like": Uniform(3.5, 4.0)})
        point = {**ok_point, "emissivity_scale_aluminium_like": 3.7}
        with self.assertRaises((ValueError, RuntimeError)):
            bad.evaluate(point)


class ReliabilityTests(unittest.TestCase):
    """P(E_c > limit) by Monte Carlo, subset simulation and FORM; Sobol ranking.

    The event is dominated by one near-threshold switch (the aluminium panel
    survives in about 8 % of the draws). The measured behaviour: FORM on the
    step-valued ``E_c`` fails (zero gradient), subset simulation on it fails
    (tied levels), and both work on the continuous demise margin of the
    switching fragment, FORM only with a finite-difference step above the
    solver-noise floor.
    """

    reference = 0.085  # offline: 600 draws of this model, seed 3
    reference_se = 0.0114

    @classmethod
    def setUpClass(cls) -> None:
        """Build the model, its limit states and 150 common draws once."""
        cls.model = _small_model()
        cls.event = ModelLimitState(cls.model, "casualty_expectation", _LIMIT, "above")
        cls.margin = ModelLimitState(cls.model, "demise_margin_panel", 0.0, "above")
        draws = cls.model.inputs.sample(150, seed=3)
        cls.outputs = [cls.model.evaluate(p) for p in draws]

    def test_monte_carlo_frequency_is_within_the_reference_band(self) -> None:
        """150 draws: the exceedance frequency is within 3 combined standard errors.

        Reference: 600 draws of the same model (measured offline: 0.085,
        standard error 0.0114).
        """
        hits = sum(o["casualty_expectation"] > _LIMIT for o in self.outputs)
        p = hits / 150
        se = math.sqrt(max(p * (1 - p), 1e-4) / 150)
        self.assertGreater(hits, 0)
        self.assertLess(
            abs(p - self.reference), 3.0 * math.hypot(se, self.reference_se)
        )

    def test_the_event_is_the_survival_of_the_panel(self) -> None:
        """On every draw E_c > limit exactly when the panel survives (measured)."""
        for o in self.outputs:
            self.assertEqual(
                o["casualty_expectation"] > _LIMIT, o["demise_margin_panel"] > 0.0
            )

    def test_form_on_the_step_valued_expectation_fails_loudly(self) -> None:
        """The gradient of a step function vanishes: FORM raises, never returns."""
        with self.assertRaises(ValueError) as caught:
            form_hlrf(self.event)
        self.assertIn("gradient", str(caught.exception))

    def test_subset_simulation_on_the_step_valued_expectation_fails_loudly(
        self,
    ) -> None:
        """E_c takes few distinct values, so an intermediate level can tie.

        Measured: with 100 samples per level and seed 5 the run raises; with
        other sizes or seeds it may return a value, which is not evidence of
        adequacy for a step-valued response.
        """
        with self.assertRaises(ValueError):
            subset_simulation(self.event, samples_per_level=100, seed=5)

    def test_form_on_the_continuous_margin_agrees_with_monte_carlo(self) -> None:
        """FORM on the panel's demise margin: converged, within 0.03 of 0.085.

        Measured: P = 0.0953, beta = 1.309 in 7 iterations with
        ``relative_step = 1e-4`` (the default 1e-6 does not converge within 12
        iterations because the solver noise exceeds the perturbation). The
        band 0.03 is about 2.6 reference standard errors; FORM has a
        first-order error of its own for this curved switch.
        """
        form = form_hlrf(self.margin, relative_step=1e-4, tolerance=1e-3)
        self.assertTrue(form.converged)
        self.assertLess(abs(form.probability - self.reference), 0.03)
        # The heating scale carries the event; the melt temperature does not.
        self.assertGreater(form.importance_factors["heating_scale"], 0.5)
        self.assertLess(
            form.importance_factors["melt_temperature_scale_aluminium_like"], 0.1
        )

    def test_subset_simulation_on_the_continuous_margin(self) -> None:
        """Subset simulation (60 per level) on the margin is within 3 combined CoV.

        Measured: P = 0.10 with a coefficient-of-variation bound of 0.39.
        """
        subset = subset_simulation(self.margin, samples_per_level=60, seed=5)
        self.assertTrue(subset.converged)
        sigma = math.hypot(
            self.reference_se,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - self.reference), 3.0 * sigma)

    def test_sobol_ranking_puts_the_heating_scale_first(self) -> None:
        """The heating scale dominates the first-order indices of A_c (N = 16).

        Measured at N = 16: first-order 0.44 for the heating scale against
        0.16 for the panel's ballistic scale, and 0.0 for the aluminium melt
        temperature.
        """
        result = sobol_analysis(self.model, "total_casualty_area", 16, seed=9)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        self.assertEqual(max(first, key=lambda k: first[k]), "heating_scale")
        self.assertGreater(first["heating_scale"], 0.3)


class DesignLeverTests(unittest.TestCase):
    """Which fragment dominates E_c, and the effect of design-for-demise changes."""

    @classmethod
    def setUpClass(cls) -> None:
        """Evaluate the baseline and two variants on common draws (30 each)."""
        base = _three()
        tank = base[0]
        cooler = replace(
            tank.material, melt_temperature=0.65 * tank.material.melt_temperature
        )
        smaller = sphere_fragment("tank", tank.material, tank.mass, 0.5)
        variants = {
            "baseline": base,
            "lower_melt": (replace(tank, material=cooler), *base[1:]),
            "smaller_tank": (smaller, *base[1:]),
        }
        model = _small_model()
        draws = model.inputs.sample(30, seed=11)
        cls.results = {
            label: [_small_model(frags).evaluate(p) for p in draws]
            for label, frags in variants.items()
        }

    def _mean(self, label: str, quantity: str) -> float:
        """Return the sample mean of a quantity for a variant.

        Args:
            label: Variant name.
            quantity: QoI name.

        Returns:
            The mean over the common draws.

        """
        return statistics.fmean(r[quantity] for r in self.results[label])

    def test_the_tank_dominates_the_expected_casualty_area(self) -> None:
        """The tank carries most of E[A_c] although the panel has the larger area.

        Measured on 30 common draws: tank 1.49 m^2, shaft 0.54 m^2, panel
        0.11 m^2 on average (the panel's own casualty area is 1.66 m^2 but it
        survives rarely).
        """
        tank = self._mean("baseline", "casualty_area_tank")
        shaft = self._mean("baseline", "casualty_area_shaft")
        panel = self._mean("baseline", "casualty_area_panel")
        self.assertGreater(tank, 2.0 * shaft)
        self.assertGreater(shaft, panel)
        self.assertGreater(tank / (tank + shaft + panel), 0.6)
        frag_panel = next(f for f in _three() if f.name == "panel")
        self.assertGreater(casualty_area(frag_panel.reference_area, _PERSON), tank)

    def test_design_for_demise_reduces_the_expected_expectation(self) -> None:
        """Lowering the tank melt temperature by 35 % cuts E[E_c] clearly.

        Measured on the same draws: baseline 8.57e-5, melt temperature -35 %
        5.19e-5 (the tank survives in 13 of 30 draws instead of 30), tank
        diameter 0.5 m at equal mass 6.82e-5. The assertions use margins of
        10 % of the baseline.
        """
        base = self._mean("baseline", "casualty_expectation")
        cooler = self._mean("lower_melt", "casualty_expectation")
        smaller = self._mean("smaller_tank", "casualty_expectation")
        self.assertLess(cooler, 0.9 * base)
        self.assertLess(smaller, 0.9 * base)
        self.assertLess(cooler, smaller)
        # The lower-melt tank survives in a measurable fraction fewer draws.
        survive_base = sum(
            r["demise_margin_tank"] > 0 for r in self.results["baseline"]
        )
        survive_cool = sum(
            r["demise_margin_tank"] > 0 for r in self.results["lower_melt"]
        )
        self.assertLess(survive_cool, survive_base)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudyTests(unittest.TestCase):
    """Larger studies, run only with PROBSTAT_SLOW=1."""

    def test_reliability_methods_on_the_nine_input_model(self) -> None:
        """MC (400), subset simulation (100 per level) and FORM on the margin."""
        model = _full_model()
        event = ModelLimitState(model, "casualty_expectation", _LIMIT, "above")
        mc = model_monte_carlo(event, 400, seed=3)
        self.assertGreater(mc.probability, 0.03)
        self.assertLess(mc.probability, 0.25)
        margin = ModelLimitState(model, "demise_margin_panel", 0.0, "above")
        form = form_hlrf(margin, relative_step=1e-4, tolerance=1e-3)
        self.assertTrue(form.converged)
        self.assertLess(abs(form.probability - mc.probability), 0.05)
        with self.assertRaises(ValueError):
            form_hlrf(event)

    def test_sobol_ranking_is_stable_with_more_samples(self) -> None:
        """With N = 64 the heating scale still has the largest first-order index."""
        result = sobol_analysis(_full_model(), "total_casualty_area", 64, seed=9)
        first = {
            n: i.estimate
            for n, i in zip(result.input_names, result.first_order, strict=True)
        }
        self.assertEqual(max(first, key=lambda k: first[k]), "heating_scale")


if __name__ == "__main__":
    unittest.main()
