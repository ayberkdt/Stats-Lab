"""Spacecraft mission-risk laboratory: one synthetic mission, five stages, one seed.

Run with ``PYTHONPATH=src python examples/spacecraft_mission_risk_lab.py``. The
lab follows one invented mission from launch to return and applies the
library's uncertainty tools to the physical models of ``probstat.aerospace`` at
each stage: (a) ascent: dispersion of the orbit insertion of the synthetic
launch vehicle (3-sigma intervals, a dispersion ellipse, an RSS dispersion
budget); (b) the propellant tank: a probabilistic buckling knockdown against
the NASA SP-8007 design value and the probability of buckling by FORM and
Monte Carlo; (c) on orbit: the probability of no penetration (PNP) by
micrometeoroids and debris with epistemic uncertainty on the flux and on the
ballistic limit, and the shield thickness that meets a PNP requirement with and
without an epistemic margin; (d) engine health during a burn: a digital twin,
CUSUM detection of an injected efficiency fault and the remaining-useful-life
interval; (e) entry: the probability that the heat shield is too thin by FORM,
subset simulation and Monte Carlo, and the entry corridor. The lab ends with a
mission risk summary table. Every number is printed; with Matplotlib installed
two figures are written to ``examples/generated/``.

EVERY CASE IS SYNTHETIC. The vehicle, the tank, the spacecraft shield, the
engine, the capsule and every flux, ballistic-limit coefficient, ablator
property and noise level are invented demonstration values. In particular the
MMOD flux and the ballistic-limit coefficients are the library's synthetic
demonstration set (not ORDEM, not MASTER, not a published equation). None of
the numbers is data from a real vehicle, spacecraft or engine, and none is a
statement about the risk of any real mission.

Definitions:
    insertion: the burnout orbit of the ascent; it succeeds when the perigee
        altitude is at least ``PERIGEE_MIN`` and the apogee altitude lies in
        ``[APOGEE_MIN, APOGEE_MAX]``.
    dispersion: the spread of an output under the input spreads; the 3-sigma
        interval is the normal interval with 99.73 % coverage.
    knockdown: the ratio ``lambda`` of the buckling load of the imperfect
        shell to the classical load; ``gamma`` is the SP-8007 design value and
        ``k_p`` the probabilistic knockdown exceeded with probability ``1 - p``
        for a random imperfection amplitude.
    buckling: failure is ``P_cr - P_applied <= 0`` with the Koiter law for the
        imperfect cylinder.
    PNP: probability of no penetration ``exp(-N)``, ``N`` the expected number
        of penetrations over the exposure; the epistemic factors ``f`` (flux)
        and ``g`` (critical diameter) make PNP itself a random variable.
    requirement: the PNP must be at least ``PNP_REQUIREMENT``.
    health: the combustion-efficiency state ``eta`` of the engine twin;
        a fault is a step in ``eta``; the CUSUM statistic is run on the
        standardised pressure innovations of the filter.
    RUL: remaining useful life, the time until ``eta`` first falls to the
        failure threshold ``ETA_FAILURE``.
    margin: ``design thickness - required thickness`` of the heat shield; the
        failure event is a required thickness above the design thickness.
    corridor: the interval of entry flight-path angles between the skip-out
        boundary and the load-factor limit.

Assumptions:
    - Every input spread is an assumption, not a measurement; inputs are
      independent unless a model states otherwise.
    - The ascent is a planar point-mass gravity turn on a non-rotating
      spherical Earth with a 1976 standard atmosphere; the RSS budget is a
      linear theory and is compared with Monte Carlo, not trusted.
    - The tank is an isotropic cylinder in axial compression with one
      axisymmetric imperfection (Koiter asymptotic law); the verdict "SP-8007
      is conservative" holds only for the assumed imperfection law and the
      lab prints the case where it reverses.
    - MMOD impacts form a homogeneous Poisson process; the epistemic factors
      are lognormal and independent of each other; penetration is
      ``d >= d_c``; the demonstration flux and ballistic limit are synthetic.
    - The engine twin is the matched synthetic plant of the library (first-order
      chamber lag, efficiency as a Wiener process with drift, Gaussian
      noise); the fault is an abrupt efficiency step unknown to the filter.
    - The entry is a planar point-mass capsule with constant lift-to-drag
      ratio, Sutton-Graves convective heating and a first-order ablative
      shield sizing with synthetic material constants; no radiative heating.
    - The stages are not coupled: the tank load, the exposure, the burn and the
      entry state are separate synthetic design values, not outputs of the
      ascent run, so the summary table does not combine the stages into one
      mission probability.

Equations:
    EQ-AERO-ASC-007: probabilistic ascent model.
    EQ-AERO-ASC-008: dispersion intervals, ellipse and RSS budget.
    EQ-AERO-BUCK-004: SP-8007 axial knockdown.
    EQ-AERO-BUCK-005: probabilistic knockdown for a random imperfection.
    EQ-AERO-BUCK-006: probabilistic buckling model and its margin.
    EQ-AERO-MMOD-006: distribution of PNP over the epistemic factors.
    EQ-AERO-MMOD-008: shield thickness for a PNP requirement.
    EQ-AERO-PHM-002: joint state-parameter estimation of the engine twin.
    EQ-AERO-PHM-003: CUSUM monitor on the standardised innovations.
    EQ-AERO-PHM-007: predictive remaining-useful-life distribution.
    EQ-AERO-ENTRY-009: probabilistic entry and heat-shield model.
    EQ-AERO-ENTRY-010: entry corridor by bisection.
    EQ-REL-PF-001: failure probability.
    EQ-REL-FORM-001: FORM design point and reliability index.
    EQ-REL-SUBSET-001: subset simulation.

Diagnostics:
    - FORM is compared with a crude Monte Carlo count whose binomial standard
      error is printed (tank, entry); subset simulation prints its
      coefficient of variation and convergence.
    - The ascent RSS budget prints its ratio to the Monte Carlo standard
      deviation and its nonlinearity; the dispersion ellipse prints the
      fraction of the sample it contains.
    - The mean PNP of the nested epistemic propagation is checked against a
      Gauss-Hermite quadrature and the thickness designs report the PNP they
      achieve.
    - The engine fault is also tested on fault-free records and on a set of
      seeds (false-alarm runs, detection probability, delay); the twin's
      efficiency estimate is compared with the simulated truth.
    - The corridor reports the number of trajectory integrations and the
      load factor at both bounds.

Cost and error:
    - Ascent: ``ascent_mc`` runs for the intervals, ``ellipse_runs`` for the
      ellipse and ``1 + 4 d + budget_mc`` for the budget (``d = 8`` inputs).
    - Tank: FORM costs a few dozen runs; ``tank_mc`` Monte Carlo runs.
    - MMOD: ``mmod_samples`` epistemic draws (the Poisson part is analytic),
      ``design_samples`` draws per thickness evaluation of the quantile design.
    - Engine health: one 110-step record, then ``health_runs`` fault and
      fault-free record pairs.
    - Entry: FORM, ``entry_subset`` samples per subset level, ``entry_mc``
      crude Monte Carlo runs and about 30 corridor integrations.
    - Monte Carlo errors are binomial standard errors or sample standard
      errors of the mean and are printed with the estimates; subset-simulation
      and filter results are those of one seed and are not averaged
      over seeds.

References:
    REF-CURTIS-2014: orbital mechanics and the ascent model.
    REF-NASA-SP-8007-2020: buckling knockdown of thin cylinders.
    REF-MELCHERS-BECK-2018: FORM and structural reliability.
    REF-AU-BECK-2001: subset simulation.
    REF-COUR-PALAIS-1987: single-wall ballistic-limit form.
    REF-CHRISTIANSEN-1993: shielded ballistic limits.
    REF-NASA-PRA-2011: probabilistic risk assessment procedures.
    REF-SMITH-UQ-2024: epistemic uncertainty propagation.
    REF-PAGE-1954: CUSUM.
    REF-SAXENA-GOEBEL-SIMON-EKLUND-2008: prognostics.
    REF-ALLEN-EGGERS-1958: ballistic entry.
    REF-SUTTON-GRAVES-1971: stagnation-point heating.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.aerospace import debris_risk as dr
from probstat.aerospace.ascent import (
    DragModel,
    GuidanceSpec,
    InsertionWindow,
    StageSpec,
    VehicleSpec,
    ascent_model,
    dispersion_budget,
    dispersion_ellipse,
    dispersion_intervals,
)
from probstat.aerospace.entry import (
    EntryVehicle,
    HeatShield,
    entry_corridor,
    entry_model,
    simulate_entry,
)
from probstat.aerospace.health_monitoring import (
    EnginePhmSpec,
    StepFault,
    assess_engine_health,
    detection_operating_characteristic,
    simulate_plant,
)
from probstat.aerospace.shell_buckling import (
    classical_buckling_load,
    compare_knockdowns,
    koiter_coefficient,
    lognormal_from_mean_cov,
    probabilistic_knockdown,
    shell_buckling_model,
    sp8007_axial_knockdown,
    sp8007_buckling_load,
)
from probstat.distributions import Normal
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)

# Root seed; every stage below derives its own seeds from it.
SEED = 2028
OUTPUT = Path(__file__).parent / "generated"

# (a) ascent: synthetic two-stage vehicle and insertion window.
FIRST_STAGE = StageSpec(4000.0, 42000.0, 1.4e6, 300.0, 1.0, 3.0, coast_after=2.0)
SECOND_STAGE = StageSpec(800.0, 4500.0, 60e3, 340.0, 0.1, 3.0)
VEHICLE = VehicleSpec((FIRST_STAGE, SECOND_STAGE), 500.0)
GUIDANCE = GuidanceSpec(8.0, 0.5)
DRAG = DragModel.tabulated(
    [(0.0, 0.3), (0.8, 0.3), (1.2, 0.5), (2.0, 0.4), (5.0, 0.25), (50.0, 0.2)]
)
PERIGEE_MIN = 125.0e3  # m, synthetic insertion window
APOGEE_MIN = 4.5e6  # m
APOGEE_MAX = 5.8e6  # m
WINDOW = InsertionWindow(PERIGEE_MIN, APOGEE_MIN, APOGEE_MAX)

# (b) tank: synthetic aluminium-class cylinder.
TANK_E = 71.0e9  # Pa
TANK_THICKNESS = 3.0e-3  # m
TANK_RADIUS = 1.5  # m
TANK_NU = 0.33
TANK_LOAD = 1.0e6  # N, synthetic design axial load
IMPERFECTION_MEAN = 0.3e-3  # m, mean imperfection amplitude (0.1 t)
IMPERFECTION_COV = 0.5
KNOCKDOWN_PROBABILITY = 0.01  # p of the probabilistic knockdown

# (c) MMOD: synthetic spacecraft shield and requirement.
PROJECTILE_DENSITY = 2700.0  # kg/m^3
WALL_DENSITY = 2800.0  # kg/m^3
EXPOSURE_YEARS = 10.0
PNP_REQUIREMENT = 0.95
DESIGN_WALL = 6.0e-3  # m, baseline shield thickness
EPISTEMIC = dr.EpistemicFactors(0.6, 0.15)  # flux and ballistic-limit log-sigmas
CREDIBILITY = 0.9

# (d) engine health.
ETA_FAILURE = 0.85
FAULT = StepFault(70, -0.03)
RECORD_STEPS = 110
CUSUM_THRESHOLD = 5.0

# (e) entry: synthetic Apollo-/Orion-class capsule and ablator.
CAPSULE = EntryVehicle.from_mass(5000.0, 1.30, 12.0, 4.7, 0.3)
ABLATOR = HeatShield(500.0, 2.5e7, 1.2, 2.7e-7, 2500.0, 450.0, 250.0, 0.027)
ENTRY_SPEED = 11000.0  # m/s
ENTRY_ANGLE_DEG = -7.9
ENTRY_ANGLE_SD_DEG = 0.12
G_LIMIT = 15.0
ENTRY_MAX_STEP = 3.0  # s, integrator step bound
ENTRY_RESOLUTION = 0.5  # scale heights per step


@dataclass(frozen=True)
class Budget:
    """Sample sizes of one run of the lab.

    Attributes:
        ascent_mc: Monte Carlo runs of the ascent dispersion intervals.
        ellipse_runs: Runs of the perigee-burnout altitude dispersion ellipse.
        budget_mc: Monte Carlo runs of the RSS budget reference.
        tank_mc: Crude Monte Carlo runs of the buckling probability.
        mmod_samples: Epistemic draws of the PNP distribution.
        design_samples: Epistemic draws per evaluation of the quantile design.
        health_runs: Fault and fault-free record pairs of the detection study.
        entry_subset: Samples per level of the entry subset simulation.
        entry_mc: Crude Monte Carlo runs of the heat-shield probability.
        sweep_points: Flight-path angles of the entry trade-off curve.

    References:
        [EQ-AERO-ASC-008]. No equation is evaluated.

    """

    ascent_mc: int = 800
    ellipse_runs: int = 400
    budget_mc: int = 300
    tank_mc: int = 20_000
    mmod_samples: int = 4000
    design_samples: int = 2000
    health_runs: int = 15
    entry_subset: int = 200
    entry_mc: int = 1000
    sweep_points: int = 14


FULL = Budget()
QUICK = Budget(
    ascent_mc=60,
    ellipse_runs=40,
    budget_mc=30,
    tank_mc=4000,
    mmod_samples=1500,
    design_samples=500,
    health_runs=4,
    entry_subset=60,
    entry_mc=100,
    sweep_points=6,
)

Row = tuple[str, str, str, str]


def headline(text: str) -> None:
    """Print a section heading.

    Args:
        text: Heading.

    References:
        [EQ-AERO-ASC-008]. No equation is evaluated.

    """
    print(f"\n== {text} ==")


# ---------------------------------------------------------------------------
# (a) ascent
# ---------------------------------------------------------------------------


def ascent_dispersion_model() -> Any:
    """Return the synthetic ascent model with eight uncertain inputs.

    Thrust, specific impulse and propellant scales are normal with 0.2 %
    spread, the dry-mass scale 1 %, the drag scale 5 %, the density scale 3 %,
    the kick angle 0.004 rad and the kick time 0.1 s; the window adds the
    ``insertion_success`` output.

    Returns:
        A probabilistic model of the ascent.

    References:
        [EQ-AERO-ASC-007], [REF-CURTIS-2014].

    """
    return ascent_model(
        VEHICLE,
        GUIDANCE,
        DRAG,
        thrust_scale=Normal(1.0, 0.002),
        isp_scale=Normal(1.0, 0.002),
        propellant_scale=Normal(1.0, 0.002),
        dry_mass_scale=Normal(1.0, 0.01),
        drag_scale=Normal(1.0, 0.05),
        density_scale=Normal(1.0, 0.03),
        pitch_kick_angle=Normal(0.5, 0.004),
        pitch_kick_time=Normal(8.0, 0.1),
        window=WINDOW,
        max_step=4.0,
    )


def ascent_part(budget: Budget) -> dict[str, Any]:
    """Disperse the insertion orbit, fit its ellipse and test the RSS budget.

    Args:
        budget: Sample sizes.

    Returns:
        Intervals, ellipse, budget, insertion probability and summary rows.

    References:
        [EQ-AERO-ASC-007], [EQ-AERO-ASC-008], [REF-CURTIS-2014].

    """
    headline("(a) ascent: insertion dispersion")
    model = ascent_dispersion_model()
    names = ["perigee_altitude", "apogee_altitude", "insertion_success"]
    intervals = dispersion_intervals(model, names, budget.ascent_mc, seed=SEED)
    for item in intervals[:2]:
        low, high = item.normal_interval
        e_low, e_high = item.empirical_interval
        print(
            f"{item.quantity}: mean={item.mean / 1e3:.1f} km "
            f"sd={item.standard_deviation / 1e3:.1f} km "
            f"3-sigma [{low / 1e3:.1f}, {high / 1e3:.1f}] km "
            f"empirical [{e_low / 1e3:.1f}, {e_high / 1e3:.1f}] km "
            f"(n={item.samples}, se of the mean "
            f"{item.standard_deviation / math.sqrt(item.samples) / 1e3:.2f} km)"
        )
    success = intervals[2]
    p_in = success.mean
    se_in = math.sqrt(p_in * (1.0 - p_in) / success.samples)
    print(
        f"P(insertion in window: perigee >= {PERIGEE_MIN / 1e3:.0f} km, apogee "
        f"{APOGEE_MIN / 1e3:.0f}-{APOGEE_MAX / 1e3:.0f} km) = {p_in:.3f} "
        f"(binomial se {se_in:.3f}, n={success.samples})"
    )
    points = model.inputs.sample(budget.ellipse_runs, seed=SEED + 1)
    outputs = model.evaluate_many(points)
    ellipse = dispersion_ellipse(
        outputs["perigee_altitude"], outputs["burnout_altitude"]
    )
    print(
        f"perigee-burnout ellipse ({ellipse.coverage:.2%} Gaussian): semi-axes "
        f"{ellipse.semi_axes[0] / 1e3:.1f} and {ellipse.semi_axes[1] / 1e3:.1f} km "
        f"correlation={ellipse.correlation:.3f} "
        f"empirical coverage={ellipse.empirical_coverage:.3f} "
        f"(n={budget.ellipse_runs})"
    )
    rss = dispersion_budget(model, "perigee_altitude", budget.budget_mc, seed=SEED + 2)
    order = sorted(range(len(rss.input_names)), key=lambda i: -rss.contributions[i])
    print(
        f"RSS budget of the perigee: rss={rss.rss / 1e3:.2f} km vs Monte Carlo sd="
        f"{rss.monte_carlo_std / 1e3:.2f} km (ratio {rss.ratio:.3f}, "
        f"nonlinearity {rss.nonlinearity:.3f}, mean shift "
        f"{rss.mean_shift / 1e3:.2f} km, n={budget.budget_mc})"
    )
    for i in order[:4]:
        print(
            f"  {rss.input_names[i]:16s} 1-sigma contribution "
            f"{rss.contributions[i] / 1e3:.2f} km"
        )
    perigee = intervals[0]
    rows: list[Row] = [
        (
            "ascent",
            "perigee altitude (km)",
            f"{perigee.mean / 1e3:.1f}",
            f"sd {perigee.standard_deviation / 1e3:.1f}; 3-sigma "
            f"[{perigee.normal_interval[0] / 1e3:.0f}, "
            f"{perigee.normal_interval[1] / 1e3:.0f}]",
        ),
        (
            "ascent",
            "P(insertion in window)",
            f"{p_in:.3f}",
            f"binomial se {se_in:.3f} (n={success.samples})",
        ),
    ]
    return {
        "intervals": intervals,
        "ellipse": ellipse,
        "budget": rss,
        "p_insertion": p_in,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# (b) propellant tank
# ---------------------------------------------------------------------------


def tank_model(applied_load: float) -> Any:
    """Return the buckling model of the tank for one applied axial load.

    Args:
        applied_load: Axial load in N.

    Returns:
        A probabilistic model of the tank (lognormal modulus, thickness,
        imperfection and load).

    References:
        [EQ-AERO-BUCK-006], [REF-NASA-SP-8007-2020].

    """
    return shell_buckling_model(
        youngs_modulus=lognormal_from_mean_cov(TANK_E, 0.05),
        thickness=lognormal_from_mean_cov(TANK_THICKNESS, 0.03),
        radius=TANK_RADIUS,
        imperfection=lognormal_from_mean_cov(IMPERFECTION_MEAN, IMPERFECTION_COV),
        applied_load=lognormal_from_mean_cov(applied_load, 0.1),
        poisson_ratio=TANK_NU,
    )


def tank_part(budget: Budget) -> dict[str, Any]:
    """Compare the probabilistic knockdown with SP-8007 and estimate P(buckling).

    Args:
        budget: Sample sizes.

    Returns:
        Knockdowns, FORM and Monte Carlo results and summary rows.

    References:
        [EQ-AERO-BUCK-004], [EQ-AERO-BUCK-005], [EQ-AERO-BUCK-006],
        [EQ-REL-FORM-001], [REF-NASA-SP-8007-2020].

    """
    headline("(b) propellant tank: buckling knockdown and probability")
    coefficient = koiter_coefficient(TANK_NU)
    law = lognormal_from_mean_cov(IMPERFECTION_MEAN, IMPERFECTION_COV)
    ratio = TANK_RADIUS / TANK_THICKNESS
    classical = classical_buckling_load(TANK_E, TANK_THICKNESS, TANK_RADIUS, TANK_NU)
    design = sp8007_buckling_load(TANK_E, TANK_THICKNESS, TANK_RADIUS, TANK_NU)
    gamma = sp8007_axial_knockdown(TANK_RADIUS, TANK_THICKNESS)
    k_p = probabilistic_knockdown(
        law, TANK_THICKNESS, KNOCKDOWN_PROBABILITY, coefficient
    )
    print(
        f"tank R/t={ratio:.0f}: classical load {classical / 1e6:.3f} MN, "
        f"SP-8007 design load {design / 1e6:.3f} MN (gamma={gamma:.3f}), "
        f"Koiter constant c={coefficient:.3f}"
    )
    print(
        f"probabilistic knockdown k_p (p={KNOCKDOWN_PROBABILITY:g}, lognormal "
        f"imperfection mean {IMPERFECTION_MEAN * 1e3:.2f} mm, c.o.v. "
        f"{IMPERFECTION_COV}): {k_p:.3f} vs SP-8007 {gamma:.3f}"
    )
    table = compare_knockdowns(
        law, TANK_RADIUS, (200.0, 500.0, 1000.0), KNOCKDOWN_PROBABILITY, coefficient
    )
    for r, k, g, safe in zip(
        table.radius_to_thickness,
        table.probabilistic,
        table.sp8007,
        table.sp8007_conservative,
        strict=True,
    ):
        print(f"  R/t={r:6.0f} k_p={k:.3f} SP-8007={g:.3f} SP-8007 below k_p: {safe}")
    large = lognormal_from_mean_cov(5.0 * IMPERFECTION_MEAN, IMPERFECTION_COV)
    reverse = compare_knockdowns(
        large, TANK_RADIUS, (500.0,), KNOCKDOWN_PROBABILITY, coefficient
    )
    print(
        f"  with a five times larger imperfection (mean "
        f"{5.0 * IMPERFECTION_MEAN * 1e3:.1f} mm): k_p={reverse.probabilistic[0]:.3f} "
        f"vs SP-8007 {reverse.sp8007[0]:.3f}, SP-8007 below k_p: "
        f"{reverse.sp8007_conservative[0]}"
    )
    state = ModelLimitState(tank_model(TANK_LOAD), "margin", 0.0, "below")
    form = form_hlrf(state)
    mc = model_monte_carlo(state, budget.tank_mc, seed=SEED + 3)
    print(
        f"P(buckling) at {TANK_LOAD / 1e6:.2f} MN: FORM={form.probability:.3e} "
        f"(beta={form.beta:.3f}, converged={form.converged}, runs="
        f"{form.evaluations}); Monte Carlo={mc.probability:.3e} "
        f"se={mc.standard_error:.2e} (n={budget.tank_mc})"
    )
    weights = ", ".join(
        f"{name}={value:.2f}" for name, value in form.importance_factors.items()
    )
    print(f"  FORM importance factors: {weights}")
    at_design = form_hlrf(ModelLimitState(tank_model(design), "margin", 0.0, "below"))
    print(
        f"P(buckling) when loaded to the SP-8007 design load {design / 1e6:.3f} MN: "
        f"FORM={at_design.probability:.3e} (beta={at_design.beta:.3f})"
    )
    rows: list[Row] = [
        (
            "tank",
            "knockdown k_p vs SP-8007",
            f"{k_p:.3f} vs {gamma:.3f}",
            f"p={KNOCKDOWN_PROBABILITY:g}; holds only for the assumed imperfection law",
        ),
        (
            "tank",
            f"P(buckling) at {TANK_LOAD / 1e6:.1f} MN",
            f"{mc.probability:.2e}",
            f"MC se {mc.standard_error:.1e}; FORM {form.probability:.2e}",
        ),
    ]
    return {
        "k_p": k_p,
        "gamma": gamma,
        "table": table,
        "form": form,
        "mc": mc,
        "at_design": at_design,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# (c) micrometeoroid and orbital debris
# ---------------------------------------------------------------------------


def shield_surfaces(wall: float) -> list[dr.Surface]:
    """Return the two exposed surfaces for a baseline wall thickness.

    The radiator has wall ``t`` and the tank wall ``1.5 t``; both use the
    SYNTHETIC demonstration single-wall limit and flux of the library.

    Args:
        wall: Baseline thickness ``t`` in m.

    Returns:
        The radiator (12 m^2) and the tank wall (8 m^2).

    References:
        [EQ-AERO-MMOD-008], [REF-COUR-PALAIS-1987].

    """
    limit = dr.demonstration_single_wall_limit()
    flux = dr.demonstration_flux()
    return [
        dr.Surface(  # datawiz: izin spacecraft surface, not a 3D plot
            "radiator", 12.0, dr.SingleWallShield(limit, wall, WALL_DENSITY), flux
        ),
        dr.Surface(  # datawiz: izin spacecraft surface, not a 3D plot
            "tank wall",
            8.0,
            dr.SingleWallShield(limit, 1.5 * wall, WALL_DENSITY),
            flux,
        ),
    ]


def impact_environment() -> dr.ImpactEnvironment:
    """Return the synthetic impact environment (isotropic angles, three speeds).

    Returns:
        Eight equal-probability angle bins and speeds 7, 10 and 13 km/s with
        weights 0.25, 0.5 and 0.25.

    References:
        [EQ-AERO-MMOD-006], [REF-NASA-PRA-2011].

    """
    angles, weights = dr.isotropic_angle_bins(8)
    return dr.ImpactEnvironment.product(
        angles,
        weights,
        (7000.0, 10000.0, 13000.0),
        (0.25, 0.5, 0.25),
        PROJECTILE_DENSITY,
    )


def mmod_part(budget: Budget) -> dict[str, Any]:
    """Propagate epistemic uncertainty to PNP and size the shield for the requirement.

    Args:
        budget: Sample sizes.

    Returns:
        The PNP distribution, the two designs and summary rows.

    References:
        [EQ-AERO-MMOD-006], [EQ-AERO-MMOD-008], [REF-SMITH-UQ-2024].

    """
    headline("(c) on orbit: micrometeoroid and debris risk (SYNTHETIC coefficients)")
    print(
        "WARNING: the flux and the ballistic-limit coefficients below are the "
        "library's synthetic demonstration set, not ORDEM/MASTER and not a "
        "published equation; the numbers illustrate the method only."
    )
    environment = impact_environment()
    surfaces = shield_surfaces(DESIGN_WALL)
    risk = dr.penetration_risk(surfaces, environment, EXPOSURE_YEARS)
    print(
        f"nominal design (wall {DESIGN_WALL * 1e3:.1f} mm, {EXPOSURE_YEARS:g} yr): "
        f"N={risk.expected_penetrations:.4f} PNP={risk.pnp:.4f}; requirement "
        f"PNP >= {PNP_REQUIREMENT:g}"
    )
    distribution = dr.pnp_distribution(
        surfaces,
        environment,
        EXPOSURE_YEARS,
        EPISTEMIC,
        samples=budget.mmod_samples,
        level=CREDIBILITY,
        requirement=PNP_REQUIREMENT,
        seed=SEED + 4,
    )
    print(
        f"epistemic factors: flux log-sigma {EPISTEMIC.flux_sigma}, "
        f"ballistic-limit log-sigma {EPISTEMIC.limit_sigma}"
    )
    print(
        f"PNP over {distribution.samples} epistemic draws: "
        f"mean={distribution.mean_pnp:.4f} "
        f"(se {distribution.mean_pnp_standard_error:.4f}) "
        f"sd={distribution.std_pnp:.4f} "
        f"nominal={distribution.pnp_at_median_inputs:.4f} "
        f"PNP at the mean factors={distribution.pnp_at_mean_inputs:.4f}"
    )
    print(
        f"  {CREDIBILITY:.0%} credible lower PNP={distribution.pnp_lower:.4f} "
        f"(N quantile {distribution.expected_penetrations_quantile:.4f}); "
        f"P(PNP < {PNP_REQUIREMENT:g}) = "
        f"{distribution.probability_below_requirement:.3f}"
    )
    quadrature = dr.mean_pnp_quadrature(
        surfaces, environment, EXPOSURE_YEARS, EPISTEMIC, order=30
    )
    print(
        f"  Gauss-Hermite check of the mean PNP: {quadrature.mean_pnp:.4f} "
        f"(difference {distribution.mean_pnp - quadrature.mean_pnp:+.4f}); Jensen gap "
        f"{distribution.jensen_gap:+.4f}, Poisson Jensen gap "
        f"{distribution.poisson_jensen_gap:+.4f}"
    )
    bracket = (1.0e-4, 0.1)
    nominal = dr.required_thickness(
        shield_surfaces,
        environment,
        EXPOSURE_YEARS,
        PNP_REQUIREMENT,
        bracket=bracket,
        basis="nominal",
    )
    margin = dr.required_thickness(
        shield_surfaces,
        environment,
        EXPOSURE_YEARS,
        PNP_REQUIREMENT,
        bracket=bracket,
        basis="quantile",
        factors=EPISTEMIC,
        level=CREDIBILITY,
        samples=budget.design_samples,
        seed=SEED + 5,
    )
    for label, design in (
        ("nominal", nominal),
        (f"{CREDIBILITY:.0%} quantile", margin),
    ):
        print(
            f"required wall, {label:12s}: {design.thickness * 1e3:.2f} mm "
            f"(nominal PNP there {design.nominal_pnp:.4f}, criterion "
            f"{design.achieved_metric:.4f}, converged={design.converged}, "
            f"{design.iterations} iterations)"
        )
    print(
        f"  epistemic margin adds {(margin.thickness / nominal.thickness - 1.0):.1%} "
        f"to the wall ({(margin.thickness - nominal.thickness) * 1e3:.2f} mm; "
        f"{budget.design_samples} draws per evaluation)"
    )
    rows: list[Row] = [
        (
            "MMOD",
            f"PNP, {EXPOSURE_YEARS:g} yr, {DESIGN_WALL * 1e3:.0f} mm wall",
            f"{distribution.mean_pnp:.3f}",
            f"epistemic mean (se {distribution.mean_pnp_standard_error:.4f}); "
            f"P(PNP<{PNP_REQUIREMENT:g})={distribution.probability_below_requirement:.2f}",
        ),
        (
            "MMOD",
            f"wall for PNP >= {PNP_REQUIREMENT:g} (mm)",
            f"{nominal.thickness * 1e3:.2f} -> {margin.thickness * 1e3:.2f}",
            f"nominal -> {CREDIBILITY:.0%} epistemic margin",
        ),
    ]
    return {
        "distribution": distribution,
        "risk": risk,
        "nominal": nominal,
        "margin": margin,
        "quadrature": quadrature,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# (d) engine health
# ---------------------------------------------------------------------------


def health_part(budget: Budget) -> dict[str, Any]:
    """Detect an injected efficiency fault and predict the remaining useful life.

    Args:
        budget: Sample sizes.

    Returns:
        The assessment, the study and summary rows.

    References:
        [EQ-AERO-PHM-002], [EQ-AERO-PHM-003], [EQ-AERO-PHM-007],
        [REF-PAGE-1954], [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    headline("(d) engine health during a burn: twin, CUSUM, remaining useful life")
    spec = EnginePhmSpec()
    plant = simulate_plant(spec, RECORD_STEPS, seed=SEED, fault=FAULT)
    out = assess_engine_health(
        spec,
        plant.observations,
        failure_threshold=ETA_FAILURE,
        cusum_threshold=CUSUM_THRESHOLD,
        seed=SEED + 6,
    )
    eta_true = plant.states[-1][2]
    eta_hat = out.filter.filtered[-1].mean[2]
    eta_sd = math.sqrt(out.filter.filtered[-1].covariance[2][2])
    print(
        f"record of {RECORD_STEPS} steps, injected efficiency step {FAULT.size:+.3f} "
        f"at step {FAULT.index}; final eta true={eta_true:.4f} estimated="
        f"{eta_hat:.4f} (filter sd {eta_sd:.4f})"
    )
    first = FAULT.index - 1
    cusum = out.cusum.alarm_index
    glr = out.glr.alarm_index
    magnitude = out.glr.magnitude
    size = "none" if magnitude is None else f"{magnitude:.3f}"
    print(
        f"CUSUM (k={out.cusum.reference}, h={out.cusum.threshold}): alarm at index "
        f"{cusum} side={out.cusum.alarm_side}, delay "
        f"{None if cusum is None else cusum - first} steps; GLR alarm at {glr}, "
        f"located change at {out.glr.change_index}, size "
        f"{size} (in innovation sd)"
    )
    clean = simulate_plant(spec, RECORD_STEPS, seed=SEED)
    quiet = assess_engine_health(
        spec,
        clean.observations,
        failure_threshold=ETA_FAILURE,
        cusum_threshold=CUSUM_THRESHOLD,
        seed=SEED + 6,
    )
    print(
        f"same noise, no fault: CUSUM alarm {quiet.cusum.alarm_index}, GLR alarm "
        f"{quiet.glr.alarm_index}"
    )
    low, high = out.rul.interval(0.9)
    median = out.rul.quantile(0.5)
    print(
        f"RUL to eta={ETA_FAILURE}: median {median:.1f} s, 90 % interval "
        f"[{low:.1f}, {high:.1f}] s (plug-in {out.rul.plug_in_mean_seconds:.1f} s, "
        f"already failed {out.rul.already_failed:.0%}, non-degrading "
        f"{out.rul.non_degrading:.0%})"
    )
    study = detection_operating_characteristic(
        spec,
        thresholds=(CUSUM_THRESHOLD,),
        fault_size=FAULT.size,
        fault_index=FAULT.index,
        runs=budget.health_runs,
        steps=RECORD_STEPS,
        seed=SEED + 7,
    )
    point = study.points[0]
    delay = (
        "none" if point.mean_delay_steps is None else f"{point.mean_delay_steps:.1f}"
    )
    print(
        f"CUSUM over {study.runs} seeds, h={point.threshold}: false-alarm runs "
        f"{point.false_alarm_runs}/{study.runs}, detection probability "
        f"{point.detection_probability:.2f}, early alarms {point.early_alarms}, "
        f"mean delay {delay} steps"
    )
    rows: list[Row] = [
        (
            "engine",
            "CUSUM delay after fault (steps)",
            "none" if cusum is None else f"{cusum - first}",
            f"detection prob {point.detection_probability:.2f} over "
            f"{study.runs} seeds; false-alarm runs {point.false_alarm_runs}",
        ),
        (
            "engine",
            "RUL to failure threshold (s)",
            f"{median:.0f}",
            f"90% interval [{low:.0f}, {high:.0f}]",
        ),
    ]
    return {
        "assessment": out,
        "plant": plant,
        "quiet": quiet,
        "study": study,
        "rul_interval": (low, high),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# (e) entry
# ---------------------------------------------------------------------------


def entry_probabilistic_model() -> Any:
    """Return the synthetic entry model with seven uncertain inputs.

    Returns:
        A probabilistic model of the capsule and its shield: entry speed and
        flight-path angle, ballistic coefficient, nose radius, density scale,
        heating scale and heat of ablation.

    References:
        [EQ-AERO-ENTRY-009], [REF-SUTTON-GRAVES-1971].

    """
    return entry_model(
        CAPSULE,
        ABLATOR,
        entry_speed=Normal(ENTRY_SPEED, 100.0),
        flight_path_angle=Normal(
            math.radians(ENTRY_ANGLE_DEG), math.radians(ENTRY_ANGLE_SD_DEG)
        ),
        ballistic_coefficient=Normal(320.0, 25.0),
        nose_radius=Normal(4.7, 0.1),
        density_scale=Normal(1.0, 0.1),
        heating_scale=Normal(1.0, 0.1),
        heat_of_ablation=Normal(2.5e7, 2.5e6),
        max_step=ENTRY_MAX_STEP,
        resolution=ENTRY_RESOLUTION,
    )


def entry_part(budget: Budget) -> dict[str, Any]:
    """Estimate the heat-shield shortfall probability and the entry corridor.

    Args:
        budget: Sample sizes.

    Returns:
        Reliability results, the corridor, the trade-off sweep and summary rows.

    References:
        [EQ-AERO-ENTRY-009], [EQ-AERO-ENTRY-010], [EQ-REL-FORM-001],
        [EQ-REL-SUBSET-001], [EQ-REL-PF-001], [REF-AU-BECK-2001].

    """
    headline("(e) entry: heat-shield margin and entry corridor")
    angle = math.radians(ENTRY_ANGLE_DEG)
    nominal = simulate_entry(
        CAPSULE,
        altitude=120e3,
        speed=ENTRY_SPEED,
        flight_path_angle=angle,
        max_step=ENTRY_MAX_STEP,
        resolution=ENTRY_RESOLUTION,
    )
    print(
        f"nominal entry at {ENTRY_ANGLE_DEG} deg, {ENTRY_SPEED / 1e3:.0f} km/s: "
        f"status={nominal.status} peak load {nominal.peak_deceleration_g:.2f} g, "
        f"peak heat flux {nominal.peak_heat_flux / 1e6:.2f} MW/m2, heat load "
        f"{nominal.heat_load / 1e6:.1f} MJ/m2"
    )
    state = ModelLimitState(
        entry_probabilistic_model(),
        "required_thickness",
        ABLATOR.design_thickness,
        "above",
    )
    form = form_hlrf(state)
    subset = subset_simulation(
        state, samples_per_level=budget.entry_subset, seed=SEED + 8
    )
    mc = model_monte_carlo(state, budget.entry_mc, seed=SEED + 9)
    print(f"P(required thickness > design {ABLATOR.design_thickness * 1e3:.0f} mm):")
    print(
        f"  FORM   P={form.probability:.3e} beta={form.beta:.3f} "
        f"converged={form.converged} runs={form.evaluations}"
    )
    print(
        f"  subset P={subset.probability:.3e} "
        f"cov={subset.coefficient_of_variation:.3f} "
        f"converged={subset.converged} runs={subset.evaluations}"
    )
    print(
        f"  crude  P={mc.probability:.3e} se={mc.standard_error:.2e} "
        f"runs={mc.evaluations}"
    )
    weights = ", ".join(
        f"{name}={value:.2f}" for name, value in form.importance_factors.items()
    )
    print(f"  FORM importance factors: {weights}")
    corridor = entry_corridor(
        CAPSULE,
        entry_speed=ENTRY_SPEED,
        g_limit=G_LIMIT,
        tolerance=1.0e-4,
        max_step=ENTRY_MAX_STEP,
        resolution=ENTRY_RESOLUTION,
    )
    shallow = math.degrees(corridor.shallow_angle)
    steep = math.degrees(corridor.steep_angle)
    width = math.degrees(corridor.width)
    print(
        f"entry corridor at {ENTRY_SPEED / 1e3:.0f} km/s, {G_LIMIT:g} g limit: "
        f"skip-out {shallow:.3f} deg ({corridor.peak_g_at_shallow:.2f} g, "
        f"{corridor.heat_load_at_shallow / 1e6:.1f} MJ/m2) to {steep:.3f} deg "
        f"({corridor.peak_g_at_steep:.2f} g, {corridor.heat_load_at_steep / 1e6:.1f} "
        f"MJ/m2); width {width:.3f} deg, {corridor.evaluations} integrations"
    )
    law = Normal(ENTRY_ANGLE_DEG, ENTRY_ANGLE_SD_DEG)
    outside = float(law.cdf(steep)) + 1.0 - float(law.cdf(shallow))
    print(
        f"  angle dispersion alone (sd {ENTRY_ANGLE_SD_DEG} deg) leaves the corridor "
        f"with probability {outside:.3f} (closed-form normal law)"
    )
    sweep = []
    first = shallow
    last = steep - 0.12
    for k in range(budget.sweep_points):
        gamma_deg = first + (last - first) * k / (budget.sweep_points - 1)
        run = simulate_entry(
            CAPSULE,
            altitude=120e3,
            speed=ENTRY_SPEED,
            flight_path_angle=math.radians(gamma_deg),
            max_step=ENTRY_MAX_STEP,
            resolution=ENTRY_RESOLUTION,
        )
        sweep.append((gamma_deg, run.peak_deceleration_g, run.heat_load / 1e6))
    print("trade-off sweep (angle deg, peak load g, heat load MJ/m2):")
    for gamma_deg, peak, heat in sweep:
        print(f"  {gamma_deg:8.3f} {peak:8.2f} {heat:8.1f}")
    rows: list[Row] = [
        (
            "entry",
            "P(heat shield too thin)",
            f"{mc.probability:.3f}",
            f"MC se {mc.standard_error:.3f}; FORM {form.probability:.3f}; "
            f"subset {subset.probability:.3f} "
            f"(cov {subset.coefficient_of_variation:.2f})",
        ),
        (
            "entry",
            "corridor width (deg)",
            f"{width:.2f}",
            f"{shallow:.2f} to {steep:.2f} deg; bisection tol 1e-4 rad",
        ),
    ]
    return {
        "form": form,
        "subset": subset,
        "mc": mc,
        "corridor": corridor,
        "sweep": sweep,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def summary_table(rows: list[Row]) -> str:
    """Return the mission risk summary table as text.

    Args:
        rows: ``(stage, quantity, estimate, uncertainty statement)`` rows.

    Returns:
        The aligned table.

    References:
        [EQ-REL-PF-001]. No equation is evaluated.

    """
    header: Row = ("stage", "quantity", "estimate", "uncertainty statement")
    table = [header, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(4)]
    lines = []
    for index, row in enumerate(table):
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        )
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def has_matplotlib() -> bool:
    """Return whether Matplotlib is importable.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def save(figure: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    Args:
        figure: Matplotlib figure.
        name: File name.

    Returns:
        The file name written.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def entry_figure(entry: dict[str, Any]) -> str:
    """Draw the heat-load versus peak-load trade-off across the corridor.

    Args:
        entry: Result of :func:`entry_part`.

    Returns:
        The file name written.

    References:
        [EQ-AERO-ENTRY-010], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    corridor = entry["corridor"]
    sweep = entry["sweep"]
    axes = theme.new_axes(figsize=(7.2, 4.4))
    colour = theme.role_colour("model")
    theme.line(
        axes,
        [peak for _, peak, _ in sweep],
        [heat for _, _, heat in sweep],
        color=colour,
    )
    theme.points(
        axes,
        [corridor.peak_g_at_shallow, corridor.peak_g_at_steep],
        [corridor.heat_load_at_shallow / 1e6, corridor.heat_load_at_steep / 1e6],
        color=colour,
        size=52.0,
    )
    theme.reference_line(
        axes,
        x=corridor.g_limit,
        entity="threshold",
        label=f"{corridor.g_limit:g} g limit",
    )
    theme.note(
        axes,
        f"skip-out bound, {math.degrees(corridor.shallow_angle):.2f} deg",
        xy=(corridor.peak_g_at_shallow, corridor.heat_load_at_shallow / 1e6),
        offset=(30.0, 2.0),
    )
    theme.note(
        axes,
        f"load bound, {math.degrees(corridor.steep_angle):.2f} deg",
        xy=(corridor.peak_g_at_steep, corridor.heat_load_at_steep / 1e6),
        offset=(-110.0, -26.0),
    )
    theme.axis_labels(axes, "Peak deceleration (g)", "Convective heat load (MJ/m²)")
    theme.title_block(
        axes,
        f"The corridor is {math.degrees(corridor.width):.2f} deg wide: heat load falls "
        f"from {corridor.heat_load_at_shallow / 1e6:.0f} to "
        f"{corridor.heat_load_at_steep / 1e6:.0f} MJ/m² as peak load rises "
        f"from {corridor.peak_g_at_shallow:.1f} to {corridor.peak_g_at_steep:.1f} g",
        f"Synthetic capsule at {ENTRY_SPEED / 1e3:.0f} km/s, {len(sweep)} flight-path "
        f"angles from the skip-out bound to beyond the load bound",
    )
    return save(axes.figure, "spacecraft-mission-entry-tradeoff.png")


def mmod_figure(mmod: dict[str, Any]) -> str:
    """Draw the distribution of PNP under epistemic uncertainty.

    Args:
        mmod: Result of :func:`mmod_part`.

    Returns:
        The file name written.

    References:
        [EQ-AERO-MMOD-006], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization import theme

    distribution = mmod["distribution"]
    values = list(distribution.pnp_sorted)
    count = len(values)
    axes = theme.new_axes(figsize=(7.2, 4.4))
    theme.line(
        axes,
        values,
        [(i + 1) / count for i in range(count)],
        color=theme.role_colour("model"),
    )
    theme.reference_line(axes, x=PNP_REQUIREMENT, entity="threshold")
    theme.note(
        axes,
        f"requirement {PNP_REQUIREMENT:g}",
        xy=(PNP_REQUIREMENT, 0.6),
        offset=(-125.0, 0.0),
    )
    theme.points(
        axes,
        [PNP_REQUIREMENT],
        [distribution.probability_below_requirement],
        color=theme.role_colour("series-3"),
        size=52.0,
    )
    theme.axis_labels(axes, "Probability of no penetration", "Cumulative probability")
    theme.title_block(
        axes,
        f"PNP misses {PNP_REQUIREMENT:g} with probability "
        f"{distribution.probability_below_requirement:.2f} although its nominal value "
        f"is {distribution.pnp_at_median_inputs:.3f}",
        f"Synthetic MMOD coefficients, {DESIGN_WALL * 1e3:.0f} mm wall, "
        f"{EXPOSURE_YEARS:g} yr; {count} epistemic draws, "
        f"mean {distribution.mean_pnp:.3f}",
    )
    return save(axes.figure, "spacecraft-mission-mmod-pnp.png")


def main() -> None:
    """Run the five stages, print every number and the summary, write the figures.

    References:
        [EQ-AERO-ASC-008], [EQ-AERO-BUCK-006], [EQ-AERO-MMOD-006],
        [EQ-AERO-PHM-003], [EQ-AERO-ENTRY-010].

    """
    started = time.perf_counter()
    print("Spacecraft mission-risk laboratory (all cases synthetic)")
    ascent = ascent_part(FULL)
    tank = tank_part(FULL)
    mmod = mmod_part(FULL)
    health = health_part(FULL)
    entry = entry_part(FULL)
    headline("mission risk summary (synthetic; stages are not combined)")
    print(
        summary_table(
            ascent["rows"]
            + tank["rows"]
            + mmod["rows"]
            + health["rows"]
            + entry["rows"]
        )
    )
    print(f"\ntotal wall time {time.perf_counter() - started:.1f} s")
    if not has_matplotlib():
        print("Matplotlib is not installed; figures skipped")
        return
    written = (entry_figure(entry), mmod_figure(mmod))
    print("figures: " + ", ".join(written))


if __name__ == "__main__":
    main()
