"""Figures of the spacecraft systems guide.

One seeded computation on four synthetic failure events of the guide's
sections 2, 3 and 5, each judged twice: by the first-order reliability method
(FORM) and by plain Monte Carlo with its 95 % Wilson interval. The cases are a
power margin that is linear in one random input (FORM is exact), the series
minimum of two power margins (FORM follows one branch), a 25-year orbital
lifetime with a random solar phase (the lifetime is periodic in the phase) and
a reentry casualty expectation that is a step function of the heating
(the gradient vanishes, so FORM raises; FORM on the continuous demise margin
of the one switching fragment is drawn instead). The facts are the numbers the
caption in the guide quotes. Every vehicle is synthetic: none of the events
describes a real spacecraft.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-HASOFER-LIND-1974],
    [REF-MELCHERS-BECK-2018].

"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

from probstat.aerospace import orbital_decay as od
from probstat.aerospace import power_budget as pb
from probstat.aerospace.reentry_casualty import (
    demonstration_fragments,
    reentry_casualty_model,
)
from probstat.distributions import LogNormal, Normal, Uniform
from probstat.reliability import ModelLimitState, form_hlrf, model_monte_carlo
from probstat.visualization import theme

from . import GuideFigure

WIDE = (7.6, 4.6)
Z95 = 1.959963984540054
POWER_SAMPLES = 3000
ORBIT_SAMPLES = 1000
CASUALTY_SAMPLES = 400
LIMIT = 1.0e-4  # synthetic casualty-expectation limit
ROWS = (
    "Casualty expectation\n(step in the heating)",
    "Orbital lifetime\n(random solar phase)",
    "Power, series minimum\n(two margins)",
    "Power, one random input\n(linear margin)",
)


def _wilson(hits: int, n: int) -> tuple[float, float]:
    """Return the 95 % Wilson score interval of a binomial proportion.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    p = hits / n
    centre = (p + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / (1 + Z95**2 / n)
    return centre - half, centre + half


def _power_model(design: Any, bol: float) -> Any:
    """Return the series-minimum power model of the guide's section 2.

    References:
        [EQ-AERO-POWER-007].

    """
    return pb.power_budget_model(
        design,
        10.0,
        bol_power=LogNormal(math.log(bol), 0.03),
        degradation_rate=Uniform(0.015, 0.035),
        fade_rate=Uniform(2.0e-6, 8.0e-6),
        charge_efficiency=Uniform(0.90, 0.98),
        discharge_efficiency=Uniform(0.90, 0.98),
        load_factor=LogNormal(0.0, 0.04),
        failed_strings=Uniform(0.0, 2.0),
    )


@functools.cache
def _cases() -> dict[str, tuple[float, int, int]]:
    """Run FORM and Monte Carlo on the four events.

    Returns a mapping from case name to ``(form, hits, samples)``; ``form`` is
    ``nan`` where FORM raises.

    References:
        [EQ-AERO-POWER-008], [EQ-AERO-DECAY-007], [EQ-AERO-CASUALTY-007].

    """
    out: dict[str, tuple[float, int, int]] = {}
    design = pb.demonstration_design()
    smooth = pb.power_budget_model(
        design,
        10.0,
        bol_power=Normal(450.0, 25.0),
        degradation_rate=0.025,
        fade_rate=5.0e-6,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        load_factor=1.0,
        failed_strings=0.0,
    )
    series = _power_model(dataclasses.replace(design, dod_max=0.30), 500.0)
    for name, model, seed in (("smooth", smooth, 3), ("series", series, 4)):
        form = pb.power_failure_probability(model)
        mc = pb.power_failure_probability(
            model, method="monte_carlo", samples=POWER_SAMPLES, seed=seed
        )
        out[name] = (
            form.probability,
            round(mc.probability * POWER_SAMPLES),
            POWER_SAMPLES,
        )
    thermosphere = od.demonstration_thermosphere()
    orbit = od.orbital_lifetime_model(
        thermosphere,
        25.0,
        initial_altitude=Normal(640.0e3, 3000.0),
        drag_coefficient=LogNormal(math.log(2.2), 0.1),
        area_to_mass=LogNormal(math.log(0.01), 0.2),
        attitude_factor=Uniform(0.7, 1.0),
        solar_phase=Uniform(0.0, 1.0),
        solar_amplitude=Uniform(0.2, 0.5),
        density_factor=LogNormal(0.0, 0.3),
        altitude_fraction=0.1,
    )
    form = od.noncompliance_probability(orbit, 25.0, method="form")
    mc = od.noncompliance_probability(
        orbit, 25.0, method="monte_carlo", samples=ORBIT_SAMPLES, seed=4
    )
    out["orbit"] = (
        form.probability,
        round(mc.probability * ORBIT_SAMPLES),
        ORBIT_SAMPLES,
    )
    fragments = tuple(
        f for f in demonstration_fragments() if f.name in ("tank", "panel", "shaft")
    )
    casualty = reentry_casualty_model(
        fragments,
        breakup_altitude=Normal(78.0e3, 4.0e3),
        flight_path_angle=-0.01,
        person_area=0.36,
        population_density=40.0,
        energy_threshold=15.0,
        casualty_threshold=LIMIT,
        heating_scale=LogNormal(0.0, 0.3),
        melt_temperature_scale={"aluminium_like": Normal(1.0, 0.03)},
        ballistic_scale={"panel": LogNormal(0.0, 0.15)},
    )
    event = ModelLimitState(casualty, "casualty_expectation", LIMIT, "above")
    margin = ModelLimitState(casualty, "demise_margin_panel", 0.0, "above")
    try:
        form_hlrf(event)
    except ValueError:
        pass
    else:  # pragma: no cover - the figure states that FORM raises here
        raise AssertionError("FORM on the step-valued expectation must raise")
    margin_form = form_hlrf(margin, relative_step=1.0e-4, tolerance=1.0e-3)
    draws = model_monte_carlo(event, CASUALTY_SAMPLES, seed=3)
    out["casualty"] = (
        margin_form.probability,
        round(draws.probability * CASUALTY_SAMPLES),
        CASUALTY_SAMPLES,
    )
    return out


def _form_figure() -> tuple[Any, dict[str, str]]:
    """Draw FORM against Monte Carlo with its interval on the four events.

    References:
        [EQ-AERO-POWER-008], [EQ-AERO-DECAY-007], [EQ-AERO-CASUALTY-007],
        [REF-HASOFER-LIND-1974].

    """
    cases = _cases()
    names = ("casualty", "orbit", "series", "smooth")
    mc = {n: cases[n][1] / cases[n][2] for n in names}
    form = {n: cases[n][0] for n in names}
    intervals = {n: _wilson(cases[n][1], cases[n][2]) for n in names}
    facts = {
        "smooth_form": f"{form['smooth']:.3f}",
        "smooth_mc": f"{mc['smooth']:.3f}",
        "series_form": f"{form['series']:.3f}",
        "series_mc": f"{mc['series']:.3f}",
        "orbit_form": f"{form['orbit']:.3f}",
        "orbit_mc": f"{mc['orbit']:.3f}",
        "casualty_mc": f"{mc['casualty']:.3f}",
        "casualty_margin_form": f"{form['casualty']:.3f}",
        "power_samples": f"{POWER_SAMPLES}",
        "orbit_samples": f"{ORBIT_SAMPLES}",
        "casualty_samples": f"{CASUALTY_SAMPLES}",
    }
    palette = theme.active_palette()
    observed = theme.role_colour("observed")
    model = theme.role_colour("model")
    axes = theme.new_axes(figsize=WIDE)
    rows = {name: float(index) for index, name in enumerate(reversed(names))}
    for name in names:
        low, high = intervals[name]
        axes.hlines(
            rows[name] + 0.14, low, high, color=observed, linewidth=3.0, zorder=3
        )
    theme.points(
        axes,
        [mc[n] for n in names],
        [rows[n] + 0.14 for n in names],
        color=observed,
        size=70.0,
    )
    shown = [n for n in names if n != "casualty"]
    theme.points(
        axes,
        [form[n] for n in shown],
        [rows[n] - 0.14 for n in shown],
        color=model,
        size=70.0,
        marker="D",
    )
    theme.points(
        axes,
        [form["casualty"]],
        [rows["casualty"] - 0.14],
        color=model,
        size=70.0,
        marker="D",
    )
    axes.set_yticks([rows[n] for n in names])
    axes.set_yticklabels(ROWS)
    axes.set_xlim(0.0, 0.5)
    axes.set_ylim(-0.6, 3.6)
    note_size = theme.ANNOTATION_SIZE
    axes.annotate(
        "FORM on the expectation raises (zero gradient);\n"
        "diamond: FORM on the demise margin of the panel",
        xy=(0.0, rows["casualty"]),
        xytext=(0.17, rows["casualty"] - 0.38),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=note_size,
        color=palette.ink,
    )
    axes.annotate(
        "Monte Carlo, 95 % Wilson interval (circle)",
        xy=(mc["smooth"], rows["smooth"] + 0.14),
        xytext=(0.0, 14.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=note_size,
        color=observed,
    )
    axes.annotate(
        "FORM (diamond)",
        xy=(form["series"], rows["series"] - 0.14),
        xytext=(-8.0, -16.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=note_size,
        color=model,
    )
    theme.axis_labels(axes, "Probability of the failure event", "Synthetic case")
    theme.title_block(
        axes,
        f"FORM misses on the series minimum ({facts['series_form']} vs "
        f"{facts['series_mc']}) and with a random solar phase",
        "Synthetic spacecraft cases; Monte Carlo with "
        f"{CASUALTY_SAMPLES}, {ORBIT_SAMPLES}, {POWER_SAMPLES} and "
        f"{POWER_SAMPLES} samples from the top row down",
    )
    return axes.figure, facts


FIGURES = [
    GuideFigure(
        name="spacecraft-form-versus-simulation",
        guide="spacecraft-systems-uq",
        alt=(
            "Dot chart of four synthetic failure probabilities: Monte Carlo "
            "estimates with 95 percent intervals as circles and FORM estimates as "
            "diamonds, for two power cases, a lifetime and a casualty case"
        ),
        draw=_form_figure,
    ),
]
