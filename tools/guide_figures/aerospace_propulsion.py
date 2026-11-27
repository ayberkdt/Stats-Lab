"""Figures of the aerospace propulsion UQ guide.

Three seeded computations on the synthetic cases of
``examples/rocket_propulsion_uq_lab.py``, which this module loads so that the
guide, the lab and the figures share one motor, one chamber and one set of
input laws: the pressure-time band of a dispersed solid motor against the
burst pressure, the probability of thermoacoustic instability against the
flame time lag, and the Gaussian-process Sobol indices of the growth rate.
The facts are the numbers the captions in the guide quote. Every case is
synthetic; none is data from a real motor or engine.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-SUTTON-BIBLARZ-2017],
    [REF-MAGRI-JUNIPER-2013].

"""

from __future__ import annotations

import functools
import importlib.util
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from probstat.aerospace import simulate_solid_motor, thermoacoustic_model
from probstat.reliability import ModelLimitState, subset_simulation
from probstat.visualization import theme

from . import GuideFigure

SEED = 20261127
WIDE = (7.6, 4.4)
LAB = Path(__file__).resolve().parents[2] / "examples" / "rocket_propulsion_uq_lab.py"
LAGS = tuple(0.0005 * k for k in range(1, 17))  # flame time lag grid (s)
LAG_RUNS = 400


@functools.cache
def _lab() -> ModuleType:
    """Load the laboratory script as a module (it is not a package).

    References:
        [EQ-AERO-SRM-007]. No equation is evaluated.

    """
    spec = importlib.util.spec_from_file_location("_rocket_lab_for_figures", LAB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def _motor_band() -> dict[str, Any]:
    """Disperse the motor: pressure traces, peak pressures and burst probability.

    References:
        [EQ-AERO-SRM-007], [EQ-REL-SUBSET-001].

    """
    lab = _lab()
    from probstat.aerospace import solid_motor_model

    model = solid_motor_model(lab.GRAIN, lab.MOTOR, lab.BURST_MEDIAN, steps=40)
    draws = model.inputs.sample(150, seed=SEED)
    traces = []
    for point in draws:
        values = dict(zip(model.inputs.names, point, strict=True))
        run = simulate_solid_motor(lab.GRAIN, lab.motor_from_inputs(values), steps=60)
        traces.append((run.times, run.pressures, run.burn_time, run.meop))
    limit = ModelLimitState(model, "burst_margin", 0.0, failure="below")
    subset = subset_simulation(limit, samples_per_level=500, seed=SEED)
    return {"traces": traces, "subset": subset, "burst": lab.BURST_MEDIAN}


def _interpolate(trace: tuple[Any, ...], moment: float) -> float:
    """Return the linearly interpolated pressure of one trace.

    References:
        [EQ-AERO-SRM-007]. No equation is evaluated.

    """
    times, pressures = trace[0], trace[1]
    for k in range(1, len(times)):
        if times[k] >= moment:
            weight = (moment - times[k - 1]) / (times[k] - times[k - 1])
            return float(pressures[k - 1] + weight * (pressures[k] - pressures[k - 1]))
    return float(pressures[-1])


def _motor_figure() -> tuple[Any, dict[str, str]]:
    """Draw the pressure band of the dispersed motor against the burst median.

    References:
        [EQ-AERO-SRM-007], [EQ-REL-SUBSET-001], [REF-SUTTON-BIBLARZ-2017].

    """
    data = _motor_band()
    traces = data["traces"]
    peaks = sorted(trace[3] / 1e6 for trace in traces)
    horizon = min(trace[2] for trace in traces)
    grid = [horizon * i / 60 for i in range(61)]
    low, mid, high = [], [], []
    for moment in grid:
        column = sorted(_interpolate(trace, moment) / 1e6 for trace in traces)
        low.append(column[int(0.05 * len(column))])
        mid.append(column[len(column) // 2])
        high.append(column[int(0.95 * len(column))])
    facts = {
        "mean_peak": f"{statistics.fmean(peaks):.2f}",
        "q95_peak": f"{peaks[int(0.95 * len(peaks))]:.2f}",
        "burst": f"{data['burst'] / 1e6:.2f}",
        "gap": f"{data['burst'] / 1e6 - peaks[int(0.95 * len(peaks))]:.1f}",
        "probability": f"{data['subset'].probability:.1e}",
        "runs": f"{data['subset'].evaluations}",
    }
    palette = theme.active_palette()
    axes = theme.new_axes(figsize=WIDE)
    colour = theme.role_colour("model")
    theme.band(axes, grid, low, high, color=colour)
    theme.line(axes, grid, mid, color=colour)
    theme.reference_line(axes, y=data["burst"] / 1e6, entity="threshold")
    axes.set_xlim(0.0, horizon)
    axes.set_ylim(4.0, 9.0)
    axes.annotate(
        f"median burst pressure {facts['burst']} MPa",
        xy=(horizon, data["burst"] / 1e6),
        xytext=(-4.0, -12.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    theme.axis_labels(axes, "Time since ignition (s)", "Chamber pressure (MPa)")
    theme.title_block(
        axes,
        f"The 95th-percentile peak sits {facts['gap']} MPa below the median "
        "burst pressure,\nyet subset simulation finds a burst probability of "
        f"{data['subset'].probability:.4f}",
        "Synthetic BATES motor, 5-95 % band and median of 150 draws; "
        "burst probability by subset simulation",
    )
    return axes.figure, facts


@functools.cache
def _lag_curve() -> dict[str, Any]:
    """Estimate P(unstable) on a grid of flame time lags with 400 runs each.

    References:
        [EQ-AERO-CI-006], [EQ-AERO-CI-007].

    """
    lab = _lab()
    probabilities, lows, highs = [], [], []
    for lag in LAGS:
        model = thermoacoustic_model(replace(lab.CHAMBER, tau=lag))
        points = model.inputs.sample(LAG_RUNS, seed=SEED)
        growth = model.evaluate_many(points)["growth_rate"]
        p = sum(1 for g in growth if g > 0.0) / LAG_RUNS
        centre = (p + 1.96**2 / (2 * LAG_RUNS)) / (1 + 1.96**2 / LAG_RUNS)
        half = (
            1.96
            * math.sqrt(p * (1 - p) / LAG_RUNS + 1.96**2 / (4 * LAG_RUNS**2))
            / (1 + 1.96**2 / LAG_RUNS)
        )
        probabilities.append(p)
        lows.append(max(0.0, centre - half))
        highs.append(min(1.0, centre + half))
    return {"p": probabilities, "low": lows, "high": highs}


def _lag_figure() -> tuple[Any, dict[str, str]]:
    """Draw the instability probability against the flame time lag.

    References:
        [EQ-AERO-CI-007], [REF-MAGRI-JUNIPER-2013].

    """
    data = _lag_curve()
    lags_ms = [1e3 * lag for lag in LAGS]
    p = data["p"]
    top = max(range(len(p)), key=p.__getitem__)
    run = [k for k in range(top, len(p)) if p[k] < 0.005]
    start = run[0]
    end = start
    while end + 1 < len(p) and p[end + 1] < 0.005:
        end += 1
    facts = {
        "peak": f"{p[top]:.2f}",
        "peak_lag": f"{lags_ms[top]:.1f}",
        "calm_from": f"{lags_ms[start]:.1f}",
        "calm_to": f"{lags_ms[end]:.1f}",
        "last": f"{p[-1]:.2f}",
        "last_lag": f"{lags_ms[-1]:.1f}",
        "runs": f"{LAG_RUNS}",
    }
    palette = theme.active_palette()
    axes = theme.new_axes(figsize=WIDE)
    colour = theme.role_colour("model")
    theme.band(axes, lags_ms, data["low"], data["high"], color=colour)
    theme.line(axes, lags_ms, p, color=colour)
    theme.points(axes, lags_ms, p, color=colour, size=18.0)
    axes.set_xlim(lags_ms[0], lags_ms[-1])
    axes.set_ylim(0.0, 1.0)
    axes.annotate(
        f"peak {facts['peak']} at {facts['peak_lag']} ms",
        xy=(lags_ms[top], p[top]),
        xytext=(10.0, 8.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    middle = 0.5 * (lags_ms[start] + lags_ms[end])
    axes.annotate(
        f"stable from {facts['calm_from']} to {facts['calm_to']} ms",
        xy=(middle, 0.0),
        xytext=(0.0, 16.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    theme.axis_labels(axes, "Mean flame time lag (ms)", "Probability of instability")
    theme.title_block(
        axes,
        "The probability of instability repeats with the flame time lag\n"
        "because the phase omega tau decides whether heat release feeds the mode",
        f"Synthetic closed-choked chamber, {LAG_RUNS} runs per lag, "
        "95 % Wilson band, all five parameters uncertain",
    )
    return axes.figure, facts


@functools.cache
def _gp_indices() -> dict[str, Any]:
    """Fit an 80-run GP in z coordinates and return its Sobol posterior.

    References:
        [EQ-SUR-GPB-004], [EQ-SUR-GPB-006].

    """
    from probstat.sampling import latin_hypercube
    from probstat.surrogates.gaussian_process import fit_gaussian_process
    from probstat.surrogates.gp_bayes import gp_sobol_posterior, validate_gp_emulator

    lab = _lab()
    model = thermoacoustic_model(lab.CHAMBER)
    mean, sd = model.inputs.means(), model.inputs.standard_deviations()

    def to_z(x: tuple[float, ...]) -> tuple[float, ...]:
        """Return standard-normal coordinates of a physical input.

        References:
            [EQ-AERO-CI-006].

        """
        return tuple((v - m) / s for v, m, s in zip(x, mean, sd, strict=True))

    design = model.inputs.sample_design(latin_hypercube(80, 5, seed=SEED))
    held = model.inputs.sample_design(latin_hypercube(30, 5, seed=SEED + 1))
    train = model.evaluate_many(design)["growth_rate"]
    truth = model.evaluate_many(held)["growth_rate"]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = fit_gaussian_process(
            [to_z(x) for x in design], train, seed=SEED, restarts=1
        )
    check = validate_gp_emulator(gp, [to_z(x) for x in held], truth)
    sobol = gp_sobol_posterior(
        gp,
        (-3.0,) * 5,
        (3.0,) * 5,
        base_samples=64,
        paths=60,
        seed=SEED,
        max_points=600,
    )
    return {"names": model.inputs.names, "sobol": sobol, "check": check}


def _sobol_figure() -> tuple[Any, dict[str, str]]:
    """Draw GP first-order and total Sobol indices of the growth rate.

    References:
        [EQ-SUR-GPB-006], [REF-OAKLEY-OHAGAN-2004].

    """
    data = _gp_indices()
    sobol, names = data["sobol"], data["names"]
    order = sorted(range(len(names)), key=lambda i: sobol.first_order_mean[i])
    first = {names[i]: sobol.first_order_mean[i] for i in range(len(names))}
    total = {names[i]: sobol.total_mean[i] for i in range(len(names))}
    ranked = sorted(first, key=first.__getitem__, reverse=True)
    assert set(ranked[:2]) == {"tau", "n"}, ranked
    facts = {
        "first": ranked[0],
        "first_index": f"{first[ranked[0]]:.2f}",
        "second": ranked[1],
        "second_index": f"{first[ranked[1]]:.2f}",
        "third_total": f"{total[ranked[2]]:.2f}",
        "coverage": f"{data['check'].coverage:.2f}",
    }
    palette = theme.active_palette()
    axes = theme.new_axes(figsize=WIDE)
    rows = list(range(len(order)))
    theme.points(
        axes,
        [sobol.first_order_mean[i] for i in order],
        [r - 0.14 for r in rows],
        color=theme.role_colour("model"),
        size=60.0,
    )
    theme.points(
        axes,
        [sobol.total_mean[i] for i in order],
        [r + 0.14 for r in rows],
        color=theme.role_colour("series-3"),
        size=60.0,
        marker="D",
    )
    axes.set_yticks(rows)
    axes.set_yticklabels([names[i] for i in order])
    axes.set_xlim(-0.05, 0.65)
    axes.set_ylim(-0.6, len(order) - 0.2)
    axes.annotate(
        "total effect (diamond)",
        xy=(sobol.total_mean[order[-1]], rows[-1] + 0.14),
        xytext=(12.0, 2.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    axes.annotate(
        "first order (circle)",
        xy=(sobol.first_order_mean[order[-1]], rows[-1] - 0.14),
        xytext=(12.0, -2.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=theme.ANNOTATION_SIZE,
        color=palette.ink,
    )
    theme.axis_labels(axes, "Sobol index of the growth rate", "Uncertain parameter")
    theme.title_block(
        axes,
        "The flame time lag and the interaction index carry most of the "
        "variance of the growth rate",
        "Synthetic chamber; GP with 80 runs, posterior mean over 60 sample paths, "
        "inputs uniform on |z| <= 3",
    )
    return axes.figure, facts


FIGURES = [
    GuideFigure(
        name="propulsion-motor-pressure-band",
        guide="aerospace-propulsion-uq",
        alt=(
            "Chamber pressure in megapascals against time since ignition: a shaded "
            "5 to 95 percent band and median of 150 motor burns, with a dashed line "
            "at the median burst pressure"
        ),
        draw=_motor_figure,
    ),
    GuideFigure(
        name="propulsion-instability-flame-lag",
        guide="aerospace-propulsion-uq",
        alt=(
            "Probability of thermoacoustic instability against the mean flame time "
            "lag in milliseconds: a line with markers and a shaded 95 percent "
            "band that rises and falls repeatedly across the lag range"
        ),
        draw=_lag_figure,
    ),
    GuideFigure(
        name="propulsion-growth-rate-sobol",
        guide="aerospace-propulsion-uq",
        alt=(
            "Dot chart of first-order and total-effect Sobol indices of the growth "
            "rate for five uncertain chamber parameters, ordered by first-order "
            "index"
        ),
        draw=_sobol_figure,
    ),
]
