"""Lorenz assimilation laboratory: ensemble filters on Lorenz-96, 4D-Var on Lorenz-63.

Run with ``PYTHONPATH=src python examples/lorenz_assimilation_lab.py``. Part 1
is a Lorenz-96 twin experiment (40 variables, forcing 8, every component
observed) in which four ensemble filters with 20 members are compared by
their analysis error and spread, followed by one ensemble Kalman smoother
run. Part 2 is strong-constraint 4D-Var on Lorenz-63: a short window against
its background, the outer-loop costs of the incremental form, and the local
minima of a long window against a cycle of short windows. With Matplotlib two
figures are written to ``examples/generated/``. Every random step is seeded;
the run takes about a minute in pure Python.

Definitions:
    Lorenz-96 truth: ``J = 40``, ``F = 8``, one RK4 step of 0.05 per cycle,
        started from a state spun up for 400 steps from ``x_i = 8`` (with
        ``x_0 = 8.01``); 80 cycles simulated with ``R = I`` and prior
        ``N(spun-up state, I)``.
    Filters: stochastic EnKF without localization; stochastic EnKF with
        Gaspari-Cohn localization ``c = 2`` (periodic) and RTPS 0.5; global
        ETKF with RTPS 0.5; LETKF with ``c = 2`` and RTPS 0.5; all with
        ``N = 20`` members and filter seed ``SEED``.
    RMSE at cycle ``k``: ``sqrt(mean_i (xbar_i - x_i)^2)`` of the analysis
        mean; spread: ``sqrt(mean_i P_ii)`` of the analysis ensemble
        covariance; both averaged over cycles 21-80.
    Lorenz-63 twin: ``sigma = 10``, ``rho = 28``, ``beta = 8/3``, five RK4
        steps of 0.01 per observation (interval 0.05); truth spun up for 1000
        steps from ``(1, 1, 1)``; background ``x_b`` = truth + ``N(0, 4 I)``,
        ``B = 4 I``; every component observed with ``R = 0.25 I``; draws from
        ``random.Random(seed)`` as in the guide.
    Cost ``J``: the strong-constraint 4D-Var cost; ``J(truth)`` is the cost
        at the true initial state, so a converged analysis with
        ``J > 2 J(truth)`` is a spurious local minimum.

Assumptions:
    - Perfect models (``Q = 0``) and Gaussian observation errors of known
      covariance, so the twin truth is a valid reference.
    - Localization assumes correlations decay with grid distance on the
      Lorenz-96 circle.
    - 4D-Var starts BFGS from ``x_b``; the cycled windows reuse ``B = 4 I``
      instead of propagating an error covariance, so they are a simple
      cycling scheme, not an optimal one.

Equations:
    EQ-FILT-LORENZ-001: Lorenz-63 system.
    EQ-FILT-LORENZ-002: Lorenz-96 system.
    EQ-NUM-ODE-003: fixed-step RK4 trajectory over each interval.
    EQ-DA-ENKF-001: stochastic (perturbed-observation) EnKF analysis.
    EQ-DA-ENKF-002: ensemble moments behind the spread.
    EQ-DA-ETKF-001: symmetric-square-root ETKF analysis.
    EQ-DA-LOC-001: Gaspari-Cohn localization weights.
    EQ-DA-LOC-002: Schur-product localization of the stochastic gain.
    EQ-DA-LETKF-001: LETKF with R-localization.
    EQ-DA-RELAX-002: relaxation to prior spread (RTPS).
    EQ-DA-ENKS-001: fixed-lag ensemble Kalman smoother update.
    EQ-DA-4DVAR-001: strong-constraint 4D-Var cost.
    EQ-DA-4DVAR-002: adjoint (reverse-mode) gradient of the cost.
    EQ-DA-4DVAR-003: incremental 4D-Var inner problem.
    EQ-OPT-BFGS-001: BFGS update of the 4D-Var minimization.

Diagnostics:
    - Each filter prints its time-averaged RMSE, spread, and their ratio; a
      ratio far below one is an over-confident ensemble, and an RMSE above
      the observation error (1) is filter divergence.
    - The smoother prints its RMSE against the filter it is built on.
    - 4D-Var prints the error of ``x_0`` and of the end of the window
      against the background error, BFGS iterations, and convergence.
    - The incremental form prints its outer-loop costs and its distance from
      the BFGS analysis.
    - The long-window section prints ``J`` against ``J(truth)`` for every
      seed, so a local minimum is identified by the cost, not by eye.

Cost and error:
    - Lorenz-96: 4 filters x 80 cycles of 20 members plus one smoother run;
      one truth and one filter seed, so the RMSE differences between the two
      localized filters (a few hundredths) are within seed-to-seed scatter.
    - Lorenz-63: one reverse-mode gradient per BFGS evaluation; a 40-interval
      window costs about four times a 10-interval one per evaluation and
      needs more iterations.
    - The local-minimum count is over five seeds only; the guide figure
      measures the rate over more seeds and window lengths.

References:
    REF-LORENZ-1963: the Lorenz-63 system.
    REF-LORENZ-EMANUEL-1998: Lorenz-96 setting and observation design.
    REF-EVENSEN-2003: the ensemble Kalman filter.
    REF-BURGERS-VANLEEUWEN-EVENSEN-1998: perturbed observations.
    REF-HUNT-KOSTELICH-SZUNYOGH-2007: ETKF in symmetric form and the LETKF.
    REF-HOUTEKAMER-MITCHELL-2001: localization of the stochastic EnKF.
    REF-GASPARI-COHN-1999: compactly supported correlation function.
    REF-WHITAKER-HAMILL-2012: relaxation to prior spread.
    REF-EVENSEN-VANLEEUWEN-2000: ensemble smoother.
    REF-LE-DIMET-TALAGRAND-1986: 4D-Var and its adjoint.
    REF-COURTIER-THEPAUT-HOLLINGSWORTH-1994: incremental 4D-Var.
    REF-CARRASSI-ETAL-2018: data assimilation in the geosciences.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from probstat.data_assimilation import (
    Localization,
    Relaxation,
    ensemble_kalman_filter,
    ensemble_kalman_smoother,
    four_dimensional_variational,
    four_dimensional_variational_cost,
    incremental_four_dimensional_variational,
)
from probstat.filtering import (
    lorenz63,
    lorenz63_rhs,
    lorenz96,
    lorenz96_rhs,
    simulate,
)
from probstat.numerics import integrate_fixed_step

# Root seed: Lorenz-96 filters use SEED; the Lorenz-96 truth uses seed 1 as
# in the guide; Lorenz-63 twin ``s`` draws from random.Random(s), as in the
# guide and tests/test_four_dimensional_variational.py.
SEED = 2026
OUTPUT = Path(__file__).parent / "generated"
CYCLES, BURN = 80, 20
MEMBERS = 20
LAG = 4
SHORT, LONG = 10, 40
TWIN_SEEDS = tuple(range(5))
L63 = lorenz63(substeps=5)


def has_matplotlib() -> bool:
    """Return whether the optional plotting dependency is installed.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return importlib.util.find_spec("matplotlib") is not None


def diagonal(size: int, value: float) -> tuple[tuple[float, ...], ...]:
    """Return ``value * I``.

    References:
        [REF-CARRASSI-ETAL-2018]. No equation is evaluated.

    """
    return tuple(tuple(value if i == j else 0.0 for j in range(size)) for i in range(size))


def rmse(estimate: Sequence[float], truth: Sequence[float]) -> float:
    """Return the root-mean-square difference of two states.

    References:
        [REF-EVENSEN-2003]. No equation is evaluated.

    """
    return math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(estimate, truth, strict=True)) / len(truth)
    )


# --------------------------------------------------------------------------
# Part 1: Lorenz-96 ensemble filters
# --------------------------------------------------------------------------


def lorenz96_section() -> dict[str, Any]:
    """Compare four ensemble filters and one smoother on the twin experiment.

    References:
        [EQ-FILT-LORENZ-002], [EQ-DA-ENKF-001], [EQ-DA-ETKF-001],
        [EQ-DA-LOC-002], [EQ-DA-LETKF-001], [EQ-DA-RELAX-002],
        [EQ-DA-ENKS-001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """
    start = tuple(8.0 + (0.01 if i == 0 else 0.0) for i in range(40))
    spun = integrate_fixed_step(lorenz96_rhs, 0.0, start, step=0.05, steps=400).final_state
    model = lorenz96(initial_mean=spun)
    truth = simulate(model, CYCLES, seed=1)
    localization = Localization(range(40), range(40), 2.0, period=40)
    relaxation = Relaxation("rtps", 0.5)
    settings: dict[str, dict[str, Any]] = {
        "EnKF": {},
        "EnKF, localized": {"localization": localization, "relaxation": relaxation},
        "ETKF": {"analysis": "etkf", "relaxation": relaxation},
        "LETKF": {"analysis": "letkf", "localization": localization, "relaxation": relaxation},
    }
    print(
        f"1. Lorenz-96 twin (J = 40, F = 8, R = I, N = {MEMBERS}, {CYCLES} cycles, "
        f"averages over cycles {BURN + 1}-{CYCLES})"
    )
    print("   filter            RMSE    spread   spread/RMSE   seconds")
    series: dict[str, tuple[list[float], list[float]]] = {}
    for name, options in settings.items():
        began = time.perf_counter()
        result = ensemble_kalman_filter(
            model, truth.observations, ensemble_size=MEMBERS, seed=SEED, **options
        )
        seconds = time.perf_counter() - began
        errors = [rmse(result.analysis_means[k], truth.states[k]) for k in range(1, CYCLES + 1)]
        spreads = [
            math.sqrt(math.fsum(result.analysis_covariances[k][i][i] for i in range(40)) / 40)
            for k in range(1, CYCLES + 1)
        ]
        series[name] = (errors, spreads)
        error, spread = statistics.fmean(errors[BURN:]), statistics.fmean(spreads[BURN:])
        print(
            f"   {name:16s}  {error:5.3f}   {spread:5.3f}    {spread / error:6.2f}      "
            f"{seconds:5.1f}"
        )
    smoother = ensemble_kalman_smoother(
        model, truth.observations, ensemble_size=MEMBERS, lag=LAG, seed=SEED,
        **settings["LETKF"],
    )  # fmt: skip
    window = range(BURN + 1, CYCLES + 1 - LAG)
    smoothed = statistics.fmean(rmse(smoother.smoothed_means[k], truth.states[k]) for k in window)
    filtered = statistics.fmean(
        rmse(smoother.filter.analysis_means[k], truth.states[k]) for k in window
    )
    print(
        f"   EnKS (LETKF + RTPS, lag {LAG}): RMSE {smoothed:.3f} against the filter's "
        f"{filtered:.3f} over cycles {BURN + 1}-{CYCLES - LAG}\n"
    )
    return {"series": series, "smoothed": smoothed, "filtered": filtered}


# --------------------------------------------------------------------------
# Part 2: Lorenz-63 4D-Var
# --------------------------------------------------------------------------


SPUN_UP = integrate_fixed_step(
    lorenz63_rhs, 0.0, (1.0, 1.0, 1.0), step=0.01, steps=1000
).final_state
B = diagonal(3, 4.0)
R = diagonal(3, 0.25)


def twin(seed: int, window: int) -> tuple[list[tuple[float, ...]], tuple[float, ...], list[tuple[float, ...]]]:
    """Return the truth ``x_0..x_K``, the background, and the observations.

    References:
        [EQ-FILT-LORENZ-001], [EQ-NUM-ODE-003].

    """
    generator = random.Random(seed)
    truth = [tuple(SPUN_UP)]
    for _ in range(window):
        truth.append(tuple(L63.transition_function(truth[-1])))
    background = tuple(v + 2.0 * generator.gauss(0.0, 1.0) for v in truth[0])
    observations = [
        tuple(v + 0.5 * generator.gauss(0.0, 1.0) for v in truth[k])
        for k in range(1, window + 1)
    ]
    return truth, background, observations


def forecast(state: Sequence[float], steps: int) -> list[tuple[float, ...]]:
    """Return the model trajectory ``x_0..x_steps`` from ``state``.

    References:
        [EQ-FILT-LORENZ-001], [EQ-NUM-ODE-003].

    """
    path = [tuple(state)]
    for _ in range(steps):
        path.append(tuple(L63.transition_function(path[-1])))
    return path


def strong(background: Sequence[float], observations: Sequence[Any]) -> Any:
    """Run strong-constraint 4D-Var with the adjoint gradient.

    References:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-002], [EQ-OPT-BFGS-001].

    """
    return four_dimensional_variational(
        background, B, observations, L63.transition_function, L63.observation_function, R
    )


def truth_cost(background: Sequence[float], observations: Sequence[Any], state: Sequence[float]) -> float:
    """Return the 4D-Var cost at a given initial state.

    References:
        [EQ-DA-4DVAR-001].

    """
    cost = four_dimensional_variational_cost(
        background, B, observations, L63.transition_function, L63.observation_function, R
    )
    return float(cost(tuple(state)))


def short_window_section() -> dict[str, Any]:
    """Run 4D-Var and its incremental form over a 10-interval window.

    References:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-003],
        [REF-LE-DIMET-TALAGRAND-1986], [REF-COURTIER-THEPAUT-HOLLINGSWORTH-1994].

    """
    truth, background, observations = twin(0, SHORT)
    result = strong(background, observations)
    incremental = incremental_four_dimensional_variational(
        background, B, observations, L63.transition_function, L63.observation_function, R
    )
    gap = max(abs(a - b) for a, b in zip(incremental.analysis, result.analysis, strict=True))
    print(f"2. Lorenz-63 strong-constraint 4D-Var, {SHORT}-interval window (twin seed 0)")
    print(
        f"   x_0 error: background {math.dist(background, truth[0]):.3f}, analysis "
        f"{math.dist(result.analysis, truth[0]):.3f}; end of window: background "
        f"{math.dist(forecast(background, SHORT)[-1], truth[-1]):.3f}, analysis "
        f"{math.dist(result.trajectory[-1], truth[-1]):.3f}"
    )
    print(
        f"   BFGS: {result.iterations} iterations, {result.evaluations} evaluations, "
        f"converged={result.converged}, J = {result.cost:.2f} (J(truth) = "
        f"{truth_cost(background, observations, truth[0]):.2f})"
    )
    print(
        f"   incremental: {incremental.iterations} outer loops, costs "
        + ", ".join(f"{c:.1f}" for c in incremental.outer_costs[:5])
        + f", ...; max |x_inc - x_BFGS| = {gap:.1e}\n"
    )
    return {"truth": truth, "background": background, "observations": observations, "result": result}


def cycled(background: Sequence[float], observations: Sequence[Any], windows: int) -> list[tuple[float, ...]]:
    """Assimilate ``windows`` consecutive short windows, cycling the background.

    Each window's background is the previous analysis run forward to the
    window start; ``B`` stays ``4 I``.

    Returns:
        The analysis trajectory over all windows, ``x_0..x_(windows * K)``.

    References:
        [EQ-DA-4DVAR-001], [REF-CARRASSI-ETAL-2018, Section 3.2].

    """
    path: list[tuple[float, ...]] = []
    start = tuple(background)
    for window in range(windows):
        piece = observations[window * SHORT : (window + 1) * SHORT]
        result = strong(start, piece)
        trajectory = [tuple(state) for state in result.trajectory]
        path.extend(trajectory if window == 0 else trajectory[1:])
        start = trajectory[-1]
    return path


def long_window_section() -> dict[str, Any]:
    """Compare one 40-interval window with four cycled 10-interval windows.

    References:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-002], [REF-CARRASSI-ETAL-2018].

    """
    print(
        f"3. One {LONG}-interval window against {LONG // SHORT} cycled "
        f"{SHORT}-interval windows (twin seeds {TWIN_SEEDS[0]}-{TWIN_SEEDS[-1]})"
    )
    print("   seed   J(analysis)   J(truth)   iterations   end error: long   cycled")
    shown: dict[str, Any] = {}
    stuck = 0
    for seed in TWIN_SEEDS:
        truth, background, observations = twin(seed, LONG)
        result = strong(background, observations)
        reference = truth_cost(background, observations, truth[0])
        path = cycled(background, observations, LONG // SHORT)
        local = result.cost > 2.0 * reference
        stuck += local
        print(
            f"   {seed:4d}   {result.cost:11.1f}   {reference:8.1f}   {result.iterations:10d}   "
            f"{math.dist(result.trajectory[-1], truth[-1]):15.3f}   "
            f"{math.dist(path[-1], truth[-1]):6.3f}" + ("   local minimum" if local else "")
        )
        if local and not shown:
            shown = {
                "seed": seed, "truth": truth, "observations": observations,
                "long": result, "cycled": path, "reference": reference,
            }  # fmt: skip
    print(f"   {stuck} of {len(TWIN_SEEDS)} long windows end in a local minimum (J > 2 J(truth))\n")
    return shown


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def save(figure: Any, name: str) -> str:
    """Save and close one figure under ``examples/generated``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import save_figure

    path = save_figure(figure, OUTPUT / name)
    plt.close(figure)
    return path.name


def rmse_figure(part: dict[str, Any]) -> str:
    """Draw analysis RMSE and spread over the cycles, one panel per filter.

    References:
        [EQ-DA-ENKF-002], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    plt = theme.pyplot()
    figure, grid = plt.subplots(
        2, 2, figsize=(7.6, 5.4), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.3, "wspace": 0.08},
    )  # fmt: skip
    palette = theme.active_palette()
    cycles = list(range(1, CYCLES + 1))
    error_colour = theme.role_colour("model")
    spread_colour = theme.role_colour("series-5")
    for axes, (name, (errors, spreads)) in zip(grid.flat, part["series"].items(), strict=True):
        theme.style_axes(axes)
        theme.reference_line(axes, y=1.0, entity="threshold")
        theme.line(axes, cycles, errors, color=error_colour)
        theme.line(axes, cycles, spreads, color=spread_colour, dashes=theme.FITTED)
        axes.set_yscale("log")
        axes.set_xlim(0.0, CYCLES)
        axes.set_ylim(0.06, 20.0)
        axes.set_yticks([0.1, 0.3, 1.0, 3.0, 10.0])
        axes.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
        axes.minorticks_off()
        axes.axvspan(0.0, BURN, color=palette.rule, alpha=0.35, linewidth=0.0, zorder=0)
        error = statistics.fmean(errors[BURN:])
        spread = statistics.fmean(spreads[BURN:])
        axes.text(
            0.97, 0.97, f"{name}: RMSE {error:.2f}, spread {spread:.2f}",
            transform=axes.transAxes, ha="right", va="top",
            fontsize=theme.ANNOTATION_SIZE, fontweight="semibold", color=palette.ink,
        )  # fmt: skip
    first = grid[0][0]
    for axes in grid[1]:
        theme.axis_labels(axes, "Assimilation cycle", None)
    for axes in grid[:, 0]:
        theme.axis_labels(axes, None, "Analysis RMSE, spread")
    letkf = statistics.fmean(part["series"]["LETKF"][0][BURN:])
    plain = statistics.fmean(part["series"]["EnKF"][0][BURN:])
    theme.title_block(
        first,
        f"Localization keeps 20 members on Lorenz-96 at RMSE {letkf:.2f}; the "
        f"plain EnKF diverges to {plain:.2f} with a collapsed spread",
        "Solid: analysis RMSE · dashed: ensemble spread · line at 1: observation "
        f"error\nShaded: cycles 1-{BURN}, left out of the averages · truth seed 1, "
        f"filter seed {SEED}",
    )
    return save(figure, "lorenz96-rmse-spread.png")


def trajectory_figure(short: dict[str, Any], long: dict[str, Any]) -> str:
    """Draw the x-component of truth, background, and 4D-Var analyses.

    References:
        [EQ-DA-4DVAR-001], [REF-MATPLOTLIB].

    """
    from probstat.visualization import theme

    plt = theme.pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.6, 4.2), sharey=True, gridspec_kw={"wspace": 0.08,
        "width_ratios": (1.0, 2.0)},
    )  # fmt: skip
    palette = theme.active_palette()
    truth_style = theme.entity_style("truth")
    model = theme.role_colour("model")
    context = theme.role_colour("context")
    other = theme.role_colour("series-4")
    step = 0.05

    def times(count: int) -> list[float]:
        """Return the observation times ``0, 0.05, ...``.

        References:
            [EQ-NUM-ODE-003]. No other equation is evaluated.

        """
        return [step * k for k in range(count)]

    def draw_truth(axes: Any, truth: Sequence[Sequence[float]], observations: Any) -> None:
        """Draw the true x-component and the observations of it.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        theme.line(
            axes, times(len(truth)), [s[0] for s in truth], color=truth_style.colour(),
            dashes=truth_style.dashes, width=truth_style.width, zorder=4.0,
        )  # fmt: skip
        theme.points(
            axes, times(len(truth))[1:], [y[0] for y in observations],
            entity="observations", size=16.0, alpha=0.8,
        )  # fmt: skip

    theme.style_axes(left)
    result = short["result"]
    background_path = forecast(short["background"], SHORT)
    theme.line(left, times(SHORT + 1), [s[0] for s in background_path], color=context, dashes=theme.FITTED)
    theme.line(left, times(SHORT + 1), [s[0] for s in result.trajectory], color=model)
    draw_truth(left, short["truth"], short["observations"])
    theme.direct_labels(
        left,
        [
            (step * SHORT, background_path[-1][0], "background", context),
            (step * SHORT, result.trajectory[-1][0], "4D-Var", model),
        ],
        gap_points=13.0,
    )
    left.set_xlim(0.0, step * SHORT)
    theme.axis_labels(left, "Time", "x-component of Lorenz-63")
    left.text(
        0.03, 0.03, f"{SHORT} intervals, seed 0", transform=left.transAxes,
        ha="left", va="bottom", fontsize=theme.ANNOTATION_SIZE, color=palette.ink,
    )  # fmt: skip

    theme.style_axes(right)
    span = times(LONG + 1)
    theme.line(right, span, [s[0] for s in long["long"].trajectory], color=other, dashes=theme.FITTED)
    theme.line(right, span, [s[0] for s in long["cycled"]], color=model)
    draw_truth(right, long["truth"], long["observations"])
    for window in range(1, LONG // SHORT):
        right.axvline(step * SHORT * window, color=palette.rule, linewidth=0.8, zorder=0)
    theme.direct_labels(
        right,
        [
            (step * LONG, long["long"].trajectory[-1][0], "one long window", other),
            (step * LONG, long["cycled"][-1][0], "4 cycled windows", model),
        ],
        gap_points=13.0,
    )
    right.set_xlim(0.0, step * LONG)
    theme.axis_labels(right, "Time", None)
    right.text(
        0.02, 0.03,
        f"{LONG} intervals, seed {long['seed']}: J = {long['long'].cost:.0f} against "
        f"J(truth) = {long['reference']:.0f}",
        transform=right.transAxes, ha="left", va="bottom",
        fontsize=theme.ANNOTATION_SIZE, color=palette.ink,
    )  # fmt: skip
    theme.title_block(
        left,
        "A short window pulls the background onto the truth; a long window "
        "converges to a wrong trajectory that cycling short windows avoids",
        "Dashed accent: truth · dots: observations of x (R = 0.25) · grey rules: "
        "window boundaries of the cycled analysis · B = 4 I",
    )
    return save(figure, "lorenz63-4dvar-trajectories.png")


def main() -> None:
    """Run both parts, write the figures, and print the elapsed time.

    References:
        [REF-EVENSEN-2003], [REF-LE-DIMET-TALAGRAND-1986].

    """
    start = time.perf_counter()
    part = lorenz96_section()
    short = short_window_section()
    long = long_window_section()
    if has_matplotlib():
        written = [rmse_figure(part)]
        if long:
            written.append(trajectory_figure(short, long))
        print("figures: " + ", ".join(written))
    else:
        print("Matplotlib is not installed; figures skipped")
    print(f"elapsed {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
