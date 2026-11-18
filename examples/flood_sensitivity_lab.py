"""Sensitivity laboratory: the inputs that drive the overflow and cost of a dike.

The flood model computes the water level above a dike crest (overflow ``S``,
in metres) and the annual maintenance cost of the dike (``C``, in million
euros) from eight uncertain inputs: discharge ``Q``, Strickler coefficient
``Ks``, river-bed levels ``Zv`` and ``Zm``, dike height ``Hd``, bank level
``Cb``, section length ``L``, and river width ``B``. The laboratory runs the
sensitivity measures of a course on the same model, from the cheapest and
most local to the most general: quadratic-sum (first-order Taylor)
importance factors, with finite differences checked against exact
derivatives; Morris screening (``mu*``, ``sigma``, and the signed ``mu``);
standardized regression and rank-regression coefficients; Sobol indices
with four estimators on one design each; and HSIC dependence measures,
including target HSIC on the top decile of the overflow. The overflow is
nearly linear in its inputs and every measure agrees. The cost is not: it
falls and then rises with the dike height, a non-monotone effect that a
linear regression misses and that Morris, Sobol and HSIC detect. Run with
``PYTHONPATH=src python examples/flood_sensitivity_lab.py``; figures are
written to ``examples/generated/``.

Definitions:
    ``H = (Q / (Ks B sqrt((Zm - Zv) / L)))^0.6``: water height; overflow
        ``S = Zv + H - Hd - Cb`` (m), flooding when ``S > 0``.
    ``C``: cost (MEUR), 1 when ``S >= 0``, otherwise
        ``0.2 + 0.8 (1 - exp(-1000 / S^4))``, plus ``max(Hd, 8) / 20``.
    ``eta_i^2 = (dg/dx_i)^2 sigma_i^2 / sum_j (dg/dx_j)^2 sigma_j^2``:
        quadratic-sum importance factor at a nominal point.
    ``mu*``, ``sigma``, ``mu``: mean absolute, standard deviation, and mean of
        Morris elementary effects on each input's probability scale.
    ``SRC_i = b_i s_i / s_y`` from the least-squares fit of ``y`` on the
        inputs; ``R^2`` its coefficient of determination; SRRC and rank
        ``R^2`` the same on ranks.
    ``S_i``, ``S_Ti``: first-order and total Sobol indices.
    ``R2_HSIC``: normalized Hilbert-Schmidt independence criterion of an
        input and the output (or the event ``S > c``).

Assumptions:
    - The eight inputs are independent with the published laws of the
      benchmark (``probstat.modeling.flood_model``); Taylor factors use the
      nominal point and standard deviations of the course instead.
    - Regression indices are variance shares only when ``R^2`` is close to
      one; HSIC values depend on the kernel bandwidth (the sample standard
      deviation) and are compared through ``R2_HSIC`` and p-values.
    - Sobol and Morris need independent inputs; the regression and HSIC
      tests need i.i.d. rows, so their sample is plain Monte Carlo.

Equations:
    EQ-MOD-FLOOD-001: overflow of the flood model.
    EQ-MOD-FLOOD-002: maintenance cost of the flood model.
    EQ-SENS-TAYLOR-001: quadratic-sum importance factors.
    EQ-SENS-LOCAL-001: central-difference derivatives.
    EQ-SENS-MORRIS-001: Morris elementary effects ``mu``, ``mu*``, ``sigma``.
    EQ-SENS-SRC-001: standardized regression coefficients.
    EQ-SENS-PCC-001: partial correlation coefficients.
    EQ-SENS-RANK-001: rank-transformed SRRC and PRCC.
    EQ-SENS-FIRST-001: first-order Sobol index (default estimator).
    EQ-SENS-TOTAL-001: total Sobol index (default estimator).
    EQ-SENS-RQMC-001: replicate intervals over randomized Sobol designs.
    EQ-SENS-EST-003: Jansen estimators.
    EQ-SENS-EST-004: Martinez (correlation) estimators.
    EQ-SENS-EST-006: Janon estimators.
    EQ-SENS-HSIC-001: biased HSIC estimate and ``R2_HSIC``.
    EQ-SENS-HSIC-003: Gamma approximation of the HSIC independence test.
    EQ-SENS-HSIC-005: target HSIC of a thresholded output.
    EQ-STAT-RANK-001: scaled ranks of the cobweb plot.
    EQ-SMOOTH-009: lowess smoother of the input scatterplots.

Diagnostics:
    - Taylor: the relative error of each finite-difference derivative of the
      overflow against its exact value, and the Taylor variance against the
      sampled variance of ``S``.
    - Morris: ``|mu|`` close to ``mu*`` means a monotone effect; ``|mu|`` well
      below ``mu*`` means effects of both signs.
    - Regression: ``R^2`` and ``sum SRC^2``; the library adds a note when
      ``R^2 < 0.7``. A rank ``R^2`` above ``R^2`` points to a monotone
      nonlinear response.
    - Sobol: replicate 95% intervals over 8 randomizations, and the spread of
      four estimators on the same design size.
    - HSIC: Gamma p-values of independence; inputs with ``p > 0.05`` are
      reported as not detected.

Cost and error:
    - Taylor: 16 model calls per output (central differences in 8 inputs).
    - Morris: ``r (d + 1) = 900`` calls per output with 100 trajectories.
    - Regression and HSIC: one Monte Carlo sample of 1000 runs, both outputs
      from each run; the HSIC Gamma test costs ``O(n^2)`` per input.
    - Sobol: ``N (d + 2) = 40 960`` calls per estimator for both outputs
      (``N = 4096``); replicate half-widths are at most about 0.02 here.
    - The laboratory runs in about two minutes, most of it in the four Sobol
      designs and the HSIC Gram matrices; it prints its wall-clock time.

References:
    REF-IOOSS-LEMAITRE-2015: the flood model and its input laws.
    REF-SALTELLI-2008: local, screening, regression, and variance-based
        sensitivity.
    REF-MORRIS-1991: elementary effects.
    REF-CAMPOLONGO-CARIBONI-SALTELLI-2007: ``mu*``.
    REF-HELTON-ETAL-2006: sampling-based sensitivity and cobweb plots.
    REF-SALTELLI-2010: Sobol estimators.
    REF-JANSEN-1999: the Jansen estimator.
    REF-BAUDIN-ETAL-2016: the Martinez estimator.
    REF-JANON-ETAL-2014: the Janon estimator.
    REF-GRETTON-ETAL-2005: the Hilbert-Schmidt independence criterion.
    REF-DA-VEIGA-2015: HSIC sensitivity indices.
    REF-MARREL-CHABRIDON-2021: target HSIC.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, variance
from typing import Any

from probstat.modeling import (
    FLOOD_INPUT_NAMES,
    FLOOD_NOMINAL,
    FLOOD_NOMINAL_STANDARD_DEVIATIONS,
    flood_model,
)
from probstat.sensitivity import (
    SOBOL_ESTIMATORS,
    hsic_indices,
    morris_screening,
    regression_sensitivity,
    sobol_analysis_all,
    taylor_importance_factors,
)
from probstat.statistics import quantile
from probstat.visualization import (
    plot_cobweb,
    plot_input_scatter,
    save_figure,
    theme,
)

SEED = 2026
SAMPLE_SIZE = 1000
SOBOL_SIZE = 4096
SOBOL_REPLICATES = 8
MORRIS_TRAJECTORIES = 100
ESTIMATORS = ("saltelli2010", "jansen", "martinez", "janon")
OUTPUTS = ("overflow", "cost")
OUTPUT_LABELS = {"overflow": "Overflow S", "cost": "Cost C"}
UNITS = {"overflow": "m", "cost": "MEUR"}
INPUT_UNITS = ("m3/s", "", "m", "m", "m", "m", "m", "m")
TOP_FRACTION = 0.1
HIGHLIGHT = "Hd"
OUTPUT = Path(__file__).parent / "generated"


# --------------------------------------------------------------------------
# Local measures: Taylor importance factors and Morris screening
# --------------------------------------------------------------------------


def overflow_gradient(x: Sequence[float]) -> tuple[float, ...]:
    """Return the exact gradient of the overflow ``S`` in model input order.

    ``dS/dQ = 0.6 H / Q``, ``dS/dKs = -0.6 H / Ks``,
    ``dS/dZv = 1 + 0.3 H / (Zm - Zv)``, ``dS/dZm = -0.3 H / (Zm - Zv)``,
    ``dS/dHd = dS/dCb = -1``, ``dS/dL = 0.3 H / L``, ``dS/dB = -0.6 H / B``.

    Args:
        x: ``(Q, Ks, Zv, Zm, Hd, Cb, L, B)``.

    Returns:
        The eight partial derivatives.

    References:
        [EQ-MOD-FLOOD-001], [REF-IOOSS-LEMAITRE-2015].

    """
    q, ks, zv, zm, _, _, length, width = x
    height = (q / (ks * width * math.sqrt((zm - zv) / length))) ** 0.6
    gap = zm - zv
    return (
        0.6 * height / q,
        -0.6 * height / ks,
        1.0 + 0.3 * height / gap,
        -0.3 * height / gap,
        -1.0,
        -1.0,
        0.3 * height / length,
        -0.6 * height / width,
    )


@dataclass(frozen=True)
class TaylorStudy:
    """Quadratic-sum importance factors of both outputs.

    Args:
        differences: ``output -> result`` with 1% central differences at
            the nominal point.
        exact: Overflow factors from the exact gradient.
        law_means: Overflow factors at the means and standard deviations of
            the input law.

    References:
        [EQ-SENS-TAYLOR-001], [REF-SALTELLI-2008, Chapter 1].

    """

    differences: Mapping[str, Any]
    exact: Any
    law_means: Any


@functools.lru_cache(maxsize=1)
def taylor_study() -> TaylorStudy:
    """Compute Taylor importance factors by differences and exactly.

    References:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-LOCAL-001], [REF-SALTELLI-2008].

    """
    model = flood_model()
    nominal = {
        "point": FLOOD_NOMINAL,
        "standard_deviations": FLOOD_NOMINAL_STANDARD_DEVIATIONS,
    }
    differences = {
        name: taylor_importance_factors(model, name, **nominal) for name in OUTPUTS
    }
    exact = taylor_importance_factors(
        model, "overflow", gradient=overflow_gradient, **nominal
    )
    law_means = taylor_importance_factors(model, "overflow")
    return TaylorStudy(differences, exact, law_means)


@functools.lru_cache(maxsize=1)
def morris_study() -> Mapping[str, Any]:
    """Run Morris screening on both outputs with the same trajectories.

    References:
        [EQ-SENS-MORRIS-001], [REF-MORRIS-1991],
        [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007].

    """
    model = flood_model()
    return {
        name: morris_screening(model, name, MORRIS_TRAJECTORIES, seed=SEED)
        for name in OUTPUTS
    }


# --------------------------------------------------------------------------
# Sampling-based measures on one Monte Carlo sample
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardSample:
    """One plain Monte Carlo sample of the flood model.

    Args:
        rows: Input vectors in model order.
        outputs: ``output -> values`` for overflow and cost.

    References:
        [EQ-MOD-FLOOD-001], [EQ-MOD-FLOOD-002], [REF-IOOSS-LEMAITRE-2015].

    """

    rows: tuple[tuple[float, ...], ...]
    outputs: Mapping[str, tuple[float, ...]]

    @property
    def overflow_threshold(self) -> float:
        """Return the empirical ``1 - TOP_FRACTION`` quantile of the overflow.

        References:
            [EQ-STAT-QUANT-001], [REF-HYNDMAN-FAN-1996].

        """
        return quantile(self.outputs["overflow"], 1.0 - TOP_FRACTION)


@functools.lru_cache(maxsize=1)
def forward_sample() -> ForwardSample:
    """Draw ``SAMPLE_SIZE`` input rows and run the model once per row.

    References:
        [EQ-MOD-FLOOD-001], [EQ-MOD-FLOOD-002], [REF-HELTON-ETAL-2006].

    """
    model = flood_model()
    rows = tuple(tuple(row) for row in model.inputs.sample(SAMPLE_SIZE, seed=SEED))
    runs = [model.evaluate(row) for row in rows]
    outputs = {name: tuple(run[name] for run in runs) for name in OUTPUTS}
    return ForwardSample(rows, outputs)


@functools.lru_cache(maxsize=1)
def regression_study() -> Mapping[str, Any]:
    """Fit linear and rank regressions of each output on the eight inputs.

    References:
        [EQ-SENS-SRC-001], [EQ-SENS-PCC-001], [EQ-SENS-RANK-001],
        [REF-HELTON-ETAL-2006].

    """
    sample = forward_sample()
    return {
        name: regression_sensitivity(
            sample.rows, sample.outputs[name], names=FLOOD_INPUT_NAMES, rank=True
        )
        for name in OUTPUTS
    }


@functools.lru_cache(maxsize=1)
def hsic_study() -> Mapping[str, Any]:
    """Compute HSIC of both outputs and target HSIC of the top overflow decile.

    References:
        [EQ-SENS-HSIC-001], [EQ-SENS-HSIC-003], [EQ-SENS-HSIC-005],
        [REF-DA-VEIGA-2015], [REF-MARREL-CHABRIDON-2021].

    """
    sample = forward_sample()
    results: dict[str, Any] = {
        name: hsic_indices(
            sample.rows, sample.outputs[name], names=FLOOD_INPUT_NAMES, test="gamma"
        )
        for name in OUTPUTS
    }
    results["overflow top decile"] = hsic_indices(
        sample.rows,
        sample.outputs["overflow"],
        names=FLOOD_INPUT_NAMES,
        test="gamma",
        target=sample.overflow_threshold,
    )
    return results


# --------------------------------------------------------------------------
# Variance-based measures: Sobol indices with several estimators
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def sobol_study() -> Mapping[str, Any]:
    """Estimate Sobol indices of both outputs with each estimator of the lab.

    Every estimator gets its own scrambled Sobol design of ``SOBOL_SIZE``
    rows (same seed, hence the same rows) split into ``SOBOL_REPLICATES``
    randomizations, so the intervals are replicate intervals.

    References:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [EQ-SENS-RQMC-001],
        [EQ-SENS-EST-003], [EQ-SENS-EST-004], [EQ-SENS-EST-006],
        [REF-SALTELLI-2010].

    """
    model = flood_model()
    unknown = set(ESTIMATORS) - set(SOBOL_ESTIMATORS)
    if unknown:
        raise ValueError(f"unknown Sobol estimators: {sorted(unknown)}")
    return {
        name: sobol_analysis_all(
            model,
            SOBOL_SIZE,
            seed=SEED,
            replicates=SOBOL_REPLICATES,
            estimator=name,
        )
        for name in ESTIMATORS
    }


# --------------------------------------------------------------------------
# Figures: each returns the figure and the formatted facts it states
# --------------------------------------------------------------------------


def _legible(colour: str, minimum: float = 4.5) -> str:
    """Mix a series colour toward the ink until text in it reaches ``minimum``.

    References:
        [REF-W3C-WCAG21-2018]. No equation of the registry is evaluated.

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = theme.active_palette()

    def luminance(rgb: Sequence[float]) -> float:
        """Return the WCAG relative luminance of an sRGB colour.

        References:
            [REF-W3C-WCAG21-2018].

        """
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    canvas = luminance(to_rgb(palette.canvas))
    base, ink = to_rgb(colour), to_rgb(palette.ink)
    for step in range(21):
        weight = step / 20
        mixed = [(1 - weight) * b + weight * i for b, i in zip(base, ink, strict=True)]
        value = luminance(mixed)
        light, dark = max(value, canvas), min(value, canvas)
        if (light + 0.05) / (dark + 0.05) >= minimum:
            return str(to_hex(mixed))
    return palette.ink


@dataclass(frozen=True)
class MeasureTable:
    """Four sensitivity measures of one output, input by input.

    Args:
        output: Output name.
        names: Input names, ordered by decreasing total Sobol index.
        columns: ``measure label -> values`` in the order of ``names``.

    References:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-SRC-001], [EQ-SENS-FIRST-001],
        [EQ-SENS-HSIC-001].

    """

    output: str
    names: tuple[str, ...]
    columns: Mapping[str, tuple[float, ...]]

    def value(self, measure: str, name: str) -> float:
        """Return one measure of one input.

        References:
            [EQ-SENS-FIRST-001]. No further equation is evaluated.

        """
        return self.columns[measure][self.names.index(name)]


def measure_table(output: str) -> MeasureTable:
    """Collect Taylor, SRC^2, Sobol ``S_i`` and ``R2_HSIC`` of one output.

    References:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-SRC-001], [EQ-SENS-FIRST-001],
        [EQ-SENS-HSIC-001], [REF-SALTELLI-2008].

    """
    sobol = sobol_study()[ESTIMATORS[0]][output]
    order = sorted(
        range(len(FLOOD_INPUT_NAMES)),
        key=lambda i: -sobol.result.total_effect[i],
    )
    raw = {
        "Taylor eta^2": taylor_study().differences[output].importance_factors,
        "SRC^2": regression_study()[output].src_squared,
        "Sobol S_i": sobol.result.first_order,
        "R2_HSIC": hsic_study()[output].r2_hsic,
    }
    return MeasureTable(
        output,
        tuple(FLOOD_INPUT_NAMES[i] for i in order),
        {label: tuple(values[i] for i in order) for label, values in raw.items()},
    )


def figure_measure_comparison(output: str = "cost") -> tuple[Any, dict[str, str]]:
    """Draw four sensitivity measures of one output as small multiples.

    Each panel is a lollipop per input; the dike height ``Hd`` is drawn in
    the accent colour so that its four values can be compared across panels.

    References:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-SRC-001], [EQ-SENS-FIRST-001],
        [EQ-SENS-HSIC-001], [REF-SALTELLI-2008].

    """
    table = measure_table(output)
    facts = {
        "src": f"{table.value('SRC^2', HIGHLIGHT):.3f}",
        "sobol": f"{table.value('Sobol S_i', HIGHLIGHT):.2f}",
        "hsic": f"{table.value('R2_HSIC', HIGHLIGHT):.2f}",
        "taylor": f"{table.value('Taylor eta^2', HIGHLIGHT):.2f}",
        "r2": f"{regression_study()[output].r_squared:.2f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(
        1, len(table.columns), figsize=(7.4, 4.0), sharey=True, squeeze=False
    )
    palette = theme.active_palette()
    accent = theme.role_colour("observed")
    positions = list(range(len(table.names)))[::-1]
    for ax, (label, values) in zip(axes[0], table.columns.items(), strict=True):
        theme.style_axes(ax)
        for position, name, value in zip(positions, table.names, values, strict=True):
            colour = accent if name == HIGHLIGHT else palette.muted
            ax.plot([0.0, value], [position, position], color=colour,
                    linewidth=1.6 if name == HIGHLIGHT else 1.0, zorder=2)  # fmt: skip
            ax.plot([value], [position], marker="o", color=colour,
                    markersize=6 if name == HIGHLIGHT else 4.5, zorder=3)  # fmt: skip
        top = max(0.25, 1.2 * max(values))
        step = 0.1 if top <= 0.3 else 0.2
        ax.set_xlim(-0.02 * top, top)
        ax.set_xticks([step * k for k in range(int(top / step) + 1)])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=palette.rule, linewidth=0.6)
        theme.axis_labels(ax, label)
        index = table.names.index(HIGHLIGHT)
        value = values[index]
        ax.annotate(
            f"{value:.3f}" if value < 0.01 else f"{value:.2f}",
            (value, positions[index]),
            xytext=(6.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=theme.ANNOTATION_SIZE,
            fontweight="bold",
            color=_legible(accent),
        )
    axes[0][0].set_yticks(positions)
    axes[0][0].set_yticklabels(table.names)
    axes[0][0].tick_params(axis="y", length=0)
    figure.subplots_adjust(wspace=0.18)
    if output == "cost":
        title = (
            f"Regression gives the dike height Hd {facts['src']} of the cost "
            f"variance; Sobol ({facts['sobol']}) and HSIC ({facts['hsic']}) "
            "see its V-shaped effect"
        )
    else:
        title = (
            "For the overflow the four measures agree: Q, Hd, Zv and Ks matter; "
            "Zm, B and L do not"
        )
    theme.title_block(
        axes[0][0],
        title,
        f"{OUTPUT_LABELS[output]} of the flood model · rows by total "
        f"Sobol index · linear fit R^2 = {facts['r2']} · R2_HSIC is not a "
        "variance share",
    )
    return figure, facts


def _top_decile_drivers(sample: ForwardSample) -> list[tuple[str, float]]:
    """Return inputs whose ranks in the top overflow decile depart from 0.5.

    The median scaled rank of the highlighted runs on each input is compared
    with the median 0.5 of all runs; inputs departing by more than 0.15 are
    returned, largest departure first.

    References:
        [EQ-STAT-RANK-001], [REF-HELTON-ETAL-2006].

    """
    cut = sample.overflow_threshold
    flagged = [value > cut for value in sample.outputs["overflow"]]
    n = len(sample.rows)
    shifts: list[tuple[str, float]] = []
    for index, name in enumerate(FLOOD_INPUT_NAMES):
        column = [row[index] for row in sample.rows]
        order = sorted(range(n), key=column.__getitem__)
        ranks = [0.0] * n
        for position, row in enumerate(order):
            ranks[row] = position / (n - 1)
        chosen = sorted(ranks[k] for k in range(n) if flagged[k])
        middle = len(chosen) // 2
        median = (
            chosen[middle]
            if len(chosen) % 2
            else 0.5 * (chosen[middle - 1] + chosen[middle])
        )
        if abs(median - 0.5) > 0.15:
            shifts.append((name, median - 0.5))
    return sorted(shifts, key=lambda item: -abs(item[1]))


def figure_cobweb() -> tuple[Any, dict[str, str]]:
    """Draw the cobweb plot of the forward sample with the top overflow decile.

    References:
        [EQ-STAT-RANK-001], [REF-INSELBERG-1985], [REF-HELTON-ETAL-2006].

    """
    sample = forward_sample()
    cut = sample.overflow_threshold
    count = sum(value > cut for value in sample.outputs["overflow"])
    drivers = _top_decile_drivers(sample)
    words = [f"{'high' if shift > 0 else 'low'} {name}" for name, shift in drivers]
    facts = {
        "count": f"{count}",
        "threshold": f"{cut:.2f} m",
        "drivers": ", ".join(words[:-1]) + f" and {words[-1]}",
    }
    ax = plot_cobweb(
        sample.rows,
        FLOOD_INPUT_NAMES,
        output=sample.outputs["overflow"],
        threshold=cut,
        output_name="S",
        title=f"The {facts['count']} runs with the highest overflow (S > "
        f"{facts['threshold']}) come from {facts['drivers']}",
        subtitle=f"Each line is one of {len(sample.rows):,} Monte Carlo runs on "
        "scaled ranks · coloured: top decile of S; grey: the others",
    )
    return ax.figure, facts


def figure_top_decile_scatter() -> tuple[Any, dict[str, str]]:
    """Draw the overflow against each input with its top decile highlighted.

    The same runs and threshold as the cobweb plot, drawn as one scatterplot
    per input: the highlighted runs sit at one end of the inputs that drive
    the overflow and spread over the whole range of the others.

    References:
        [EQ-MOD-FLOOD-001], [REF-HELTON-ETAL-2006], [REF-SALTELLI-2008].

    """
    sample = forward_sample()
    cut = sample.overflow_threshold
    values = sample.outputs["overflow"]
    flagged = [value > cut for value in values]
    drivers = _top_decile_drivers(sample)
    words = [f"{'high' if shift > 0 else 'low'} {name}" for name, shift in drivers]
    facts = {
        "count": f"{sum(flagged)}",
        "threshold": f"{cut:.2f} m",
        "drivers": ", ".join(words[:-1]) + f" and {words[-1]}",
    }
    from matplotlib.ticker import MaxNLocator

    plt = theme.pyplot()
    figure, grid = plt.subplots(2, 4, figsize=(7.4, 4.8), sharey=True)
    palette = theme.active_palette()
    highlight = theme.entity_colour("failed samples")
    threshold = theme.entity_colour("threshold")
    for index, ax in enumerate(grid.flat):
        theme.style_axes(ax)
        column = [row[index] for row in sample.rows]
        for chosen, colour, size, alpha, order in (
            (False, palette.muted, 3.0, 0.35, 2.0),
            (True, highlight, 5.0, 0.8, 3.0),
        ):
            xs = [x for x, mark in zip(column, flagged, strict=True) if mark is chosen]
            ys = [y for y, mark in zip(values, flagged, strict=True) if mark is chosen]
            ax.scatter(xs, ys, s=size, color=colour, alpha=alpha,
                       edgecolors="none", zorder=order)  # fmt: skip
        ax.axhline(cut, color=threshold, linewidth=theme.REFERENCE_WIDTH,
                   linestyle=theme.REFERENCE, zorder=4)  # fmt: skip
        name = FLOOD_INPUT_NAMES[index]
        unit = INPUT_UNITS[index]
        theme.axis_labels(ax, name, x_unit=unit or None)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.grid(axis="x", visible=False)
    for ax in grid[:, 0]:
        theme.axis_labels(ax, None, OUTPUT_LABELS["overflow"], y_unit="m")
    figure.subplots_adjust(wspace=0.12, hspace=0.62)
    theme.title_block(
        grid[0][0],
        f"The {facts['count']} runs with the highest overflow (S > "
        f"{facts['threshold']}) come from {facts['drivers']}",
        f"{len(sample.rows):,} Monte Carlo runs · coloured: top decile of S · "
        "dashed: its threshold",
    )
    return figure, facts


def figure_input_scatter(output: str = "cost") -> tuple[Any, dict[str, str]]:
    """Draw one output against every input with lowess smoothers.

    References:
        [EQ-SMOOTH-009], [REF-CLEVELAND-1979], [REF-HELTON-ETAL-2006].

    """
    sample = forward_sample()
    values = sample.outputs[output]
    index = FLOOD_INPUT_NAMES.index(HIGHLIGHT)
    low = fmean(
        v for v, row in zip(values, sample.rows, strict=True) if row[index] < 7.4
    )
    middle = fmean(
        v for v, row in zip(values, sample.rows, strict=True) if 7.8 < row[index] < 8.2
    )
    high = fmean(
        v for v, row in zip(values, sample.rows, strict=True) if row[index] > 8.6
    )
    facts = {"low": f"{low:.2f}", "middle": f"{middle:.2f}", "high": f"{high:.2f}"}
    figure = plot_input_scatter(
        sample.rows,
        values,
        FLOOD_INPUT_NAMES,
        output_name=OUTPUT_LABELS[output],
        units=INPUT_UNITS,
        output_unit=UNITS[output],
        title=f"Mean cost falls from {facts['low']} to {facts['middle']} MEUR as Hd "
        f"rises to 8 m, then climbs to {facts['high']}",
        subtitle=f"{len(sample.rows):,} Monte Carlo runs · line: lowess smoother "
        "of the output given each input",
    )
    return figure, facts


def figure_morris() -> tuple[Any, dict[str, str]]:
    """Draw Morris ``mu*`` against ``sigma`` for both outputs.

    References:
        [EQ-SENS-MORRIS-001], [REF-MORRIS-1991],
        [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007].

    """
    results = morris_study()
    cost = results["cost"]
    index = cost.input_names.index(HIGHLIGHT)
    facts = {
        "mu": f"{cost.mu[index]:.3f}",
        "mu_star": f"{cost.mu_star[index]:.3f}",
    }
    plt = theme.pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), squeeze=False)
    palette = theme.active_palette()
    accent = theme.role_colour("observed")
    for ax, name in zip(axes[0], OUTPUTS, strict=True):
        theme.style_axes(ax)
        result = results[name]
        largest = max(result.mu_star)
        for label, star, spread in zip(
            result.input_names, result.mu_star, result.sigma, strict=True
        ):
            colour = accent if label == HIGHLIGHT else palette.muted
            ax.plot([star], [spread], marker="o", color=colour, markersize=6,
                    zorder=3)  # fmt: skip
            if star > 0.05 * largest:
                ax.annotate(label, (star, spread), xytext=(5.0, 3.0),
                            textcoords="offset points", ha="left", va="bottom",
                            fontsize=theme.ANNOTATION_SIZE,
                            color=_legible(colour), fontweight="bold")  # fmt: skip
        ax.set_xlim(0.0, 1.25 * largest)
        ax.set_ylim(0.0, 1.3 * max(max(result.sigma), 0.1 * largest))
        theme.axis_labels(
            ax,
            f"mu* of {OUTPUT_LABELS[name]} [{UNITS[name]}]",
            "Spread of effects sigma" if name == OUTPUTS[0] else None,
        )
    figure.subplots_adjust(wspace=0.28)
    theme.title_block(
        axes[0][0],
        f"For the cost, Hd has mu* = {facts['mu_star']} but a signed mean of "
        f"only {facts['mu']}: its effect changes sign",
        f"Morris screening, {MORRIS_TRAJECTORIES} trajectories per output · "
        "unlabelled: mu* below 5% of the largest",
    )
    return figure, facts


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _row(values: Sequence[float], digits: int = 3) -> str:
    """Format one row of eight input values.

    References:
        [REF-SALTELLI-2008]. No equation is evaluated.

    """
    return " ".join(f"{value:>7.{digits}f}" for value in values)


def _save(name: str, draw: Callable[[], tuple[Any, dict[str, str]]]) -> None:
    """Draw one figure, save it to ``OUTPUT``, and close it.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    with theme.chart_theme(theme.active_palette()):
        figure, _ = draw()
        path = save_figure(figure, OUTPUT / f"{name}.png")
    theme.pyplot().close(figure)
    print(f"  figure: {path.relative_to(Path(__file__).parent)}")


def main() -> None:
    """Run every sensitivity measure on both outputs and print the results.

    References:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-MORRIS-001], [EQ-SENS-SRC-001],
        [EQ-SENS-FIRST-001], [EQ-SENS-HSIC-001], [REF-SALTELLI-2008].

    """
    import matplotlib

    matplotlib.use("Agg")
    start = time.perf_counter()
    header = " ".join(f"{name:>7}" for name in FLOOD_INPUT_NAMES)
    print(f"Flood model, inputs {', '.join(FLOOD_INPUT_NAMES)}")

    print("\n1. Quadratic sum (first-order Taylor) at the nominal point")
    taylor = taylor_study()
    exact = taylor.exact
    fd = taylor.differences["overflow"]
    errors = [
        abs(a - b) / abs(b) for a, b in zip(fd.gradient, exact.gradient, strict=True)
    ]
    print(f"  {'':<26}{header}")
    print(f"  {'S: relative FD error':<26}{_row(errors, 6)}")
    print(f"  {'S: eta^2, 1% differences':<26}{_row(fd.importance_factors, 4)}")
    print(f"  {'S: eta^2, exact gradient':<26}{_row(exact.importance_factors, 4)}")
    means = taylor.law_means.importance_factors
    print(f"  {'S: eta^2 at the law means':<26}{_row(means, 4)}")
    cost_fd = taylor.differences["cost"]
    print(f"  {'C: eta^2, 1% differences':<26}{_row(cost_fd.importance_factors, 4)}")
    sample = forward_sample()
    print(
        f"  Taylor variance of S: {fd.taylor_variance:.3f} "
        f"(sampled variance {variance(sample.outputs['overflow']):.3f}); "
        f"{fd.evaluations} calls per output"
    )
    print(
        "  C has a kink at Hd = 8 (max(Hd, 8) / 20), exactly at the nominal "
        "point: the central difference averages the slopes 0 and 1/20"
    )

    print(f"\n2. Morris screening, {MORRIS_TRAJECTORIES} trajectories")
    for name, result in morris_study().items():
        print(f"  {OUTPUT_LABELS[name]} ({result.evaluations} calls)")
        print(f"    {'':<8}{header}")
        print(f"    {'mu*':<8}{_row(result.mu_star, 4)}")
        print(f"    {'sigma':<8}{_row(result.sigma, 4)}")
        print(f"    {'mu':<8}{_row(result.mu, 4)}")

    print(f"\n3. Regression on {SAMPLE_SIZE} Monte Carlo runs")
    for name, result in regression_study().items():
        print(
            f"  {OUTPUT_LABELS[name]}: R^2 = {result.r_squared:.3f}, sum SRC^2 = "
            f"{result.sum_src_squared:.3f}, rank R^2 = {result.rank_r_squared:.3f}"
        )
        print(f"    {'':<8}{header}")
        print(f"    {'SRC^2':<8}{_row(result.src_squared)}")
        print(f"    {'SRRC^2':<8}{_row(result.srrc_squared or ())}")
        print(f"    {'PCC':<8}{_row(result.pcc)}")
        for line in result.notes:
            print(f"    note: {line}")

    print(
        f"\n4. Sobol indices, {SOBOL_SIZE} rows x {SOBOL_REPLICATES} replicates "
        "per estimator"
    )
    sobol = sobol_study()
    for name in OUTPUTS:
        print(f"  {OUTPUT_LABELS[name]}")
        print(f"    {'':<20}{header}")
        for estimator in ESTIMATORS:
            result = sobol[estimator][name]
            print(f"    {estimator + ' S_i':<20}{_row(result.result.first_order)}")
            print(f"    {estimator + ' S_Ti':<20}{_row(result.result.total_effect)}")
        default = sobol[ESTIMATORS[0]][name]
        widths = [
            0.5 * (interval.upper - interval.lower) for interval in default.first_order
        ]
        print(f"    {'95% half-width S_i':<20}{_row(widths)}")
        print(
            f"    sum of S_i ({ESTIMATORS[0]}): "
            f"{math.fsum(default.result.first_order):.3f}; "
            f"{default.evaluations} calls for both outputs"
        )

    print(f"\n5. HSIC on the same {SAMPLE_SIZE} runs (Gamma test)")
    for name, result in hsic_study().items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"  {label}: not detected at 5%: {', '.join(result.independent(0.05))}")
        print(f"    {'':<10}{header}")
        print(f"    {'R2_HSIC':<10}{_row(result.r2_hsic)}")
        print(f"    {'p-value':<10}" + " ".join(f"{p:>7.1e}" for p in result.p_values))
    print(
        f"  top decile: S > {sample.overflow_threshold:.3f} m, "
        f"{hsic_study()['overflow top decile'].target_fraction:.2f} of the runs"
    )

    print("\n6. The non-monotone effect of the dike height on the cost")
    table = measure_table("cost")
    morris = morris_study()["cost"]
    index = morris.input_names.index(HIGHLIGHT)
    print(
        f"  SRC^2 = {table.value('SRC^2', HIGHLIGHT):.4f}, Taylor eta^2 = "
        f"{table.value('Taylor eta^2', HIGHLIGHT):.3f}, Sobol S_i = "
        f"{table.value('Sobol S_i', HIGHLIGHT):.3f}, R2_HSIC = "
        f"{table.value('R2_HSIC', HIGHLIGHT):.3f}; Morris mu = "
        f"{morris.mu[index]:.4f} against mu* = {morris.mu_star[index]:.4f}"
    )
    print(
        "  A low dike lets the river overflow (damage cost up to 1 MEUR); a high "
        "dike costs max(Hd, 8) / 20 to maintain. The cost is lowest near "
        "Hd = 8 m, so a straight line through it is nearly flat."
    )

    print("\nFigures")
    _save("flood-measures-overflow", lambda: figure_measure_comparison("overflow"))
    _save("flood-measures-cost", lambda: figure_measure_comparison("cost"))
    _save("flood-morris", figure_morris)
    _save("flood-cobweb-overflow", figure_cobweb)
    _save("flood-top-decile-overflow", figure_top_decile_scatter)
    _save("flood-input-scatter-cost", lambda: figure_input_scatter("cost"))
    print(f"\nWall-clock time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
