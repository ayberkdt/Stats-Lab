"""Graphical sensitivity analysis on a sample of model runs.

Two displays read an existing input-output sample before any index is
computed: scatterplots of the output against each input (with a lowess
smoother) reveal first-order trends, their shape, and inputs with no visible
effect; a cobweb (parallel-coordinate) plot draws every run as a polyline
across the inputs and highlights the runs whose output exceeds a threshold,
so a region of the input space that leads to the critical event, or a
combination of inputs (an interaction), shows as a bundle of coloured lines.
A tornado chart draws the one-at-a-time swings of a
:class:`~probstat.sensitivity.TornadoResult` as horizontal bars sorted by
absolute swing around the baseline output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.foundations.validation import require_finite_real
from probstat.inference.smoothing import loess
from probstat.sensitivity import TornadoResult
from probstat.statistics import quantile, ranks

from ._layout import halo
from .accessibility import legible_colour
from .theme import (
    ANNOTATION_SIZE,
    PATH_WIDTH,
    active_palette,
    axis_labels,
    entity_colour,
    line,
    pyplot,
    reference_line,
    style_axes,
    themed,
    title_block,
)


def _columns(
    samples: Sequence[Sequence[float]], names: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    """Return validated input columns of a rectangular sample.

    Raises:
        ValueError: For an empty or ragged sample, non-finite values, or a
            name count that does not match the columns.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    rows = tuple(
        tuple(
            require_finite_real(v, name=f"samples[{k}][{j}]") for j, v in enumerate(r)
        )
        for k, r in enumerate(samples)
    )
    if len(rows) < 2:
        raise ValueError("samples need at least two rows")
    dimension = len(rows[0])
    if dimension == 0 or any(len(r) != dimension for r in rows):
        raise ValueError("samples must be a rectangular matrix with columns")
    if len(names) != dimension:
        raise ValueError("names must match the number of input columns")
    return tuple(tuple(row[i] for row in rows) for i in range(dimension))


def _scaled(values: Sequence[float], scale: str) -> list[float]:
    """Map one column to ``[0, 1]`` by scaled rank or by min-max.

    References:
        [EQ-STAT-RANK-001], [REF-INSELBERG-1985].

    """
    n = len(values)
    if scale == "rank":
        return [(r - 1.0) / (n - 1) for r in ranks(values)]
    low, high = min(values), max(values)
    if high == low:
        return [0.5] * n
    return [(v - low) / (high - low) for v in values]


@themed
def plot_cobweb(
    samples: Sequence[Sequence[float]],
    names: Sequence[str],
    *,
    output: Sequence[float] | None = None,
    threshold: float | None = None,
    output_name: str = "Y",
    scale: str = "rank",
    title: str | None = None,
    subtitle: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw a cobweb (parallel-coordinate) plot with exceeding runs highlighted.

    Every run is a polyline through one vertical axis per input (and a last
    axis for the output when it is given). Each column is mapped to
    ``[0, 1]``: by scaled rank ``(r - 1)/(n - 1)`` (default, robust to skewed
    inputs) or by min-max. Runs with ``output > threshold`` are drawn in the
    "failed samples" colour on top of the others in grey; without a
    threshold the top 10 % of the output is highlighted. Bundles of coloured
    lines that cross the same end of an axis name the inputs that drive the
    exceedance; crossing patterns between neighbouring axes hint at
    interactions.

    Args:
        samples: ``n`` rows of ``d`` input values.
        names: Input names.
        output: Optional ``n`` output values.
        threshold: Exceedance threshold ``c`` of the output (default its
            0.9-quantile).
        output_name: Label of the output axis.
        scale: ``"rank"`` or ``"minmax"``.
        title: Assertion title; derived from the highlighted runs when
            omitted.
        subtitle: Optional subtitle.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: For an invalid sample, output, scale, or a threshold that
            no run (or every run) exceeds.

    Equation:
        [EQ-STAT-RANK-001].

    References:
        [REF-INSELBERG-1985], [REF-HELTON-ETAL-2006], [REF-MATPLOTLIB].

    """
    if scale not in ("rank", "minmax"):
        raise ValueError("scale must be 'rank' or 'minmax'")
    columns = _columns(samples, names)
    n = len(columns[0])
    labels = list(names)
    scaled = [_scaled(column, scale) for column in columns]
    flagged = [False] * n
    cut: float | None = None
    if output is not None:
        values = tuple(
            require_finite_real(v, name=f"output[{k}]") for k, v in enumerate(output)
        )
        if len(values) != n:
            raise ValueError("output must have one value per sample row")
        cut = (
            quantile(values, 0.9)
            if threshold is None
            else require_finite_real(threshold, name="threshold")
        )
        flagged = [v > cut for v in values]
        count = sum(flagged)
        if count == 0 or count == n:
            raise ValueError("the threshold must split the runs into two groups")
        scaled.append(_scaled(values, scale))
        labels.append(output_name)
    elif threshold is not None:
        raise ValueError("threshold needs the output values")
    width = max(7.2, 0.95 * len(labels) + 1.6)
    axes = ax if ax is not None else pyplot().subplots(figsize=(width, 4.4))[1]
    style_axes(axes)
    palette = active_palette()
    positions = list(range(len(labels)))
    highlight = entity_colour("failed samples")
    context = palette.muted
    has_groups = output is not None
    from matplotlib.collections import LineCollection

    def bundle(rows: Sequence[int], colour: str, alpha: float, zorder: float) -> None:
        """Draw the polylines of the given runs as one collection.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        if not rows:
            return
        segments = [
            list(zip(positions, (column[k] for column in scaled), strict=True))
            for k in rows
        ]
        axes.add_collection(
            LineCollection(
                segments,
                colors=colour,
                linewidths=PATH_WIDTH,
                alpha=alpha,
                zorder=zorder,
            )
        )

    # Opacity falls with the number of lines so that dense bundles read as
    # density instead of a solid block; the highlighted runs stay opaque.
    others = [k for k in range(n) if not flagged[k]]
    chosen = [k for k in range(n) if flagged[k]]
    bundle(
        others,
        context if has_groups else palette.observed,
        max(0.05, min(0.3, 45.0 / len(others))),
        2.0,
    )
    if chosen:
        bundle(chosen, highlight, max(0.4, min(0.9, 60.0 / len(chosen))), 3.0)
    for position in positions:
        axes.axvline(position, color=palette.ink, linewidth=0.8, alpha=0.55, zorder=1)
    if cut is not None:
        # The threshold on the output axis, in the scaled coordinate.
        output_column = scaled[-1]
        boundary = max(
            (
                value
                for value, mark in zip(output_column, flagged, strict=True)
                if not mark
            ),
            default=0.0,
        )
        axes.plot(
            [positions[-1] - 0.12, positions[-1] + 0.12],
            [boundary, boundary],
            color=palette.accent,
            linewidth=2.0,
            zorder=5,
        )
        axes.annotate(
            "threshold",
            (positions[-1] + 0.12, boundary),
            xytext=(4.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=legible_colour(palette.accent, palette.canvas, palette.ink),
            annotation_clip=False,
            zorder=6,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlim(-0.25, positions[-1] + 0.25)
    axes.set_ylim(-0.04, 1.04)
    axes.set_yticks((0.0, 0.5, 1.0))
    axes.set_yticklabels(
        ("min", "median", "max") if scale == "rank" else ("0", "0.5", "1")
    )
    axes.grid(axis="y", visible=False)
    axis_labels(
        axes,
        None,
        "Scaled rank of each column" if scale == "rank" else "Min-max scaled value",
    )
    if has_groups:
        count = sum(flagged)
        derived = _cobweb_title(scaled[:-1], names, flagged, output_name, cut)
        title_block(
            axes,
            title or derived,
            subtitle
            or (
                f"Each line is one run. Coloured: {count} of {n} runs with "
                f"{output_name} > {cut:.3g}; grey: the others"
            ),
        )
    else:
        title_block(
            axes,
            title or f"Each line is one of {n} runs across {len(labels)} inputs",
            subtitle
            or "Columns mapped to [0, 1]; crossings between axes hint at interactions",
        )
    return axes


def _cobweb_title(
    scaled: Sequence[Sequence[float]],
    names: Sequence[str],
    flagged: Sequence[bool],
    output_name: str,
    cut: float | None,
) -> str:
    """State which input ends the highlighted runs concentrate on.

    The median scaled position of the highlighted runs on each input axis is
    compared with the overall median ``0.5``; the two largest departures
    above ``0.15`` are named.

    References:
        [REF-HELTON-ETAL-2006]. No new equation is introduced.

    """
    shifts: list[tuple[float, str]] = []
    for column, name in zip(scaled, names, strict=True):
        chosen = [value for value, mark in zip(column, flagged, strict=True) if mark]
        shift = quantile(chosen, 0.5) - 0.5
        if abs(shift) > 0.15:
            shifts.append((shift, name))
    shifts.sort(key=lambda item: -abs(item[0]))
    event = f"{output_name} > {cut:.3g}" if cut is not None else "Exceedances"
    if not shifts:
        return f"Runs with {event} spread over every input range"
    words = [f"{'high' if shift > 0 else 'low'} {name}" for shift, name in shifts[:2]]
    return f"Runs with {event} need " + " and ".join(words)


@themed
def plot_input_scatter(
    samples: Sequence[Sequence[float]],
    output: Sequence[float],
    names: Sequence[str],
    *,
    smoother: str | None = "lowess",
    output_name: str = "Y",
    units: Sequence[str] | None = None,
    output_unit: str | None = None,
    columns: int | None = None,
    span: float = 2.0 / 3.0,
    smoother_points: int = 400,
    title: str | None = None,
    subtitle: str | None = None,
) -> Any:
    """Draw the output against each input as a grid of scatterplots.

    Each panel shares the output axis and carries a lowess line (local
    linear fit with tricube weights on a fraction ``span`` of the points,
    no robustness iterations), which shows the shape of ``E[Y | X_i]``: a
    flat line is an input with no first-order effect, a steep or curved one
    a strong (possibly nonlinear) effect. Interactions are invisible in
    these marginal views; use a cobweb plot or total Sobol indices for them.
    For speed the smoother is fitted on at most ``smoother_points`` points
    taken at equal steps of the sorted input.

    Args:
        samples: ``n`` rows of ``d`` input values.
        output: ``n`` output values.
        names: Input names.
        smoother: ``"lowess"`` or ``None`` (points only).
        output_name: Output axis title.
        units: Optional unit per input (``""`` for none), written as
            ``"name [unit]"``.
        output_unit: Optional output unit.
        columns: Panels per row (default ``min(4, d)``).
        span: Lowess neighbourhood fraction in ``(0, 1]``.
        smoother_points: Maximum number of points used by each smoother.
        title: Assertion title; derived from the smoothers when omitted.
        subtitle: Optional subtitle.

    Returns:
        The Matplotlib figure; ``figure.axes`` holds the panels in input
        order (unused cells are removed).

    Raises:
        ValueError: For an invalid sample or output, an unknown smoother, or
            invalid ``columns``, ``span`` or ``smoother_points``.

    Equation:
        [EQ-SMOOTH-009].

    References:
        [REF-CLEVELAND-1979], [REF-HELTON-ETAL-2006], [REF-MATPLOTLIB].

    """
    if smoother not in ("lowess", None):
        raise ValueError("smoother must be 'lowess' or None")
    inputs = _columns(samples, names)
    values = tuple(
        require_finite_real(v, name=f"output[{k}]") for k, v in enumerate(output)
    )
    n, dimension = len(values), len(inputs)
    if len(inputs[0]) != n:
        raise ValueError("output must have one value per sample row")
    if units is not None and len(units) != dimension:
        raise ValueError("units must match the number of input columns")
    count = min(4, dimension) if columns is None else columns
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("columns must be a positive integer")
    if isinstance(smoother_points, bool) or not isinstance(smoother_points, int):
        raise ValueError("smoother_points must be an integer")
    if smoother_points < 10:
        raise ValueError("smoother_points must be at least 10")
    rows = math.ceil(dimension / count)
    from matplotlib.ticker import MaxNLocator

    plt = pyplot()
    figure, grid = plt.subplots(
        rows,
        count,
        figsize=(2.15 * count + 0.9, 2.05 * rows + 1.3),
        sharey=True,
        squeeze=False,
    )
    palette = active_palette()
    panels = [grid[r][c] for r in range(rows) for c in range(count)]
    for extra in panels[dimension:]:
        extra.remove()
    panels = panels[:dimension]
    smoothed_colour = legible_colour(
        entity_colour("smoothed"), palette.canvas, palette.ink
    )
    trends: list[tuple[float, float]] = []
    point_size = 5.0 if n > 300 else 12.0
    for index, (axes, column, name) in enumerate(
        zip(panels, inputs, names, strict=True)
    ):
        style_axes(axes)
        axes.scatter(
            column,
            values,
            s=point_size,
            color=palette.observed,
            alpha=0.35 if n > 300 else 0.6,
            linewidths=0.0,
            zorder=2,
        )
        if smoother == "lowess" and len(set(column)) > 2:
            order = sorted(range(n), key=lambda k: column[k])
            step = max(1, math.ceil(n / smoother_points))
            chosen = order[::step]
            xs = [column[k] for k in chosen]
            fit = loess(xs, [values[k] for k in chosen], span=span, degree=1)
            line(axes, xs, list(fit.fitted), entity="smoothed", zorder=4)
            trends.append(
                (fit.fitted[-1] - fit.fitted[0], max(fit.fitted) - min(fit.fitted))
            )
        else:
            trends.append((0.0, 0.0))
        axes.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis_labels(axes, name, x_unit=None if units is None else units[index] or None)
        if index % count == 0:
            axis_labels(axes, None, output_name, y_unit=output_unit)
            axes.tick_params(axis="y", labelleft=True)
    if smoother == "lowess":
        key = max(range(dimension), key=lambda i: trends[i][1])
        label_axes = panels[key]
        fitted_lines = [item for item in label_axes.get_lines()]
        if fitted_lines:
            x_end = float(fitted_lines[-1].get_xdata()[-1])
            y_end = float(fitted_lines[-1].get_ydata()[-1])
            label_axes.annotate(
                "lowess",
                (x_end, y_end),
                xytext=(-4.0, 10.0 if trends[key][0] >= 0 else -12.0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=ANNOTATION_SIZE,
                fontweight="bold",
                color=smoothed_colour,
                path_effects=halo(),
                zorder=6,
            )
    if title is None:
        if smoother == "lowess":
            key = max(range(dimension), key=lambda i: trends[i][1])
            spread = max(values) - min(values)
            flat = [names[i] for i in range(dimension) if trends[i][1] < 0.05 * spread]
            direction = "rises" if trends[key][0] >= 0 else "falls"
            title = f"{output_name} {direction} most clearly with {names[key]}"
            if flat:
                verb = "shows" if len(flat) == 1 else "show"
                title += f"; {len(flat)} of {dimension} inputs {verb} no visible trend"
        else:
            title = f"{output_name} against each of {dimension} inputs"
    title_block(
        panels[0],
        title,
        subtitle
        or (
            f"Dots: {n} model runs. Dashed line: lowess smoother (span {span:.2g})"
            if smoother == "lowess"
            else f"Dots: {n} model runs"
        ),
    )
    # Fixed margins in inches: the title block spans the whole figure above
    # the grid, and the shared output axis needs tick labels only on the left.
    width, height = (float(value) for value in figure.get_size_inches())
    figure.subplots_adjust(
        left=0.75 / width,
        right=1.0 - 0.2 / width,
        bottom=0.62 / height,
        top=1.0 - 0.95 / height,
        wspace=0.16,
        hspace=0.55,
    )
    return figure


@themed
def plot_tornado(
    result: TornadoResult,
    *,
    top: int | None = None,
    output_name: str | None = None,
    unit: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw one-at-a-time swings as horizontal bars around the baseline output.

    One row per input, the largest absolute swing on top. The horizontal axis
    is the change of the output from its baseline value, so every bar starts
    at zero and its length is the change it encodes; the baseline output
    itself is written on the dashed zero reference. Each row has two bars:
    the solid bar ends at the change with the input at its low value, the
    hatched bar at the change with the input at its high value, so colour is
    never the only carrier of direction. The input value at each end is
    written next to the bar end. An input whose baseline output lies outside
    its two end outputs is marked ``*`` (the response is non-monotone
    between the ends and the swing understates its effect).

    A tornado chart shows neither interactions between inputs nor dependence,
    and the bars say nothing about the response between their two ends.

    Args:
        result: Result of :func:`~probstat.sensitivity.tornado_analysis` or
            :func:`~probstat.sensitivity.tornado_function`.
        top: Show only the ``top`` largest swings.
        output_name: Axis title (default: the QoI name, or ``"Output"``).
        unit: Optional unit of the output.
        title: Assertion title; derived from the largest swing when omitted.
        subtitle: Optional subtitle.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        TypeError: If ``result`` is not a tornado result.
        ValueError: If ``top`` is not a positive integer.

    Equation:
        [EQ-SENS-TORNADO-001].

    References:
        [REF-HOWARD-1988], [REF-MATPLOTLIB].

    """
    if not isinstance(result, TornadoResult):
        raise TypeError("result must be a TornadoResult")
    if top is not None and (
        isinstance(top, bool) or not isinstance(top, int) or top < 1
    ):
        raise ValueError("top must be a positive integer")
    bars = result.bars if top is None else result.bars[:top]
    count = len(bars)
    palette = active_palette()
    size = (7.2, 0.5 * count + 2.2)
    axes = ax if ax is not None else pyplot().subplots(figsize=size)[1]
    style_axes(axes)
    base = result.baseline_output
    label = output_name or result.quantity or "Output"
    low_colour = palette.observed
    high_colour = palette.model
    ends = [0.0]
    for bar in bars:
        ends.extend((bar.low_output - base, bar.high_output - base))
    span = max(ends) - min(ends)
    pad = 0.16 * span if span > 0.0 else max(abs(base), 1.0) * 0.1
    for row, bar in enumerate(bars):
        y = count - 1 - row
        pieces = (
            (bar.low_output, bar.low_input, low_colour, None, "low"),
            (bar.high_output, bar.high_input, high_colour, "////", "high"),
        )
        for value, shown, colour, hatch, role in pieces:
            axes.barh(
                y,
                value - base,
                height=0.62,
                color=colour,
                hatch=hatch,
                edgecolor=palette.canvas,
                linewidth=0.8,
                zorder=3,
            )
            side = 1.0 if value > base else -1.0 if value < base else 0.0
            if side == 0.0:
                # A zero-width bar: put its label on the side the other bar
                # leaves free, left for the low end when both are empty.
                other = bar.high_output if role == "low" else bar.low_output
                side = (
                    -1.0 if (other > base or (other == base and role == "low")) else 1.0
                )
            axes.annotate(
                f"{shown:.3g}",
                (value - base, y),
                xytext=(5.0 * side, 0.0),
                textcoords="offset points",
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=ANNOTATION_SIZE,
                color=palette.ink,
                annotation_clip=False,
                zorder=6,
            )
    suffix = f" {unit}" if unit else ""
    reference_line(axes, x=0.0, label=f"baseline {base:.3g}{suffix}")
    axes.set_yticks(range(count))
    axes.set_yticklabels(
        [f"{bar.name}{' *' if bar.nonmonotone else ''}" for bar in reversed(bars)]
    )
    axes.set_ylim(-0.6, count - 0.4 + 0.5)
    axes.set_xlim(min(ends) - pad, max(ends) + pad)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    axis_labels(axes, f"Change in {label} from baseline", x_unit=unit)
    lead = bars[0] if bars else None
    if lead is None or lead.swing == 0.0:
        derived = f"No single input moves {label} between its low and high values"
    else:
        derived = (
            f"{lead.name} swings {label} the most: "
            f"{lead.swing:+.3g}{suffix} from low to high"
        )
    flagged = any(bar.nonmonotone for bar in bars)
    title_block(
        axes,
        title or derived,
        subtitle
        or (
            "Solid bar: input at its low value; hatched bar: at its high value"
            + ("; * non-monotone between the ends" if flagged else "")
        ),
    )
    return axes


__all__ = ["plot_cobweb", "plot_input_scatter", "plot_tornado"]
