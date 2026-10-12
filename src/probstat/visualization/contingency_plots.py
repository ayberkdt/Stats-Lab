"""Plots for contingency tables: residual heatmaps and 2 x 2 effect measures.

The functions take a table of counts or a
:class:`~probstat.inference.TwoByTwoMeasures` result and only present them.
Colour roles (ADR-0024, ADR-0052): adjusted residuals are a polarity, so the
heatmap uses a diverging scale built from two palette hues around a neutral
midpoint (:func:`diverging_colormap`): ``series[0]`` for cells holding more
counts than independence predicts and ``series[2]`` for fewer, with the
gridline token ``rule`` at zero. Effect estimates are data summaries in the
observed hue, and the no-effect values (0 for a difference, 1 for a ratio)
are accent references.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from probstat.inference import TwoByTwoMeasures, chi_square_sf, contingency_residuals

from ._layout import halo, wrapped_title_block
from .theme import (
    ANNOTATION_SIZE,
    ChartPalette,
    active_palette,
    axis_labels,
    points,
    pyplot,
    reference_line,
    style_axes,
    themed,
)


def diverging_ramp(palette: ChartPalette | None = None) -> tuple[str, str, str]:
    """Return the three anchors of the diverging scale: negative, zero, positive.

    Two categorical hues that pass the palette's adjacent-pair colour-vision
    check (``series[2]`` violet and ``series[0]`` green, as in the classic
    purple-green scheme) sit at the ends, and the neutral ``rule`` grey is
    the midpoint, so a hue never marks "no departure". Every channel moves
    monotonically from the midpoint to each end in both palettes, so
    lightness changes monotonically along each arm.

    Args:
        palette: Palette to read (active by default).

    Returns:
        ``(negative, midpoint, positive)`` hex colours.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    tokens = palette or active_palette()
    return tokens.series[2], tokens.rule, tokens.series[0]


def diverging_colormap(palette: ChartPalette | None = None) -> Any:
    """Return a two-hue diverging colormap with a neutral midpoint.

    Args:
        palette: Palette whose hues are used (active by default).

    Returns:
        Matplotlib ``LinearSegmentedColormap`` from :func:`diverging_ramp`.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "probstat-diverging", list(diverging_ramp(palette))
    )


def _relative_luminance(colour: Sequence[float]) -> float:
    """Return the WCAG relative luminance of an RGB triple in ``[0, 1]``.

    References:
        [REF-MATPLOTLIB]. No registered equation (display colour arithmetic).

    """
    channels = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in colour[:3]
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _text_colour(fill: Sequence[float], palette: ChartPalette) -> str:
    """Return ``ink`` or ``canvas``, whichever contrasts more with ``fill``.

    References:
        [REF-MATPLOTLIB]. No registered equation (display colour arithmetic).

    """
    from matplotlib.colors import to_rgb

    background = _relative_luminance(fill)

    def contrast(token: str) -> float:
        """Return the contrast ratio of a text token against the fill.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        other = _relative_luminance(to_rgb(token))
        high, low = max(background, other), min(background, other)
        return (high + 0.05) / (low + 0.05)

    return max((palette.ink, palette.canvas), key=contrast)


def _labels(given: Sequence[str] | None, count: int, prefix: str) -> list[str]:
    """Return category labels, checking their number.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if given is None:
        return [f"{prefix} {k + 1}" for k in range(count)]
    if len(given) != count:
        raise ValueError(f"{prefix.lower()} labels must match the table")
    return [str(label) for label in given]


@themed
def plot_adjusted_residuals(
    table: Iterable[Iterable[object]],
    *,
    row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
    row_name: str = "Row",
    column_name: str = "Column",
    threshold: float = 2.0,
    ax: Any | None = None,
) -> Any:
    """Show where an ``r x c`` table departs from independence.

    Each cell is coloured by Haberman's adjusted residual
    ``(O - E)/sqrt(E (1 - R_i/n)(1 - C_j/n))``, approximately standard normal
    under independence, on a symmetric diverging scale
    (:func:`diverging_colormap`). Cells with ``|residual| > threshold`` carry
    their value; the subtitle quotes Pearson's ``X^2`` on ``(r-1)(c-1)``
    degrees of freedom. The residuals are cell-wise and not adjusted for
    multiplicity: with many cells one beyond 2 is expected by chance.

    Args:
        table: Row-wise counts, at least 2 x 2, no empty row or column.
        row_labels: Optional row category labels.
        column_labels: Optional column category labels.
        row_name: Name of the row variable.
        column_name: Name of the column variable.
        threshold: Absolute residual above which a cell is labelled (> 0).
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the table is invalid, labels do not match, or the
            threshold is not positive.

    Equation:
        [EQ-INF-CT-003], [EQ-INF-CHI2-IND-001], [EQ-INF-DIST-CHI2-001].

    References:
        [REF-HABERMAN-1973], [REF-AGRESTI-2013, Chapters 2-3],
        [REF-MATPLOTLIB].

    """
    if not (math.isfinite(threshold) and threshold > 0.0):
        raise ValueError("threshold must be positive and finite")
    counts = [list(row) for row in table]
    residuals = contingency_residuals(counts)
    adjusted = residuals.adjusted
    rows, cols = len(adjusted), len(adjusted[0])
    rows_text = _labels(row_labels, rows, "Row")
    cols_text = _labels(column_labels, cols, "Column")
    statistic = math.fsum(v * v for row in residuals.pearson for v in row)
    df = (rows - 1) * (cols - 1)
    p_value = chi_square_sf(statistic, float(df))
    total = math.fsum(math.fsum(row) for row in residuals.expected)
    palette = active_palette()
    axes = (
        ax
        if ax is not None
        else pyplot().subplots(figsize=(1.05 * cols + 3.4, 0.62 * rows + 2.6))[1]
    )
    style_axes(axes)
    largest = max(abs(v) for row in adjusted for v in row)
    limit = max(3.0, math.ceil(largest))
    colormap = diverging_colormap(palette)
    image = axes.imshow(
        [list(row) for row in adjusted], cmap=colormap, vmin=-limit, vmax=limit,
        aspect="auto", interpolation="nearest",
    )  # fmt: skip
    axes.grid(False)
    axes.set_xticks([j - 0.5 for j in range(1, cols)], minor=True)
    axes.set_yticks([i - 0.5 for i in range(1, rows)], minor=True)
    axes.grid(which="minor", color=palette.canvas, linewidth=2.0)
    axes.tick_params(which="minor", length=0)
    axes.set_xticks(range(cols))
    axes.set_xticklabels(cols_text)
    axes.set_yticks(range(rows))
    axes.set_yticklabels(rows_text)
    axes.tick_params(length=0)
    for side in ("left", "bottom"):
        axes.spines[side].set_visible(False)
    flagged = 0
    for i, row in enumerate(adjusted):
        for j, value in enumerate(row):
            if abs(value) <= threshold:
                continue
            flagged += 1
            fill = colormap((value + limit) / (2.0 * limit))
            axes.text(
                j, i, f"{value:+.1f}", ha="center", va="center",
                fontsize=ANNOTATION_SIZE + 0.5, fontweight="bold",
                color=_text_colour(fill, palette),
            )  # fmt: skip
    bar = axes.figure.colorbar(image, ax=axes, fraction=0.05, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=palette.muted, labelcolor=palette.ink, length=0)
    bar.set_label(
        "Adjusted residual (+ more, - fewer than expected)", color=palette.ink
    )
    axis_labels(axes, column_name, row_name)
    i, j, value = residuals.largest_adjusted()
    if flagged == 0:
        title = (
            f"No cell departs from independence beyond ±{threshold:g}: the "
            f"largest adjusted residual is {value:+.1f}"
        )
    else:
        direction = "more" if value > 0.0 else "fewer"
        title = (
            f"{rows_text[i]} / {cols_text[j]} holds {direction} counts than "
            f"independence predicts (adjusted residual {value:+.1f}); "
            f"{flagged} of {rows * cols} cells lie beyond ±{threshold:g}"
        )
    wrapped_title_block(
        axes,
        title,
        [
            "cells: Haberman adjusted residual (O - E) / SE",
            f"labelled: |residual| > {threshold:g}",
            f"X² = {statistic:.1f} on {df} df, p = {p_value:.2g}",
            f"n = {total:,.0f}",
        ],
        width_inches=axes.figure.get_figwidth(),
    )
    return axes


def _ratio_text(value: float) -> str:
    """Format a ratio estimate or limit for a label.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    return f"{value:.3g}" if math.isfinite(value) else "inf"


def _agreement(names: Sequence[str]) -> str:
    """Return ``"a"``, ``"a and b"``, or ``"a, b and c"``.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


@themed
def plot_effect_measures(
    measures: TwoByTwoMeasures,
    *,
    groups: tuple[str, str] = ("exposed", "unexposed"),
    event: str = "the event",
) -> Any:
    """Plot the risk difference, risk ratio, and odds ratio with intervals.

    Two panels, each on its natural scale: the risk difference (Newcombe
    interval) on a linear axis with the no-effect value 0, and the risk and
    odds ratios (Katz and Woolf intervals) on a log axis with the no-effect
    value 1, both as accent references. A risk ratio without a finite
    estimate or interval (a zero risk) is left out and named in the
    subtitle. The title says whether the measures agree on excluding no
    effect.

    Args:
        measures: Output of :func:`probstat.inference.two_by_two_measures`.
        groups: Names of the first and second row of the table.
        event: Name of the outcome in the first column.

    Returns:
        The figure (two panels).

    Equation:
        [EQ-INF-CT-001].

    References:
        [REF-NEWCOMBE-1998], [REF-KATZ-ETAL-1978], [REF-WOOLF-1955],
        [REF-AGRESTI-2013, Chapter 3], [REF-MATPLOTLIB].

    """
    plt = pyplot()
    palette = active_palette()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.4, 4.4),
        gridspec_kw={"height_ratios": (1.0, 2.0), "hspace": 0.8},
    )  # fmt: skip
    style_axes(top)
    style_axes(bottom)
    rd_interval = measures.risk_difference_interval
    entries: list[tuple[Any, str, float, float, float, float]] = [
        (top, "Risk difference", measures.risk_difference, rd_interval.lower,
         rd_interval.upper, 0.0),
    ]  # fmt: skip
    rr_interval = measures.risk_ratio_interval
    dropped = rr_interval is None or not math.isfinite(measures.risk_ratio)
    if rr_interval is not None and not dropped:
        entries.append(
            (bottom, "Risk ratio", measures.risk_ratio, rr_interval.lower,
             rr_interval.upper, 1.0)
        )  # fmt: skip
    or_interval = measures.odds_ratio_interval
    entries.append(
        (bottom, "Odds ratio", measures.odds_ratio, or_interval.lower,
         or_interval.upper, 1.0)
    )  # fmt: skip
    bottom_names = [name for axes, name, *_ in entries if axes is bottom]
    for axes, name, estimate, lower, upper, _ in entries:
        position = (
            0.0 if axes is top else len(bottom_names) - 1 - bottom_names.index(name)
        )
        axes.hlines(
            position, lower, upper, color=palette.observed, linewidth=2.2,
            alpha=0.55, zorder=2,
        )  # fmt: skip
        points(axes, [estimate], [position], color=palette.observed, size=52.0)
        axes.annotate(
            f"{_ratio_text(estimate)}  [{_ratio_text(lower)}, {_ratio_text(upper)}]",
            (upper, position), xytext=(8.0, 0.0), textcoords="offset points",
            ha="left", va="center", fontsize=ANNOTATION_SIZE, color=palette.ink,
            path_effects=halo(), zorder=5,
        )  # fmt: skip
    for axes, names in ((top, ["Risk difference"]), (bottom, bottom_names)):
        axes.set_yticks(range(len(names)))
        axes.set_yticklabels(list(reversed(names)))
        axes.set_ylim(-0.6, len(names) - 0.4)
        axes.grid(axis="y", visible=False)
        axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    rd_values = [measures.risk_difference, rd_interval.lower, rd_interval.upper, 0.0]
    low, high = min(rd_values), max(rd_values)
    span = high - low or 0.1
    top.set_xlim(low - 0.08 * span, high + 0.55 * span)
    reference_line(top, x=0.0, label="no effect")
    ratio_values = [
        value
        for axes, _, estimate, lower, upper, _ in entries
        if axes is bottom
        for value in (estimate, lower, upper, 1.0)
        if math.isfinite(value) and value > 0.0
    ]
    bottom.set_xscale("log")
    low_ratio, high_ratio = min(ratio_values), max(ratio_values)
    factor = high_ratio / low_ratio
    left, right = low_ratio / factor**0.08, high_ratio * factor**0.5
    bottom.set_xlim(left, right)
    exponents = range(math.floor(math.log10(left)), math.ceil(math.log10(right)) + 1)
    ticks: list[float] = []
    for mantissas in ((1.0, 2.0, 5.0), (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)):
        ticks = [
            mantissa * 10.0**exponent
            for exponent in exponents
            for mantissa in mantissas
            if left <= mantissa * 10.0**exponent <= right
        ]
        if len(ticks) >= 3:
            break
    bottom.set_xticks(ticks)
    bottom.set_xticks([], minor=True)
    bottom.set_xticklabels([f"{tick:g}" for tick in ticks])
    reference_line(bottom, x=1.0, label="no effect")
    axis_labels(top, "Risk difference", None)
    axis_labels(bottom, "Ratio (log scale)", None)
    excludes = [
        name for _, name, _, lower, upper, null in entries if not lower <= null <= upper
    ]
    includes = [name for _, name, *_ in entries if name not in excludes]
    level = measures.confidence
    short = {"Risk difference": "RD", "Risk ratio": "RR", "Odds ratio": "OR"}
    values = ", ".join(
        f"{short[name]} {estimate:+.3f}" if axes is top
        else f"{short[name]} {_ratio_text(estimate)}"
        for axes, name, estimate, *_ in entries
    )  # fmt: skip
    if not includes:
        word = "higher" if measures.risk_difference > 0.0 else "lower"
        title = (
            f"The {groups[0]} group has a {word} risk of {event} ({values}); "
            f"every {level:.0%} interval excludes no effect"
        )
    elif not excludes:
        title = (
            f"No measure separates {groups[0]} from {groups[1]}: every "
            f"{level:.0%} interval contains no effect ({values})"
        )
    else:
        title = (
            f"The measures disagree: the {_agreement([short[n] for n in excludes])} "
            f"interval{'s' if len(excludes) > 1 else ''} exclude"
            f"{'' if len(excludes) > 1 else 's'} no effect, the "
            f"{_agreement([short[n] for n in includes])} "
            f"interval{'s' if len(includes) > 1 else ''} do"
            f"{'' if len(includes) > 1 else 'es'} not"
        )
    parts = [
        f"risk of {event}: {measures.risk_exposed:.3f} ({groups[0]}) vs "
        f"{measures.risk_unexposed:.3f} ({groups[1]})",
        f"whiskers: {level:.0%} Newcombe (RD), Katz (RR), Woolf (OR) intervals",
    ]
    if measures.corrected:
        parts.append("OR on the table with 0.5 added to every cell")
    if dropped:
        parts.append("RR omitted: a zero risk gives no finite interval")
    wrapped_title_block(top, title, parts, width_inches=figure.get_figwidth())
    return figure
