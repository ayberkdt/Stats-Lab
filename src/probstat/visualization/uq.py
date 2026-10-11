"""Explanatory plots for uncertainty-quantification workflows.

The functions take plain numbers (or library result objects through thin
adapters) so they never decide mathematical truth; they only present it under
the shared theme rules.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from probstat.foundations.validation import coerce_finite_sample, require_finite_real
from probstat.statistics import quantile

from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    FITTED,
    INNER_BAND_ALPHA,
    PROJECTED,
    active_palette,
    axis_labels,
    band,
    direct_labels,
    line,
    new_axes,
    points,
    reference_line,
    themed,
    title_block,
)


@themed
def plot_method_comparison(
    entries: Sequence[tuple[str, float, float | None, float | None]],
    *,
    reference: float | None = None,
    reference_label: str = "reference",
    log_scale: bool = False,
    quantity_label: str = "Estimate",
    title: str | None = None,
    subtitle: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Compare point estimates with interval whiskers as a dot plot.

    Bars are avoided because a bar's length implies a quantity measured from
    zero; an estimate with an interval is a point with uncertainty.

    Args:
        entries: ``(label, estimate, lower, upper)``; bounds may be ``None``.
        reference: Optional exact or reference value drawn in the accent.
        reference_label: Label for the reference line.
        log_scale: Use a logarithmic value axis (all values must be positive).
        quantity_label: Value-axis title.
        title: Assertion title; a neutral default is used when omitted.
        subtitle: Optional subtitle naming the marks.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``entries`` is empty or a value is not finite.

    References:
        [REF-MATPLOTLIB], [REF-MELCHERS-BECK-2018, Chapter 3].

    """
    if not entries:
        raise ValueError("entries must not be empty")
    axes = new_axes(ax, figsize=(7.2, 0.62 * len(entries) + 1.9))
    palette = active_palette()
    positions = tuple(range(len(entries)))[::-1]
    for position, (label, estimate, lower, upper) in zip(
        positions, entries, strict=True
    ):
        value = require_finite_real(estimate, name=f"{label} estimate")
        if lower is not None and upper is not None:
            axes.hlines(
                position,
                require_finite_real(lower, name=f"{label} lower"),
                require_finite_real(upper, name=f"{label} upper"),
                color=palette.observed,
                linewidth=2.2,
                alpha=0.55,
                zorder=2,
            )
        axes.scatter(
            [value],
            [position],
            s=52.0,
            color=palette.observed,
            edgecolors=palette.canvas,
            linewidths=1.3,
            zorder=4,
        )
        anchor = upper if upper is not None else value
        axes.annotate(
            f"{value:.4g}",
            (anchor, position),
            xytext=(8.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    axes.set(yticks=positions, yticklabels=tuple(entry[0] for entry in entries))
    axes.set_ylim(-0.7, len(entries) - 0.2)
    if log_scale:
        axes.set_xscale("log")
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    if reference is not None:
        reference_line(axes, x=reference, label=reference_label)
    axis_labels(axes, quantity_label)
    title_block(
        axes,
        title or "Method estimates with their uncertainty",
        subtitle or "Dot: estimate. Whisker: reported interval",
    )
    return axes


@themed
def plot_sobol_indices(
    names: Sequence[str],
    first_order: Sequence[float],
    total_effect: Sequence[float],
    *,
    title: str | None = None,
    subtitle: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw first-order and total Sobol indices as a sorted dumbbell chart.

    The segment between the two dots is the share of output variance that an
    input causes only through interactions with other inputs.

    Args:
        names: Input names.
        first_order: First-order indices ``S_i``.
        total_effect: Total-effect indices ``S_Ti``.
        title: Assertion title; derived from the data when omitted.
        subtitle: Optional subtitle.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If the sequences differ in length or are empty.

    Equation:
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    References:
        [REF-SALTELLI-2010], [REF-MATPLOTLIB].

    """
    if not names or not len(names) == len(first_order) == len(total_effect):
        raise ValueError("names and indices must be non-empty and equally long")
    order = sorted(range(len(names)), key=lambda index: total_effect[index])
    axes = new_axes(ax, figsize=(7.2, 0.62 * len(names) + 2.0))
    palette = active_palette()
    first_colour, total_colour = palette.series[0], palette.series[1]
    for position, index in enumerate(order):
        s1, st = first_order[index], total_effect[index]
        axes.hlines(
            position,
            min(s1, st),
            max(s1, st),
            color=palette.rule,
            linewidth=5.0,
            zorder=1,
        )
        axes.scatter(
            [s1],
            [position],
            s=60.0,
            color=first_colour,
            edgecolors=palette.canvas,
            linewidths=1.3,
            zorder=3,
        )
        axes.scatter(
            [st],
            [position],
            s=60.0,
            color=total_colour,
            edgecolors=palette.canvas,
            linewidths=1.3,
            zorder=3,
        )
        axes.annotate(
            f"{st:.2f}",
            (st, position),
            xytext=(9.0, 0.0),
            textcoords="offset points",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    top = len(order) - 1
    axes.annotate(
        "first-order S_i",
        (first_order[order[top]], top),
        xytext=(0.0, 12.0),
        textcoords="offset points",
        ha="center",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=first_colour,
    )
    axes.annotate(
        "total S_Ti",
        (total_effect[order[top]], top),
        xytext=(0.0, 12.0),
        textcoords="offset points",
        ha="center",
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        color=total_colour,
    )
    axes.set(
        yticks=tuple(range(len(order))), yticklabels=tuple(names[i] for i in order)
    )
    axes.set_ylim(-0.6, len(order) - 0.1)
    upper = max(max(total_effect), max(first_order), 0.0)
    axes.set_xlim(min(0.0, min(first_order)) - 0.03, upper * 1.18 + 0.02)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    axis_labels(axes, "Share of output variance")
    gaps = [total_effect[i] - first_order[i] for i in range(len(names))]
    candidates = [i for i in range(len(names)) if total_effect[i] > 0.05]
    if title is None and candidates:
        key = max(candidates, key=lambda i: gaps[i] / total_effect[i])
        share = gaps[key] / total_effect[key]
        if share >= 0.8:
            title = (
                f"{names[key]} matters only through interactions "
                f"(S = {first_order[key]:.2f}, S_T = {total_effect[key]:.2f})"
            )
        else:
            title = f"Interactions add {gaps[key]:.2f} to the influence of {names[key]}"
    top_input = max(range(len(names)), key=lambda i: total_effect[i])
    title_block(
        axes,
        title or f"{names[top_input]} drives the most output variance",
        subtitle or "Grey segment: variance an input causes only jointly with others",
    )
    return axes


@themed
def plot_calibration_prediction(
    observed_x: Sequence[float],
    observed_y: Sequence[float],
    grid: Sequence[float],
    mean: Sequence[float],
    parameter_band: tuple[Sequence[float], Sequence[float]],
    predictive_band: tuple[Sequence[float], Sequence[float]],
    *,
    calibration_limit: float | None = None,
    x_label: str = "Input x",
    y_label: str = "Response y",
    confidence: float = 0.95,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Show a calibrated model, its two uncertainty layers, and the data.

    The inner band (parameter uncertainty) and the outer band (future
    observation) share the model hue; the measurements keep the data hue, and
    the calibration-domain boundary is an accent reference beyond which the
    prediction is an extrapolation (drawn as a projected dash).

    Args:
        observed_x: Calibration inputs.
        observed_y: Calibration measurements.
        grid: Prediction inputs (ascending).
        mean: Posterior mean response on ``grid``.
        parameter_band: Lower and upper bounds from parameter uncertainty.
        predictive_band: Lower and upper bounds for a future observation.
        calibration_limit: Largest calibrated input, if extrapolation is shown.
        x_label: Horizontal axis title.
        y_label: Vertical axis title.
        confidence: Nominal band level, used in labels.
        title: Assertion title.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-BAYES-PRED-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 3.3], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    model_colour = palette.model
    band(
        axes,
        grid,
        predictive_band[0],
        predictive_band[1],
        color=model_colour,
        alpha=BAND_ALPHA,
    )
    band(
        axes,
        grid,
        parameter_band[0],
        parameter_band[1],
        color=model_colour,
        alpha=INNER_BAND_ALPHA,
    )
    if calibration_limit is None:
        line(axes, grid, mean, color=model_colour)
    else:
        inside = [i for i, x in enumerate(grid) if x <= calibration_limit]
        outside = [i for i, x in enumerate(grid) if x >= calibration_limit]
        line(
            axes,
            [grid[i] for i in inside],
            [mean[i] for i in inside],
            color=model_colour,
        )
        line(
            axes,
            [grid[i] for i in outside],
            [mean[i] for i in outside],
            color=model_colour,
            dashes=PROJECTED,
        )
        reference_line(axes, x=calibration_limit, label="calibration data end")
    points(axes, observed_x, observed_y, color=palette.observed, size=40.0)
    axis_labels(axes, x_label, y_label)
    last = len(grid) - 1
    direct_labels(
        axes,
        (
            (
                grid[last],
                predictive_band[1][last],
                f"{confidence:.0%} future observation",
                model_colour,
            ),
            (grid[last], mean[last], "posterior mean", model_colour),
            (
                grid[last],
                parameter_band[0][last],
                f"{confidence:.0%} parameter band",
                model_colour,
            ),
        ),
    )
    title_block(
        axes,
        title or "Prediction uncertainty widens beyond the calibration data",
        "Dots: measurements. Inner band: parameters. "
        "Outer band: adds measurement noise",
    )
    return axes


@themed
def plot_output_distribution(
    values: Iterable[float],
    *,
    name: str = "Y",
    unit: str | None = None,
    threshold: float | None = None,
    probabilities: Sequence[float] = (0.05, 0.5, 0.95),
    bins: int | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot a propagated output distribution with quantiles and a threshold.

    Args:
        values: Propagated output sample.
        name: Output name.
        unit: Optional unit.
        threshold: Optional limit; its exceedance fraction is annotated.
        probabilities: Quantile levels drawn as muted ticks.
        bins: Histogram bin count (Freedman-Diaconis-like default).
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-STAT-QUANT-001], [EQ-PROP-THRESH-001].

    References:
        [REF-HYNDMAN-FAN-1996], [REF-MATPLOTLIB].

    """
    sample = tuple(sorted(coerce_finite_sample(values, name="values")))
    if len(sample) < 2:
        raise ValueError("at least two values are required")
    axes = new_axes(ax)
    palette = active_palette()
    count = bins or max(12, min(60, int(2.0 * len(sample) ** (1.0 / 3.0))))
    axes.hist(
        sample,
        bins=count,
        color=palette.observed,
        alpha=0.85,
        edgecolor=palette.canvas,
        linewidth=0.8,
        density=True,
        zorder=2,
    )
    top = axes.get_ylim()[1]
    for p in probabilities:
        q = quantile(sample, p)
        axes.vlines(
            q, 0.0, top * 0.92, color=palette.ink, linewidth=0.9, alpha=0.6, zorder=3
        )
        axes.annotate(
            f"q{p:g} = {q:.3g}",
            (q, top * 0.92),
            xytext=(3.0, 2.0),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            color=palette.muted,
            rotation=0,
            ha="left",
            va="bottom",
        )
    axes.set_ylim(0.0, top * 1.08)
    subtitle = f"{len(sample)} propagated samples; ticks mark the stated quantiles"
    headline = (
        f"{name}: median {quantile(sample, 0.5):.3g}, 90% range "
        f"[{quantile(sample, 0.05):.3g}, {quantile(sample, 0.95):.3g}]"
    )
    if threshold is not None:
        limit = require_finite_real(threshold, name="threshold")
        exceed = sum(value > limit for value in sample) / len(sample)
        reference_line(axes, x=limit, label=f"limit: P(exceed) = {exceed:.3g}")
    axis_labels(axes, name, "Density", x_unit=unit)
    title_block(axes, headline, subtitle)
    return axes


@themed
def plot_design_comparison(
    summary: Mapping[str, tuple[float, float]],
    *,
    size: int | None = None,
    ax: Any | None = None,
) -> Any:
    """Compare sampling designs by the spread of replicate mean estimates.

    Args:
        summary: ``design -> (mean of means, std of means)`` as returned by
            :func:`probstat.propagation.compare_designs`.
        size: Points per replicate, used in the subtitle.
        ax: Optional axes.

    Returns:
        Populated axes with a logarithmic error axis.

    Equation:
        [EQ-SAMP-RQMC-001].

    References:
        [REF-OWEN-2013, Section 17.2], [REF-MATPLOTLIB].

    """
    if not summary:
        raise ValueError("summary must not be empty")
    names = {
        "mc": "Monte Carlo",
        "lhs": "Latin hypercube",
        "halton": "Halton (RQMC)",
        "sobol": "Sobol (RQMC)",
    }
    ordered = sorted(summary.items(), key=lambda item: item[1][1], reverse=True)
    axes = new_axes(ax, figsize=(7.2, 0.62 * len(ordered) + 1.9))
    palette = active_palette()
    positions = tuple(range(len(ordered)))[::-1]
    baseline = summary.get("mc", ordered[0][1])[1]
    for position, (design, (_, spread)) in zip(positions, ordered, strict=True):
        colour = palette.muted if design == "mc" else palette.observed
        axes.hlines(
            position, spread, baseline, color=colour, linewidth=1.6, alpha=0.6, zorder=1
        )
        axes.scatter(
            [spread],
            [position],
            s=56.0,
            color=colour,
            edgecolors=palette.canvas,
            linewidths=1.3,
            zorder=3,
        )
        gain = baseline / spread if spread > 0.0 else math.inf
        axes.annotate(
            f"{spread:.3g}" + ("" if design == "mc" else f"  ({gain:.0f}x smaller)"),
            (spread, position),
            xytext=(-9.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
        )
    axes.set_xscale("log")
    low = min(spread for _, (_, spread) in ordered)
    axes.set_xlim(low / 12.0, baseline * 2.0)
    axes.set(yticks=positions, yticklabels=tuple(names.get(d, d) for d, _ in ordered))
    axes.set_ylim(-0.7, len(ordered) - 0.3)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", visible=True, color=palette.rule, linewidth=0.7)
    axis_labels(axes, "Standard deviation of the mean estimate (log scale)")
    best_name, (_, best_spread) = ordered[-1]
    title_block(
        axes,
        f"{names.get(best_name, best_name)} cuts the estimation error "
        f"{baseline / best_spread:.0f}x versus Monte Carlo",
        "Spread of replicate means"
        + ("" if size is None else f" at {size} points each")
        + "; smaller is better",
    )
    return axes


@themed
def plot_exceedance_curves(
    curves: Mapping[str, Sequence[tuple[float, float]]],
    *,
    threshold: float | None = None,
    x_label: str = "Level y",
    unit: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot one or more exceedance curves ``P(Y > y)`` on a log axis.

    Args:
        curves: Label to ``(level, exceedance)`` pairs; the first curve uses
            the data hue, the second the model hue, then further slots.
        threshold: Optional limit drawn in the accent.
        x_label: Level-axis title.
        unit: Optional unit.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-PROP-THRESH-001].

    References:
        [REF-SMITH-UQ-2024, Chapter 4], [REF-MATPLOTLIB].

    """
    if not curves:
        raise ValueError("curves must not be empty")
    axes = new_axes(ax)
    palette = active_palette()
    labels: list[tuple[float, float, str, str]] = []
    floor = 1.0
    for index, (label, pairs) in enumerate(curves.items()):
        colour = palette.series[index % len(palette.series)]
        kept = [(x, p) for x, p in pairs if p > 0.0]
        if not kept:
            continue
        xs, ps = zip(*kept, strict=True)
        floor = min(floor, min(ps))
        axes.step(
            xs,
            ps,
            where="post",
            color=colour,
            linewidth=2.0,
            linestyle="-" if index == 0 else FITTED,
        )
        labels.append((xs[-1], ps[-1], label, colour))
    axes.set_yscale("log")
    axes.set_ylim(floor / 2.0, 1.2)
    if threshold is not None:
        reference_line(axes, x=threshold, label="limit")
    axis_labels(axes, x_label, "P(Y > y)", x_unit=unit)
    direct_labels(axes, labels)
    title_block(
        axes,
        "Exceedance probability falls steeply in the tail",
        "Empirical complementary CDF on a logarithmic probability axis",
    )
    return axes


def credible_band(
    samples: Sequence[Sequence[float]], level: float = 0.95
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return pointwise median and equal-tailed band from sample curves.

    Args:
        samples: Sample curves of equal length (one curve per draw).
        level: Central band probability.

    Returns:
        Lower, median, and upper pointwise curves.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 2.3].

    """
    columns = tuple(zip(*samples, strict=True))
    tail = 0.5 * (1.0 - level)
    lower = tuple(quantile(column, tail) for column in columns)
    middle = tuple(quantile(column, 0.5) for column in columns)
    upper = tuple(quantile(column, 1.0 - tail) for column in columns)
    return lower, middle, upper


__all__ = [
    "credible_band",
    "plot_calibration_prediction",
    "plot_design_comparison",
    "plot_exceedance_curves",
    "plot_method_comparison",
    "plot_output_distribution",
    "plot_sobol_indices",
]
