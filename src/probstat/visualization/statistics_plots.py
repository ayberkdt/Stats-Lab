"""Plots for distributions, regression diagnostics, time series, and survival."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from probstat.statistics import gaussian_kde, quantile

from .theme import (
    MAX_BAND_ALPHA,
    PROJECTED,
    active_palette,
    axis_labels,
    band,
    direct_labels,
    gradient_area,
    line,
    new_axes,
    points,
    pyplot,
    reference_line,
    style_axes,
    themed,
    title_block,
)


@themed
def plot_correlogram(
    values: Sequence[float],
    band_half_width: float,
    *,
    kind: str = "ACF",
    ax: Any | None = None,
) -> Any:
    """Plot sample autocorrelations as lollipops with a white-noise band.

    Args:
        values: Correlations at lags ``0..k`` (ACF) or ``1..k`` (PACF).
        band_half_width: Half-width of the pointwise white-noise band.
        kind: ``"ACF"`` (lag 0 is skipped) or ``"PACF"``.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-TS-ACF-001].

    References:
        [REF-BROCKWELL-DAVIS-2016], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    start = 1
    series = list(values[1:]) if kind.upper() == "ACF" else list(values)
    lags = list(range(start, start + len(series)))
    axes.axhspan(
        -band_half_width,
        band_half_width,
        color=palette.muted,
        alpha=0.12,
        linewidth=0.0,
    )
    outside = [abs(v) > band_half_width for v in series]
    colours = [palette.observed if hit else palette.muted for hit in outside]
    axes.vlines(lags, 0.0, series, colors=colours, linewidth=1.6, zorder=2)
    axes.scatter(
        lags,
        series,
        s=30.0,
        color=colours,
        edgecolors=palette.canvas,
        linewidths=1.0,
        zorder=3,
    )
    axes.axhline(0.0, color=palette.ink, linewidth=0.8)
    from matplotlib.ticker import MaxNLocator

    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis_labels(axes, "Lag", kind.upper())
    count = sum(outside)
    title_block(
        axes,
        f"{count} of {len(series)} lags fall outside the white-noise band",
        f"Band: +/-{band_half_width:.3f} (95% pointwise under white noise)",
    )
    return axes


@themed
def plot_forecast(
    history: Sequence[float],
    mean: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    level: float = 0.95,
    name: str = "value",
    ax: Any | None = None,
) -> Any:
    """Plot an observed series followed by a forecast with its interval.

    Args:
        history: Observed series.
        mean: Forecast means for the following steps.
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        level: Interval level, used in labels.
        name: Series name.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-TS-FC-001].

    References:
        [REF-BROCKWELL-DAVIS-2016], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    n = len(history)
    past = list(range(1, n + 1))
    future = list(range(n, n + len(mean) + 1))
    line(axes, past, history, color=palette.observed, width=1.6)
    band(
        axes,
        future,
        [history[-1], *lower],
        [history[-1], *upper],
        color=palette.model,
        smooth=True,
    )
    line(
        axes,
        future,
        [history[-1], *mean],
        color=palette.model,
        dashes=PROJECTED,
        smooth=True,
    )
    axes.autoscale_view()
    gradient_area(axes, past, history, color=palette.observed)
    reference_line(axes, x=n + 0.5, label="forecast origin")
    axis_labels(axes, "Time index", name)
    direct_labels(
        axes,
        (
            (future[-1], upper[-1], f"{level:.0%} interval", palette.model),
            (future[-1], mean[-1], "forecast mean", palette.model),
        ),
    )
    width_first = upper[0] - lower[0]
    width_last = upper[-1] - lower[-1]
    title_block(
        axes,
        f"Forecast uncertainty grows {width_last / width_first:.1f}x over "
        f"{len(mean)} steps",
        "Solid: observed series. Dotted line and band: forecast, drawn with "
        "monotone cubic smoothing between steps",
    )
    return axes


@themed
def plot_survival_curves(
    curves: Mapping[
        str, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]
    ],
    *,
    test_label: str | None = None,
    time_label: str = "Time",
    ax: Any | None = None,
) -> Any:
    """Plot Kaplan-Meier step curves with their confidence bands.

    Args:
        curves: Group label to ``(times, survival, lower, upper)`` at event
            times (the curve starts at 1 at time 0).
        test_label: Optional test result shown in the subtitle.
        time_label: Horizontal axis title.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-SURV-KM-001].

    References:
        [REF-KAPLAN-MEIER-1958], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    labels: list[tuple[float, float, str, str]] = []
    medians: list[str] = []
    for index, (label, (times, survival, lower, upper)) in enumerate(curves.items()):
        colour = palette.series[index % len(palette.series)]
        xs = [0.0, *times]
        axes.fill_between(
            xs,
            [1.0, *lower],
            [1.0, *upper],
            step="post",
            color=colour,
            alpha=0.14,
            linewidth=0.0,
        )
        axes.step(xs, [1.0, *survival], where="post", color=colour, linewidth=2.0)
        labels.append((xs[-1], survival[-1], label, colour))
        median = _flat_aware_median(times, survival)
        medians.append(
            f"{label} median {median:.3g}"
            if median is not None
            else f"{label} median > {xs[-1]:.3g}"
        )
    axes.set_ylim(0.0, 1.05)
    reference_line(axes, y=0.5, label="50%")
    axis_labels(axes, time_label, "Survival probability")
    direct_labels(axes, labels)
    title_block(
        axes,
        "; ".join(medians),
        "Kaplan-Meier estimates with 95% log-log bands"
        + (f" · {test_label}" if test_label else ""),
    )
    return axes


def _flat_aware_median(
    times: Sequence[float], survival: Sequence[float]
) -> float | None:
    """Return the Kaplan-Meier median with the flat-segment midpoint rule.

    References:
        [EQ-SURV-MED-001]. No new equation is evaluated.

    """
    for index, (time, value) in enumerate(zip(times, survival, strict=True)):
        if value <= 0.5 + 1e-12:
            if abs(value - 0.5) <= 1e-12 and index + 1 < len(times):
                return 0.5 * (time + times[index + 1])
            return time
    return None


@themed
def plot_regression_diagnostics(
    fitted: Sequence[float],
    studentized: Sequence[float],
) -> Any:
    """Plot residuals against fitted values and a normal Q-Q panel.

    Args:
        fitted: Fitted values.
        studentized: Studentized residuals in the same order.

    Returns:
        The figure with two panels.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapter 4], [REF-MATPLOTLIB].

    """
    if len(fitted) != len(studentized) or len(fitted) < 3:
        raise ValueError("fitted and residuals must align with at least 3 points")
    plt = pyplot()
    palette = active_palette()
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for axes in (left, right):
        style_axes(axes)
    points(left, fitted, studentized, color=palette.observed, size=18.0)
    left.axhline(0.0, color=palette.ink, linewidth=0.8)
    for level in (-2.0, 2.0):
        left.axhline(
            level, color=palette.accent, linewidth=1.0, linestyle=(0, (3.5, 3.0))
        )
    flagged = sum(abs(r) > 2.0 for r in studentized)
    axis_labels(left, "Fitted value", "Studentized residual")
    ordered = sorted(studentized)
    n = len(ordered)
    theoretical = [NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)]
    points(right, theoretical, ordered, color=palette.observed, size=18.0)
    span = (min(theoretical), max(theoretical))
    right.plot(
        span, span, color=palette.accent, linewidth=1.0, linestyle=(0, (3.5, 3.0))
    )
    axis_labels(right, "Normal quantile", "Ordered residual")
    right.grid(axis="both", color=palette.rule, linewidth=0.6)
    title_block(
        left,
        f"{flagged} of {n} residuals exceed |2|; no structure means the mean "
        "model is adequate",
        "Left: residuals vs fitted. Right: normal Q-Q of studentized residuals",
    )
    figure.tight_layout()
    return figure


@themed
def plot_distribution_comparison(
    groups: Mapping[str, Sequence[float]],
    *,
    value_label: str = "Value",
    reference: float | None = None,
    reference_label: str = "reference",
    ax: Any | None = None,
) -> Any:
    """Compare group distributions as violins with an embedded box plot.

    Each violin is a mirrored Gaussian kernel density estimate in the group's
    identity hue; inside it a box spans the quartiles, a dot marks the median,
    whiskers reach the most extreme points within 1.5 IQR (Tukey), and points
    beyond the whiskers are drawn individually in muted ink.

    Args:
        groups: Group label to at least two finite observations (at most six
            groups, the categorical palette size).
        value_label: Vertical axis title.
        reference: Optional value drawn as an accent reference line.
        reference_label: Label of the reference line.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If there are no groups or more groups than palette slots.

    Equation:
        [EQ-STAT-KDE-001], [EQ-STAT-QUANT-001].

    References:
        [REF-HINTZE-NELSON-1998], [REF-TUKEY-1977],
        [REF-SILVERMAN-1986, Chapter 3], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    if not groups or len(groups) > len(palette.series):
        raise ValueError(
            f"supply between 1 and {len(palette.series)} groups; fold the rest "
            "into an 'other' group or use small multiples"
        )
    medians: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for index, (label, values) in enumerate(groups.items()):
        colour = palette.series[index]
        density = gaussian_kde(values)
        grid = density.grid(160, extend=3.0)
        heights = [density.pdf(x) for x in grid]
        scale = 0.36 / max(heights)  # full width 0.72 of the slot (sunumatik)
        axes.fill_betweenx(
            grid,
            [index - h * scale for h in heights],
            [index + h * scale for h in heights],
            color=colour,
            alpha=MAX_BAND_ALPHA,
            linewidth=0.0,
        )
        axes.plot(
            [index + h * scale for h in heights], grid, color=colour, linewidth=1.5
        )
        axes.plot(
            [index - h * scale for h in heights], grid, color=colour, linewidth=1.5
        )
        q1, q2, q3 = (quantile(values, level) for level in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        inside = [v for v in values if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
        axes.vlines(index, min(inside), max(inside), colors=palette.ink, linewidth=1.0)
        axes.add_patch(
            pyplot().Rectangle(
                (index - 0.06, q1),
                0.12,
                iqr,
                facecolor=palette.canvas,
                edgecolor=palette.ink,
                linewidth=1.0,
                zorder=3,
            )
        )
        axes.scatter(
            [index], [q2], s=28.0, color=colour, edgecolors=palette.ink, zorder=4
        )
        outliers = [v for v in values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
        if outliers:
            axes.scatter(
                [index] * len(outliers),
                outliers,
                s=12.0,
                color=palette.muted,
                zorder=3,
            )
        medians[label] = q2
        spreads[label] = iqr
    labels = list(groups)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels)
    axes.set_xlim(-0.6, len(labels) - 0.4)
    axes.grid(axis="x", visible=False)
    if reference is not None:
        reference_line(axes, y=reference, label=reference_label)
    axis_labels(axes, "", value_label)
    top = max(medians, key=lambda name: medians[name])
    widest = max(spreads, key=lambda name: spreads[name])
    narrowest = min(spreads, key=lambda name: spreads[name])
    if len(labels) == 1:
        title = f"{top}: median {medians[top]:.3g}, IQR {spreads[top]:.3g}"
    elif spreads[narrowest] > 0.0:
        ratio = spreads[widest] / spreads[narrowest]
        title = (
            f"{top} has the highest median ({medians[top]:.3g}); {widest} "
            f"spreads {ratio:.1f}x wider than {narrowest}"
        )
    else:
        title = f"{top} has the highest median ({medians[top]:.3g})"
    title_block(
        axes,
        title,
        "Violin: kernel density. Box: quartiles. Dot: median. Whiskers: 1.5 IQR",
    )
    return axes
