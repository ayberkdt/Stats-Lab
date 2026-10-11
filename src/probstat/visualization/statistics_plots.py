"""Plots for regression diagnostics, time series, and survival curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .theme import (
    PROJECTED,
    active_palette,
    axis_labels,
    band,
    direct_labels,
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
        axes, future, [history[-1], *lower], [history[-1], *upper], color=palette.model
    )
    line(axes, future, [history[-1], *mean], color=palette.model, dashes=PROJECTED)
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
        "Solid: observed series. Dotted line and band: forecast",
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
        below = [t for t, s in zip(times, survival, strict=True) if s <= 0.5]
        medians.append(
            f"{label} median {below[0]:.3g}"
            if below
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
