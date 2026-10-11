"""Plots for extreme value analysis: return levels, thresholds, tail fit.

The functions take result objects of :mod:`probstat.risk.extremes` (or plain
numbers) and only present them under the shared theme: empirical maxima use
the observed hue, fitted tail models the model hue, true or exact values the
accent, and extrapolation beyond the record is drawn with the projected dash.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.distributions import Beta
from probstat.foundations.validation import coerce_finite_sample
from probstat.inference import qq_points
from probstat.risk.extremes import GPDFit, MeanResidualLifePoint, ReturnLevel

from .theme import (
    FITTED,
    PROJECTED,
    REFERENCE,
    REFERENCE_WIDTH,
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


def _plain_log_axis(axis: Any) -> None:
    """Label a logarithmic axis with plain numbers instead of powers of ten.

    Args:
        axis: Matplotlib ``xaxis`` or ``yaxis``.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def hazen_return_periods(count: int) -> tuple[float, ...]:
    """Return empirical return periods ``1/(1 - p_i)`` of sorted maxima.

    The ``i``-th smallest of ``count`` block maxima gets the Hazen plotting
    position ``p_i = (i - 1/2)/count`` (the positions used by
    :func:`probstat.inference.qq_points`), so its empirical return period is
    ``1/(1 - p_i)`` blocks.

    Args:
        count: Number of maxima (positive).

    Returns:
        Return periods in ascending order.

    Raises:
        ValueError: If ``count`` is not positive.

    Equation:
        [EQ-INF-GOF-QQ-001], [EQ-RISK-RP-001].

    References:
        [REF-COLES-2001, Section 3.3], [REF-WILK-GNANADESIKAN-1968].

    """
    if count < 1:
        raise ValueError("count must be positive")
    return tuple(1.0 / (1.0 - (i - 0.5) / count) for i in range(1, count + 1))


@themed
def plot_return_levels(
    levels: Sequence[ReturnLevel],
    observed: Sequence[float],
    *,
    truth: Sequence[float] | None = None,
    headline_period: float | None = None,
    model_label: str = "fitted GEV",
    period_unit: str = "year",
    name: str = "Return level",
    ax: Any | None = None,
) -> Any:
    """Plot a return-level curve with its interval over empirical maxima.

    The horizontal axis is the return period on a logarithmic scale. The
    fitted curve is dashed inside the record (``FITTED``) and dotted beyond
    it (``PROJECTED``); the interval band takes the model hue. Empirical
    maxima sit at Hazen return periods. Interval endpoints that are infinite
    (an open profile interval) are left out of the band.

    Args:
        levels: Return levels in increasing return period, e.g. from
            ``GEVFit.return_level`` or ``GEVFit.profile_return_level``.
        observed: Block maxima (one per ``period_unit``).
        truth: Optional true return levels aligned with ``levels``.
        headline_period: Period quoted in the title (default: the longest
            period not exceeding ten record lengths, else the longest).
        model_label: Direct label of the fitted curve.
        period_unit: Unit of one block, used in labels.
        name: Vertical-axis title.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``levels`` has fewer than two entries, is not sorted,
            ``truth`` is misaligned, or ``observed`` is empty.

    Equation:
        [EQ-EVA-RL-001], [EQ-EVA-RL-002], [EQ-EVA-PROF-001],
        [EQ-INF-GOF-QQ-001].

    References:
        [REF-COLES-2001, Sections 3.3 and 4.3], [REF-MATPLOTLIB].

    """
    if len(levels) < 2:
        raise ValueError("at least two return levels are required")
    periods = [entry.return_period for entry in levels]
    if any(b <= a for a, b in zip(periods, periods[1:], strict=False)):
        raise ValueError("levels must be sorted by increasing return period")
    if truth is not None and len(truth) != len(levels):
        raise ValueError("truth must align with levels")
    maxima = sorted(coerce_finite_sample(observed, name="observed"))
    record = len(maxima)
    axes = new_axes(ax)
    palette = active_palette()
    finite = [
        entry
        for entry in levels
        if math.isfinite(entry.lower) and math.isfinite(entry.upper)
    ]
    if len(finite) >= 2:
        band(
            axes,
            [entry.return_period for entry in finite],
            [entry.lower for entry in finite],
            [entry.upper for entry in finite],
            color=palette.model,
        )
    estimates = [entry.level for entry in levels]
    split = sum(period <= record for period in periods)
    if split:
        line(
            axes, periods[:split], estimates[:split], color=palette.model, dashes=FITTED
        )
    if split < len(periods):
        start = max(split - 1, 0)
        line(
            axes,
            periods[start:],
            estimates[start:],
            color=palette.model,
            dashes=PROJECTED,
        )
    if truth is not None:
        axes.plot(
            periods,
            list(truth),
            color=palette.accent,
            linewidth=REFERENCE_WIDTH,
            linestyle=REFERENCE,
            zorder=2,
        )
    points(axes, hazen_return_periods(record), maxima, color=palette.observed)
    axes.set_xscale("log")
    _plain_log_axis(axes.xaxis)
    axes.set_xlim(1.0, periods[-1] * 1.05)
    reference_line(
        axes,
        x=float(record),
        label=f"record: {record} {period_unit}s",
        color=palette.muted,
    )
    axis_labels(axes, f"Return period [{period_unit}s, log scale]", name)
    labels = [(periods[-1], estimates[-1], model_label, palette.model)]
    if finite and finite[-1] is levels[-1]:
        confidence = levels[-1].confidence
        labels.append(
            (periods[-1], levels[-1].upper, f"{confidence:.0%} interval", palette.model)
        )
    if truth is not None:
        labels.append((periods[-1], truth[-1], "true", palette.accent))
    direct_labels(axes, labels)
    if headline_period is None:
        eligible = [p for p in periods if p <= 10.0 * record]
        headline_period = eligible[-1] if eligible else periods[-1]
    key = min(range(len(periods)), key=lambda i: abs(periods[i] - headline_period))
    entry = levels[key]
    if math.isfinite(entry.lower) and math.isfinite(entry.upper):
        title = (
            f"The {entry.return_period:g}-{period_unit} level is {entry.level:.3g}; "
            f"its {entry.confidence:.0%} interval spans "
            f"{entry.lower:.3g} to {entry.upper:.3g}"
        )
    else:
        title = (
            f"The {entry.return_period:g}-{period_unit} level is {entry.level:.3g}; "
            "its upper confidence limit is unbounded"
        )
    method = {"delta": "delta-method", "profile": "profile-likelihood"}.get(
        levels[0].method, levels[0].method
    )
    title_block(
        axes,
        title,
        f"Markers: {record} maxima, one per {period_unit} · dashed: fit · "
        f"dotted: extrapolation · band: {method} interval",
    )
    return axes


@themed
def plot_mean_residual_life(
    mrl: Sequence[MeanResidualLifePoint],
    *,
    threshold: float | None = None,
    gpd: GPDFit | None = None,
    name: str = "Mean excess",
    ax: Any | None = None,
) -> Any:
    """Plot the mean-residual-life curve with its band and a chosen threshold.

    With a GPD fit at ``threshold`` the implied mean excess
    ``e(u) = (sigma + xi (u - u_0))/(1 - xi)`` is drawn above the threshold
    (``FITTED``), and the title counts how many empirical points leave their
    own confidence limits around that line.

    Args:
        mrl: Output of :func:`probstat.risk.mean_residual_life`.
        threshold: Optional chosen threshold ``u_0`` (accent reference line).
        gpd: Optional GPD fit at ``threshold`` (requires ``xi < 1``).
        name: Vertical-axis title.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If ``mrl`` has fewer than two points, or ``gpd`` is given
            without ``threshold`` or with ``xi >= 1``.

    Equation:
        [EQ-EVA-MRL-001].

    References:
        [REF-COLES-2001, Section 4.3], [REF-MATPLOTLIB].

    """
    if len(mrl) < 2:
        raise ValueError("at least two mean-residual-life points are required")
    if gpd is not None and (threshold is None or gpd.xi >= 1.0):
        raise ValueError("a GPD overlay needs a threshold and xi < 1")
    axes = new_axes(ax)
    palette = active_palette()
    us = [p.threshold for p in mrl]
    band(
        axes, us, [p.lower for p in mrl], [p.upper for p in mrl], color=palette.observed
    )
    line(axes, us, [p.mean_excess for p in mrl], color=palette.observed)
    labels = [(us[-1], mrl[-1].mean_excess, "empirical e(u)", palette.observed)]
    title = (
        f"Mean excess moves from {mrl[0].mean_excess:.3g} to "
        f"{mrl[-1].mean_excess:.3g} across thresholds"
    )
    if threshold is not None:
        above = [p for p in mrl if p.threshold >= threshold]
        count = f" ({gpd.sample_size} excesses)" if gpd is not None else ""
        reference_line(axes, x=threshold, label=f"u = {threshold:.3g}{count}")
        if gpd is not None and above:

            def implied(u: float) -> float:
                """Return the GPD mean excess above ``u_0``.

                References:
                    [EQ-EVA-MRL-001], [REF-COLES-2001, Section 4.3].

                """
                return (gpd.sigma + gpd.xi * (u - threshold)) / (1.0 - gpd.xi)

            grid = [threshold, us[-1]]
            line(
                axes,
                grid,
                [implied(u) for u in grid],
                color=palette.model,
                dashes=FITTED,
            )
            labels.append(
                (us[-1], implied(us[-1]), f"GPD line, xi = {gpd.xi:.2f}", palette.model)
            )
            outside = sum(
                not (p.lower <= implied(p.threshold) <= p.upper) for p in above
            )
            if outside == 0:
                title = (
                    f"Above u = {threshold:.3g} the mean excess stays within its "
                    "limits of the fitted GPD line"
                )
            else:
                title = (
                    f"{outside} of {len(above)} thresholds above u = {threshold:.3g} "
                    "leave their limits of the GPD line"
                )
    axis_labels(axes, "Threshold u", name)
    direct_labels(axes, labels)
    title_block(
        axes,
        title,
        "Line and band: empirical mean excess with 95% normal-theory limits",
    )
    return axes


def order_statistic_band(
    count: int, confidence: float = 0.95
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return pointwise probability limits of uniform order statistics.

    The ``i``-th smallest of ``count`` independent uniforms follows
    ``Beta(i, count - i + 1)``; its central ``confidence`` interval, mapped
    through a model quantile function, gives a pointwise Q-Q band.

    Args:
        count: Sample size (positive).
        confidence: Central probability strictly inside ``(0, 1)``.

    Returns:
        Lower and upper probability limits for each order statistic.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-VIZ-QQBAND-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.4].

    """
    if count < 1:
        raise ValueError("count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    tail = 0.5 * (1.0 - confidence)
    laws = [Beta(i, count - i + 1) for i in range(1, count + 1)]
    return (
        tuple(law.quantile(tail) for law in laws),
        tuple(law.quantile(1.0 - tail) for law in laws),
    )


@themed
def plot_tail_quantiles(
    values: Sequence[float],
    distribution: Any,
    *,
    confidence: float = 0.95,
    model_label: str = "fitted GPD",
    name: str = "excess",
    ax: Any | None = None,
) -> Any:
    """Plot a Q-Q diagnostic of a tail sample with a pointwise model band.

    Points are ``(F^-1(p_i), x_(i))`` with Hazen positions. The band maps the
    central interval of each uniform order statistic through the model
    quantile function, so a point outside it is unusual for its rank.

    Args:
        values: Tail sample (excesses or block maxima).
        distribution: Fitted law exposing ``quantile``.
        confidence: Pointwise band level.
        model_label: Direct label of the model line.
        name: Quantity name for the axis titles.
        ax: Optional axes.

    Returns:
        Populated axes.

    Raises:
        ValueError: If fewer than three values are supplied.

    Equation:
        [EQ-INF-GOF-QQ-001], [EQ-VIZ-QQBAND-001].

    References:
        [REF-WILK-GNANADESIKAN-1968], [REF-COLES-2001, Section 4.3],
        [REF-MATPLOTLIB].

    """
    pairs = qq_points(values, distribution)
    if len(pairs) < 3:
        raise ValueError("at least three values are required")
    axes = new_axes(ax, figsize=(6.4, 5.2))
    palette = active_palette()
    model_q = [pair[0] for pair in pairs]
    sample_q = [pair[1] for pair in pairs]
    low_p, high_p = order_statistic_band(len(pairs), confidence)
    lower = [distribution.quantile(p) for p in low_p]
    upper = [distribution.quantile(p) for p in high_p]
    band(axes, model_q, lower, upper, color=palette.model)
    line(axes, model_q, model_q, color=palette.model, width=1.4)
    outside = [
        i for i in range(len(pairs)) if not lower[i] <= sample_q[i] <= upper[i]
    ]
    flagged = set(outside)
    inside = [i for i in range(len(pairs)) if i not in flagged]
    dense = len(pairs) > 150
    axes.scatter(
        [model_q[i] for i in inside],
        [sample_q[i] for i in inside],
        s=9.0 if dense else 18.0,
        color=palette.observed,
        alpha=0.75 if dense else 1.0,
        edgecolors=palette.canvas if not dense else "none",
        linewidths=0.0 if dense else 1.0,
        zorder=4,
    )
    if outside:
        points(
            axes,
            [model_q[i] for i in outside],
            [sample_q[i] for i in outside],
            color=palette.observed,
            size=42.0,
            zorder=5.0,
        )
    axes.grid(axis="both", color=palette.rule, linewidth=0.6)
    axis_labels(axes, f"Model quantile of {name}", f"Ordered {name}")
    direct_labels(
        axes,
        [(model_q[-1], model_q[-1], model_label, palette.model)],
    )
    largest_gap = sample_q[-1] / model_q[-1] - 1.0 if model_q[-1] else math.nan
    title_block(
        axes,
        f"{len(outside)} of {len(pairs)} tail points fall outside the "
        f"{model_label}'s {confidence:.0%} band",
        f"Largest {name}: {sample_q[-1]:.3g} vs model {model_q[-1]:.3g} "
        f"({largest_gap:+.0%}). Band: pointwise order-statistic limits",
    )
    return axes
