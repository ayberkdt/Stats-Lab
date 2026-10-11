"""Alternative encodings of the same uncertainty.

Different readers decode uncertainty best through different encodings:
nested interval bands (fan charts), individual realizations, discrete
frequency framings (quantile dot plots), and distribution functions with
their own sampling uncertainty. These plots let a lab show one quantity in
each form under the shared theme.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.foundations.validation import coerce_finite_sample
from probstat.statistics import quantile

from .theme import (
    ANNOTATION_SIZE,
    active_palette,
    axis_labels,
    line,
    new_axes,
    reference_line,
    themed,
    title_block,
)


def dkw_epsilon(size: int, confidence: float = 0.95) -> float:
    """Return the Dvoretzky-Kiefer-Wolfowitz band half-width.

    ``P(sup |F_n - F| > eps) <= 2 exp(-2 n eps^2)`` with Massart's tight
    constant, so ``eps = sqrt(ln(2 / alpha) / (2 n))`` gives a distribution-free
    simultaneous band of level ``1 - alpha``.

    Args:
        size: Sample size ``n``.
        confidence: Band level ``1 - alpha``.

    Returns:
        Half-width ``eps``.

    Raises:
        ValueError: If ``size < 1`` or ``confidence`` is not in ``(0, 1)``.

    Equation:
        [EQ-STAT-DKW-001].

    References:
        [REF-DVORETZKY-KIEFER-WOLFOWITZ-1956], [REF-MASSART-1990].

    """
    if size < 1:
        raise ValueError("size must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    return math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * size))


@themed
def plot_fan_chart(
    times: Sequence[float],
    paths: Sequence[Sequence[float]],
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
    name: str = "value",
    ax: Any | None = None,
) -> Any:
    """Plot nested central quantile bands of many paths (a fan chart).

    Args:
        times: Common time grid.
        paths: Sample paths on ``times``.
        levels: Central band probabilities, narrowest first.
        name: Quantity name.
        ax: Optional axes.

    Returns:
        Populated axes with the median line and labelled bands.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-SPIEGELHALTER-PEARSON-SHORT-2011], [REF-MATPLOTLIB].

    """
    if not paths:
        raise ValueError("at least one path is required")
    columns = list(zip(*paths, strict=True))
    axes = new_axes(ax)
    palette = active_palette()
    ordered = sorted(levels, reverse=True)
    for rank, level in enumerate(ordered):
        tail = 0.5 * (1.0 - level)
        low = [quantile(column, tail) for column in columns]
        high = [quantile(column, 1.0 - tail) for column in columns]
        axes.fill_between(
            times,
            low,
            high,
            color=palette.model,
            alpha=0.12 + 0.1 * rank,
            linewidth=0.0,
        )
        axes.annotate(
            f"{level:.0%}",
            (times[-1], high[-1]),
            xytext=(6.0, 0.0),
            textcoords="offset points",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.model,
            fontweight="semibold",
            annotation_clip=False,
        )
    median = [quantile(column, 0.5) for column in columns]
    line(axes, times, median, color=palette.model)
    axis_labels(axes, "Time", name)
    title_block(
        axes,
        f"Half of the paths stay inside the darkest band; {max(levels):.0%} "
        "inside the palest",
        f"Fan chart of {len(paths)} paths: median line and nested central bands",
    )
    return axes


@themed
def plot_spaghetti(
    times: Sequence[float],
    paths: Sequence[Sequence[float]],
    *,
    highlight: int = 3,
    name: str = "value",
    ax: Any | None = None,
) -> Any:
    """Plot individual realizations with a few highlighted.

    Args:
        times: Common time grid.
        paths: Sample paths.
        highlight: Number of paths drawn in series colours.
        name: Quantity name.
        ax: Optional axes.

    Returns:
        Populated axes.

    References:
        [REF-SPIEGELHALTER-PEARSON-SHORT-2011], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    for values in paths[highlight:]:
        axes.plot(times, values, color=palette.muted, linewidth=0.6, alpha=0.35)
    for index, values in enumerate(paths[:highlight]):
        line(axes, times, values, color=palette.series[index], width=1.6)
    axis_labels(axes, "Time", name)
    title_block(
        axes,
        "Each line is one possible future; no single one is the forecast",
        f"{len(paths)} simulated paths, {highlight} highlighted",
    )
    return axes


@themed
def plot_quantile_dots(
    values: Sequence[float],
    *,
    dots: int = 50,
    threshold: float | None = None,
    name: str = "value",
    ax: Any | None = None,
) -> Any:
    """Plot a quantile dot plot: ``dots`` equally likely outcomes.

    Each dot is one ``1/dots`` probability slice (the ``(k - 1/2)/dots``
    quantile), stacked into bins, so a reader can count outcomes ("4 in 50")
    instead of judging areas. With a threshold, exceeding dots use the data
    hue and the rest stay muted; bin edges are aligned with the threshold.

    Args:
        values: Sample from the predictive distribution.
        dots: Number of equally likely outcomes shown.
        threshold: Optional limit; dots above it are highlighted.
        name: Quantity name.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-KAY-ETAL-2016], [REF-HYNDMAN-FAN-1996].

    """
    sample = coerce_finite_sample(values, name="values")
    if len(sample) < 2 or dots < 2:
        raise ValueError("need at least two values and two dots")
    points = [quantile(sample, (k + 0.5) / dots) for k in range(dots)]
    low, high = points[0], points[-1]
    width = (high - low) / max(8, int(math.sqrt(dots) * 2)) or 1.0
    if threshold is not None:
        # Align a bin edge with the threshold so no dot straddles the limit.
        low = threshold - math.ceil((threshold - low) / width) * width
    stacks: dict[int, int] = {}
    axes = new_axes(ax)
    palette = active_palette()
    exceed = 0
    for value in points:
        slot = int((value - low) // width)
        level = stacks.get(slot, 0)
        stacks[slot] = level + 1
        hot = threshold is not None and value > threshold
        exceed += hot
        axes.scatter(
            [low + (slot + 0.5) * width],
            [level + 0.5],
            s=90.0,
            color=palette.observed if hot or threshold is None else palette.rule,
            edgecolors=palette.canvas,
            linewidths=1.0,
            zorder=3,
        )
    axes.set_ylim(0.0, max(stacks.values()) + 1.0)
    axes.set_yticks([])
    axes.spines["left"].set_visible(False)
    axes.grid(False)
    if threshold is not None:
        reference_line(axes, x=threshold, label=f"limit {threshold:g}")
    axis_labels(axes, name)
    headline = (
        f"{exceed} of {dots} equally likely outcomes exceed the limit"
        if threshold is not None
        else f"{dots} equally likely outcomes"
    )
    title_block(axes, headline, f"Quantile dot plot: each dot is a 1-in-{dots} chance")
    return axes


@themed
def plot_ecdf_band(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    reference: Any | None = None,
    name: str = "value",
    ax: Any | None = None,
) -> Any:
    """Plot an empirical CDF with a distribution-free DKW confidence band.

    Args:
        values: Sample.
        confidence: Simultaneous band level.
        reference: Optional object with ``cdf`` drawn as a fitted curve.
        name: Quantity name.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-STAT-005], [EQ-STAT-DKW-001].

    References:
        [REF-MASSART-1990], [REF-WASSERMAN-2004, Section 7.1].

    """
    sample = sorted(coerce_finite_sample(values, name="values"))
    n = len(sample)
    if n < 2:
        raise ValueError("at least two values are required")
    eps = dkw_epsilon(n, confidence)
    heights = [(index + 1) / n for index in range(n)]
    axes = new_axes(ax)
    palette = active_palette()
    axes.fill_between(
        sample,
        [max(0.0, h - eps) for h in heights],
        [min(1.0, h + eps) for h in heights],
        step="post",
        color=palette.observed,
        alpha=0.16,
        linewidth=0.0,
    )
    axes.step(sample, heights, where="post", color=palette.observed, linewidth=2.0)
    if reference is not None:
        grid = [sample[0] + (sample[-1] - sample[0]) * k / 200 for k in range(201)]
        line(
            axes,
            grid,
            [reference.cdf(x) for x in grid],
            color=palette.model,
            dashes=(0.0, (5.0, 3.5)),
        )
    axes.set_ylim(0.0, 1.04)
    axis_labels(axes, name, "Cumulative probability")
    title_block(
        axes,
        f"With n = {n}, the true CDF lies within +/-{eps:.3f} everywhere "
        f"({confidence:.0%} band)",
        "Steps: empirical CDF. Band: Dvoretzky-Kiefer-Wolfowitz (Massart)",
    )
    return axes
