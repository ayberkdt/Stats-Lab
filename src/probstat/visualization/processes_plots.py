"""Plots for random fields and their Karhunen-Loeve expansions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .theme import (
    ANNOTATION_SIZE,
    active_palette,
    axis_labels,
    band,
    line,
    new_axes,
    themed,
    title_block,
)


@themed
def plot_field_realizations(
    locations: Sequence[float],
    realizations: Sequence[Sequence[float]],
    *,
    mean: float = 0.0,
    standard_deviation: Sequence[float] | None = None,
    highlight: int = 2,
    name: str = "field",
    ax: Any | None = None,
) -> Any:
    """Plot random-field realizations with a two-sigma pointwise band.

    Most realizations are muted context; ``highlight`` of them are drawn in
    the data hue so individual sample paths stay readable.

    Args:
        locations: Spatial coordinates (ascending).
        realizations: Field values, one sequence per realization.
        mean: Field mean.
        standard_deviation: Optional pointwise standard deviation.
        highlight: Number of realizations to emphasize.
        name: Field name for the axis title.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-PROC-KL-001].

    References:
        [REF-GHANEM-SPANOS-1991], [REF-MATPLOTLIB].

    """
    if not realizations:
        raise ValueError("at least one realization is required")
    axes = new_axes(ax)
    palette = active_palette()
    if standard_deviation is not None:
        band(
            axes,
            locations,
            [mean - 2.0 * s for s in standard_deviation],
            [mean + 2.0 * s for s in standard_deviation],
            color=palette.observed,
            alpha=0.12,
        )
    for values in realizations[highlight:]:
        axes.plot(locations, values, color=palette.muted, linewidth=0.7, alpha=0.45)
    for index, values in enumerate(realizations[:highlight]):
        line(axes, locations, values, color=palette.series[index], width=1.8)
    axes.axhline(mean, color=palette.ink, linewidth=0.8, alpha=0.6)
    axis_labels(axes, "Location", name)
    title_block(
        axes,
        f"{len(realizations)} realizations of one Gaussian random field",
        "Band: mean +/- 2 pointwise standard deviations; two paths highlighted",
    )
    return axes


@themed
def plot_kl_spectrum(
    spectrum: Sequence[float],
    kept: int,
    *,
    show: int | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot the Karhunen-Loeve eigenvalue spectrum and the truncation point.

    Args:
        spectrum: All eigenvalues in descending order.
        kept: Number of retained modes.
        show: Number of leading modes to draw (default ``min(len, 3 kept)``).
        ax: Optional axes.

    Returns:
        Populated axes showing each mode's share of the total variance.

    Equation:
        [EQ-PROC-KL-002].

    References:
        [REF-GHANEM-SPANOS-1991], [REF-MATPLOTLIB].

    """
    total = math.fsum(value for value in spectrum if value > 0.0)
    if total <= 0.0:
        raise ValueError("spectrum must contain positive eigenvalues")
    count = min(len(spectrum), show or max(3 * kept, kept + 5))
    shares = [max(spectrum[k], 0.0) / total for k in range(count)]
    axes = new_axes(ax)
    palette = active_palette()
    colours = [palette.observed if k < kept else palette.rule for k in range(count)]
    axes.bar(
        range(1, count + 1),
        shares,
        color=colours,
        width=0.7,
        edgecolor=palette.canvas,
        linewidth=0.8,
    )
    explained = math.fsum(shares[:kept])
    axes.annotate(
        f"{kept} modes keep {explained:.1%} of the variance",
        xy=(kept + 0.5, shares[min(kept, count - 1)]),
        xytext=(24.0, 40.0),
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        color=palette.ink,
        arrowprops={
            "arrowstyle": "-|>,head_length=0.35,head_width=0.18",
            "color": palette.muted,
            "linewidth": 0.8,
        },
    )
    axis_labels(axes, "Mode k", "Share of field variance")
    title_block(
        axes,
        f"The field compresses to {kept} uncertain coefficients",
        "Teal: retained Karhunen-Loeve modes. Grey: truncated modes",
    )
    return axes
