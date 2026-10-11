"""Plots for posterior sampling, sequential filtering, and surrogate accuracy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from probstat.statistics import quantile

from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    FITTED,
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
def plot_chain_traces(
    chains: Sequence[Sequence[float]],
    *,
    name: str,
    rhat: float | None = None,
    truth: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot MCMC traces of one parameter, one line per chain.

    Well-mixed chains overlap into one band; separated levels or slow drifts
    signal non-convergence, which the reported R-hat quantifies.

    Args:
        chains: Draws of the parameter, one sequence per chain.
        name: Parameter name.
        rhat: Optional rank-normalized R-hat to report.
        truth: Optional known value (synthetic studies only).
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-MCMC-RHAT-002].

    References:
        [REF-VEHTARI-ETAL-2021], [REF-MATPLOTLIB].

    """
    if not chains:
        raise ValueError("at least one chain is required")
    axes = new_axes(ax, figsize=(7.2, 3.4))
    palette = active_palette()
    for index, chain in enumerate(chains):
        colour = palette.series[index % len(palette.series)]
        axes.plot(range(len(chain)), chain, color=colour, linewidth=0.7, alpha=0.75)
    if truth is not None:
        reference_line(axes, y=truth, label="true value")
    axis_labels(axes, "Iteration (after warm-up)", name)
    verdict = ""
    if rhat is not None:
        verdict = (
            f": chains agree (R-hat {rhat:.3f})"
            if rhat < 1.01
            else f": chains disagree (R-hat {rhat:.3f})"
        )
    title_block(
        axes,
        f"Trace of {name}{verdict}",
        f"{len(chains)} chains overlaid; overlap means the sampler forgot its start",
    )
    return axes


@themed
def plot_posterior_marginals(
    draws: Mapping[str, Sequence[float]],
    *,
    truth: Mapping[str, float] | None = None,
    level: float = 0.9,
) -> Any:
    """Plot posterior marginal histograms as small multiples.

    Args:
        draws: Parameter name to pooled posterior draws (at most four).
        truth: Optional known values (synthetic studies only).
        level: Central credible interval level marked on each panel.

    Returns:
        The figure.

    Raises:
        ValueError: If no parameter or more than four are given.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 2.3], [REF-MATPLOTLIB].

    """
    if not draws or len(draws) > 4:
        raise ValueError("between one and four parameters are supported")
    plt = pyplot()
    palette = active_palette()
    figure, axes_list = plt.subplots(
        1, len(draws), figsize=(3.3 * len(draws) + 0.6, 3.4), squeeze=False
    )
    tail = 0.5 * (1.0 - level)
    for axes, (name, values) in zip(axes_list[0], draws.items(), strict=True):
        style_axes(axes)
        axes.hist(
            values,
            bins=40,
            density=True,
            color=palette.model,
            alpha=0.85,
            edgecolor=palette.canvas,
            linewidth=0.6,
        )
        low, mid, high = (quantile(values, p) for p in (tail, 0.5, 1.0 - tail))
        top = axes.get_ylim()[1]
        axes.hlines(top * 1.05, low, high, color=palette.ink, linewidth=2.0)
        axes.plot([mid], [top * 1.05], "o", color=palette.ink, markersize=4)
        axes.annotate(
            f"{mid:.4g} [{low:.4g}, {high:.4g}]",
            (mid, top * 1.05),
            xytext=(0.0, 6.0),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOTATION_SIZE,
            color=palette.ink,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": palette.canvas,
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=6,
        )
        if truth and name in truth:
            reference_line(axes, x=truth[name], label=None)
            axes.annotate(
                f"true {truth[name]:.4g}",
                xy=(truth[name], 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(4.0, 4.0),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
                fontweight="semibold",
                color=palette.accent,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": palette.canvas,
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
                zorder=6,
            )
        axes.set_ylim(0.0, top * 1.25)
        axes.set_yticks([])
        axes.set_xlabel(name)
    title_block(
        axes_list[0][0],
        "Posterior marginals of the calibrated parameters",
        f"Bar: median and {level:.0%} credible interval",
    )
    figure.tight_layout()
    return figure


@themed
def plot_filter_track(
    times: Sequence[float],
    estimate: Sequence[float],
    standard_deviation: Sequence[float],
    *,
    observations: Sequence[float | None] | None = None,
    truth: Sequence[float] | None = None,
    smoothed: tuple[Sequence[float], Sequence[float]] | None = None,
    name: str = "state",
    ax: Any | None = None,
) -> Any:
    """Plot a filtered state with its uncertainty band against data and truth.

    Args:
        times: Time stamps.
        estimate: Filtered mean of one state component.
        standard_deviation: Filtered standard deviation of that component.
        observations: Optional measurements (``None`` where missing).
        truth: Optional true trajectory (simulation studies).
        smoothed: Optional smoothed mean and standard deviation.
        name: State-component name.
        ax: Optional axes.

    Returns:
        Populated axes.

    Equation:
        [EQ-LINALG-007].

    References:
        [REF-SARKKA-2013], [REF-MATPLOTLIB].

    """
    axes = new_axes(ax)
    palette = active_palette()
    lower = [m - 1.96 * s for m, s in zip(estimate, standard_deviation, strict=True)]
    upper = [m + 1.96 * s for m, s in zip(estimate, standard_deviation, strict=True)]
    band(axes, times, lower, upper, color=palette.model, alpha=BAND_ALPHA)
    labels: list[tuple[float, float, str, str]] = []
    if truth is not None:
        line(axes, times, truth, color=palette.muted, width=1.4, zorder=2)
        labels.append((times[-1], truth[-1], "truth", palette.muted))
    line(axes, times, estimate, color=palette.model)
    labels.append((times[-1], estimate[-1], "filtered mean", palette.model))
    if smoothed is not None:
        line(
            axes, times, smoothed[0], color=palette.series[2], dashes=FITTED, width=1.6
        )
        labels.append((times[-1], smoothed[0][-1], "smoothed", palette.series[2]))
    if observations is not None:
        kept = [
            (t, y) for t, y in zip(times, observations, strict=True) if y is not None
        ]
        if kept:
            points(
                axes,
                [t for t, _ in kept],
                [float(y) for _, y in kept],
                color=palette.observed,
                size=18.0,
            )
    axis_labels(axes, "Time", name)
    direct_labels(axes, labels)
    coverage = ""
    if truth is not None:
        inside = sum(
            lo <= value <= hi for lo, value, hi in zip(lower, truth, upper, strict=True)
        )
        coverage = f"; truth inside the 95% band at {inside / len(truth):.0%} of steps"
    title_block(
        axes,
        f"The filter tracks {name} from noisy data{coverage}",
        ("Dots: measurements. " if observations is not None else "Not measured. ")
        + "Band: filtered 95% interval",
    )
    return axes


@themed
def plot_surrogate_parity(
    reference: Sequence[float],
    predicted: Sequence[float],
    *,
    name: str = "y",
    label: str = "surrogate",
    ax: Any | None = None,
) -> Any:
    """Plot surrogate predictions against true model outputs.

    Args:
        reference: True model outputs on validation inputs.
        predicted: Surrogate predictions at the same inputs.
        name: QoI name.
        label: Surrogate description.
        ax: Optional axes.

    Returns:
        Populated axes with the identity line as reference.

    References:
        [REF-SUDRET-2007, Chapter 4], [REF-MATPLOTLIB].

    """
    if len(reference) != len(predicted) or len(reference) < 2:
        raise ValueError("reference and predicted must align with at least two points")
    axes = new_axes(ax, figsize=(5.4, 5.0))
    palette = active_palette()
    low = min(min(reference), min(predicted))
    high = max(max(reference), max(predicted))
    axes.plot(
        [low, high],
        [low, high],
        color=palette.accent,
        linewidth=1.2,
        linestyle=(0, (3.5, 3.0)),
    )
    points(axes, reference, predicted, color=palette.observed, size=14.0)
    mean = math.fsum(reference) / len(reference)
    residual = math.fsum(
        (r - p) ** 2 for r, p in zip(reference, predicted, strict=True)
    )
    total = math.fsum((r - mean) ** 2 for r in reference) or 1.0
    error = residual / total
    axes.set_aspect("equal")
    axes.grid(axis="both", color=palette.rule, linewidth=0.6)
    axis_labels(axes, f"Model {name}", f"{label} {name}")
    title_block(
        axes,
        f"The {label} explains {1.0 - error:.2%} of the output variance",
        f"Validation points off the training design; relative error {error:.2e}",
    )
    return axes
