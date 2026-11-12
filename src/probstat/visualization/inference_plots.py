"""Plots for posterior sampling, sequential filtering, and surrogate accuracy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from probstat.statistics import quantile

from ._layout import halo
from .theme import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    CATEGORY_ORDERS,
    PATH_WIDTH,
    active_palette,
    axis_labels,
    axis_title,
    band,
    direct_labels,
    entity_colour,
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
    signal non-convergence, which the reported R-hat quantifies. With more
    than one chain each line carries a direct label (``chain 1``, ...) at
    its end, a second cue beside the hue for readers who cannot tell two
    series hues apart.

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
    ends: list[tuple[float, float, str, str]] = []
    for index, chain in enumerate(chains):
        label = f"chain {index + 1}"
        # Chains 1-4 have fixed hues in the set contract (ADR-0098), away from
        # the observed and model hues; further chains reuse the slots.
        member = CATEGORY_ORDERS["chains"]
        colour = entity_colour(member[index % len(member)])
        axes.plot(
            range(len(chain)), chain, color=colour, linewidth=PATH_WIDTH,
            alpha=0.75, label=label,
        )  # fmt: skip
        if len(chains) > 1 and len(chain) > 0:
            ends.append((float(len(chain) - 1), float(chain[-1]), label, colour))
    if ends:
        # Labels sit right of the frame, clear of the traces and of the
        # reference line; the reference label joins them so none collide.
        last = max(len(chain) for chain in chains) - 1.0
        axes.set_xlim(0.0, max(last, 1.0))
        if truth is not None:
            reference_line(axes, y=truth, entity="truth")
            ends.append((last, float(truth), "truth", entity_colour("truth")))
    elif truth is not None:
        reference_line(axes, y=truth, label="truth", entity="truth")
    axis_labels(axes, "Iteration after warm-up", name)
    direct_labels(axes, ends)
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
    for index, (axes, (name, values)) in enumerate(
        zip(axes_list[0], draws.items(), strict=True)
    ):
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
            path_effects=halo(),
            zorder=6,
        )
        if truth and name in truth:
            # The truth line spans the histogram only, so it stops below the
            # interval bar and its label instead of running through them.
            marker = reference_line(axes, x=truth[name], label=None, entity="truth")
            marker.set_ydata([0.0, 1.0 / 1.25])
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
                color=entity_colour("truth"),
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": palette.canvas,
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
                zorder=6,
            )
        axes.set_ylim(0.0, top * 1.25)
        # The same axes as the study's posterior marginals (ADR-0098).
        axis_labels(axes, name, axis_title("density") if index == 0 else None)
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
    lower = [m - 1.96 * s for m, s in zip(estimate, standard_deviation, strict=True)]
    upper = [m + 1.96 * s for m, s in zip(estimate, standard_deviation, strict=True)]
    band(axes, times, lower, upper, entity="filtered mean", alpha=BAND_ALPHA)
    labels: list[tuple[float, float, str, str]] = []
    if truth is not None:
        line(axes, times, truth, entity="truth", zorder=2)
        labels.append((times[-1], truth[-1], "truth", entity_colour("truth")))
    line(axes, times, estimate, entity="filtered mean")
    labels.append(
        (times[-1], estimate[-1], "filtered mean", entity_colour("filtered mean"))
    )
    if smoothed is not None:
        line(axes, times, smoothed[0], entity="smoothed")
        labels.append(
            (times[-1], smoothed[0][-1], "smoothed", entity_colour("smoothed"))
        )
    if observations is not None:
        kept = [
            (t, y) for t, y in zip(times, observations, strict=True) if y is not None
        ]
        if kept:
            points(
                axes,
                [t for t, _ in kept],
                [float(y) for _, y in kept],
                entity="observations",
                size=18.0,
            )
    axis_labels(axes, axis_title("time"), name)
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
        (
            "Dots: measurements. "
            if observations is not None
            else f"{name[:1].upper()}{name[1:]} is never measured. "
        )
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
